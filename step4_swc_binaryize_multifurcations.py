import shutil
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


CONFIG = {
    "input_dirs": [
        {
            "name": "soma_pruned",
            "path": "/data/project_backup/swc_fix/step3_out",
        },
        {
            "name": "no_soma",
            "path": "/data/project_backup/swc_fix/step3_input_mannual_check/merge/no_soma",
        },
        {
            "name": "no_soma_annotation",
            "path": "/data/project_backup/swc_fix/step3_input_mannual_check/merge/no_soma_annotation",
        },
    ],

    "output_dir": "/data/project_backup/swc_fix/step4_out",

    "max_processes": 30,
    "chunksize": 10,

    "swc_scale": [1000, 1000, 1000],

    "target_node_type": 5,
    "new_bifurcation_type": 5,

    "anchor_excluded_types": [1, 6],
    "anchor_max_hops": 8,
    "anchor_max_distance_um": 8.0,

    "direction_max_hops": 8,
    "direction_max_distance_um": 8.0,

    "proximity_weight": 0.45,
    "angular_weight": 0.35,
    "length_weight": 0.15,
    "crossover_weight": 0.05,

    "check_crossover": False,
    "crossover_tolerance_um": 0.05,

    "preserve_subdirs": True,
    "clear_output": True,
}


SWC_SCALE = np.asarray(CONFIG["swc_scale"], dtype=float)
GLOBAL_OUTPUT_DIR = None


def clear_output_dir(path):
    path = Path(path)

    if path.exists():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)

    path.mkdir(parents=True, exist_ok=True)


def init_worker(output_dir):
    global GLOBAL_OUTPUT_DIR
    GLOBAL_OUTPUT_DIR = Path(output_dir)


def read_swc(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
    )

    if df.shape[1] < 7:
        raise ValueError("SWC must contain at least seven columns")

    df = df.iloc[:, :7].copy()

    df.columns = [
        "ID",
        "Type",
        "X",
        "Y",
        "Z",
        "Radius",
        "Parent",
    ]

    df[["ID", "Type", "Parent"]] = df[
        ["ID", "Type", "Parent"]
    ].astype(int)

    if df["ID"].duplicated().any():
        raise ValueError("Duplicate node IDs")

    return df


def write_swc(df, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        sep=" ",
        index=False,
        header=False,
        float_format="%.3f",
    )


def unit_vector(v):
    n = float(np.linalg.norm(v))

    return (
        v / n
        if n > 1e-12
        else np.zeros(3, dtype=float)
    )


def branch_representative(
    start,
    source,
    child_map,
    coord_map,
):
    current = int(start)

    travelled = float(
        np.linalg.norm(
            coord_map[current]
            - coord_map[source]
        )
    )

    for _ in range(
        CONFIG["direction_max_hops"]
    ):
        children = [
            int(c)
            for c in child_map.get(
                current,
                [],
            )
        ]

        if len(children) != 1:
            break

        nxt = children[0]

        edge = float(
            np.linalg.norm(
                coord_map[nxt]
                - coord_map[current]
            )
        )

        limit = CONFIG[
            "direction_max_distance_um"
        ]

        if (
            edge > 0
            and travelled + edge > limit
        ):
            ratio = (
                max(
                    0.0,
                    limit - travelled,
                )
                / edge
            )

            return (
                coord_map[current]
                + ratio
                * (
                    coord_map[nxt]
                    - coord_map[current]
                )
            )

        current = nxt
        travelled += edge

        if travelled >= limit:
            break

    return coord_map[current].copy()


def subtree_nodes(root, child_map):
    stack = [int(root)]
    seen = set()

    while stack:
        node = stack.pop()

        if node in seen:
            continue

        seen.add(node)

        stack.extend(
            child_map.get(
                node,
                [],
            )
        )

    return seen


def collect_anchor_candidates(
    root,
    source,
    child_map,
    type_map,
    coord_map,
):
    excluded = set(
        map(
            int,
            CONFIG["anchor_excluded_types"],
        )
    )

    max_hops = int(
        CONFIG["anchor_max_hops"]
    )

    max_distance = float(
        CONFIG["anchor_max_distance_um"]
    )

    start_distance = float(
        np.linalg.norm(
            coord_map[root]
            - coord_map[source]
        )
    )

    stack = [
        (
            int(root),
            1,
            start_distance,
        )
    ]

    best = {}
    visited = set()

    while stack:
        node, hops, path_distance = stack.pop()

        state = (
            node,
            hops,
        )

        if state in visited:
            continue

        visited.add(state)

        if (
            hops > max_hops
            or path_distance > max_distance
        ):
            continue

        children = [
            int(c)
            for c in child_map.get(
                node,
                [],
            )
        ]

        if (
            type_map.get(node)
            not in excluded
            and len(children) == 1
        ):
            old = best.get(node)

            info = {
                "anchor": node,
                "root": int(root),
                "hops": hops,
                "path_distance": path_distance,
            }

            if (
                old is None
                or (
                    path_distance,
                    hops,
                )
                < (
                    old["path_distance"],
                    old["hops"],
                )
            ):
                best[node] = info

        for child in children:
            edge = float(
                np.linalg.norm(
                    coord_map[child]
                    - coord_map[node]
                )
            )

            stack.append(
                (
                    child,
                    hops + 1,
                    path_distance + edge,
                )
            )

    return (
        best,
        {
            node
            for node, _ in visited
        },
    )


def segment_distance(
    a0,
    a1,
    b0,
    b1,
):
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))

    den = a * c - b * b
    eps = 1e-12

    if a < eps and c < eps:
        return float(
            np.linalg.norm(
                a0 - b0
            )
        )

    if a < eps:
        t = np.clip(
            e / c,
            0.0,
            1.0,
        )

        return float(
            np.linalg.norm(
                a0
                - (
                    b0
                    + t * v
                )
            )
        )

    if c < eps:
        s = np.clip(
            -d / a,
            0.0,
            1.0,
        )

        return float(
            np.linalg.norm(
                (
                    a0
                    + s * u
                )
                - b0
            )
        )

    if den < eps:
        s = 0.0

        t = np.clip(
            e / c,
            0.0,
            1.0,
        )

    else:
        s = np.clip(
            (
                b * e
                - c * d
            )
            / den,
            0.0,
            1.0,
        )

        t = np.clip(
            (
                a * e
                - b * d
            )
            / den,
            0.0,
            1.0,
        )

    for _ in range(2):
        s = np.clip(
            (
                b * t
                - d
            )
            / a,
            0.0,
            1.0,
        )

        t = np.clip(
            (
                b * s
                + e
            )
            / c,
            0.0,
            1.0,
        )

    return float(
        np.linalg.norm(
            (
                a0
                + s * u
            )
            - (
                b0
                + t * v
            )
        )
    )


def local_edge_segments(
    local_nodes,
    parent_map,
    coord_map,
):
    edges = []

    local_nodes = set(
        local_nodes
    )

    for child in local_nodes:
        parent = parent_map.get(
            child,
            -1,
        )

        if (
            parent != -1
            and parent in coord_map
        ):
            edges.append(
                (
                    parent,
                    child,
                    coord_map[parent],
                    coord_map[child],
                )
            )

    for child, parent in parent_map.items():
        if (
            parent != -1
            and parent in local_nodes
            and child not in local_nodes
        ):
            edges.append(
                (
                    parent,
                    child,
                    coord_map[parent],
                    coord_map[child],
                )
            )

    return edges


def crossover_penalty(
    anchor,
    child,
    local_edges,
    coord_map,
):
    tolerance = float(
        CONFIG[
            "crossover_tolerance_um"
        ]
    )

    if tolerance <= 0:
        return 0.0

    a0 = coord_map[anchor]
    a1 = coord_map[child]

    penalty = 0.0

    for p, c, b0, b1 in local_edges:
        if child == c:
            continue

        if (
            anchor in (p, c)
            or child in (p, c)
        ):
            continue

        distance = segment_distance(
            a0,
            a1,
            b0,
            b1,
        )

        if distance < tolerance:
            penalty = max(
                penalty,
                1.0
                - distance
                / tolerance,
            )

    return penalty


def relink_type5_multifurcations(df):
    rows = {
        int(r.ID): {
            "ID": int(r.ID),
            "Type": int(r.Type),
            "X": float(r.X),
            "Y": float(r.Y),
            "Z": float(r.Z),
            "Radius": float(r.Radius),
            "Parent": int(r.Parent),
        }
        for r in df.itertuples(
            index=False
        )
    }

    original_ids = set(rows)

    original_order = (
        df["ID"]
        .astype(int)
        .tolist()
    )

    type_map = {
        i: r["Type"]
        for i, r in rows.items()
    }

    parent_map = {
        i: r["Parent"]
        for i, r in rows.items()
    }

    coord_map = {
        i: np.array(
            [
                r["X"],
                r["Y"],
                r["Z"],
            ],
            dtype=float,
        )
        / SWC_SCALE
        for i, r in rows.items()
    }

    child_map = defaultdict(list)

    for node_id in original_order:
        parent = parent_map[
            node_id
        ]

        if parent != -1:
            child_map[
                parent
            ].append(
                node_id
            )

    target_type = int(
        CONFIG["target_node_type"]
    )

    target_nodes = [
        i
        for i in original_order
        if type_map[i] == target_type
    ]

    records = []
    step_records = []

    def set_parent(
        child,
        parent,
    ):
        old_parent = parent_map[
            child
        ]

        if (
            old_parent != -1
            and child
            in child_map.get(
                old_parent,
                [],
            )
        ):
            child_map[
                old_parent
            ].remove(
                child
            )

        parent_map[
            child
        ] = int(parent)

        rows[
            child
        ]["Parent"] = int(
            parent
        )

        if (
            parent != -1
            and child
            not in child_map[parent]
        ):
            child_map[
                parent
            ].append(
                child
            )

    for source_id in target_nodes:
        children_before = list(
            child_map.get(
                source_id,
                [],
            )
        )

        if len(children_before) <= 2:
            continue

        source_coord = coord_map[
            source_id
        ]

        source_steps = []

        while (
            len(
                child_map.get(
                    source_id,
                    [],
                )
            )
            > 2
        ):
            direct_children = list(
                child_map[
                    source_id
                ]
            )

            candidate_pairs = []
            local_nodes = {
                source_id
            }

            anchor_by_root = {}

            for root in direct_children:
                (
                    candidates,
                    visited,
                ) = collect_anchor_candidates(
                    root,
                    source_id,
                    child_map,
                    type_map,
                    coord_map,
                )

                anchor_by_root[
                    root
                ] = candidates

                local_nodes.update(
                    visited
                )

            local_edges = (
                local_edge_segments(
                    local_nodes,
                    parent_map,
                    coord_map,
                )
            )

            distance_scale = max(
                float(
                    CONFIG[
                        "anchor_max_distance_um"
                    ]
                ),
                1e-12,
            )

            for move_child in direct_children:
                rep = (
                    branch_representative(
                        move_child,
                        source_id,
                        child_map,
                        coord_map,
                    )
                )

                original_direction = (
                    unit_vector(
                        rep
                        - source_coord
                    )
                )

                original_edge = float(
                    np.linalg.norm(
                        coord_map[
                            move_child
                        ]
                        - source_coord
                    )
                )

                for root in direct_children:
                    if root == move_child:
                        continue

                    for (
                        anchor,
                        info,
                    ) in anchor_by_root[
                        root
                    ].items():

                        if (
                            len(
                                child_map.get(
                                    anchor,
                                    [],
                                )
                            )
                            != 1
                        ):
                            continue

                        new_direction = (
                            unit_vector(
                                rep
                                - coord_map[
                                    anchor
                                ]
                            )
                        )

                        angular = (
                            0.5
                            * (
                                1.0
                                - float(
                                    np.clip(
                                        np.dot(
                                            original_direction,
                                            new_direction,
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )

                        proximity = (
                            info[
                                "path_distance"
                            ]
                            / distance_scale
                        )

                        new_edge = float(
                            np.linalg.norm(
                                coord_map[
                                    move_child
                                ]
                                - coord_map[
                                    anchor
                                ]
                            )
                        )

                        length = (
                            max(
                                0.0,
                                new_edge
                                - original_edge,
                            )
                            / distance_scale
                        )

                        crossing = (
                            crossover_penalty(
                                anchor,
                                move_child,
                                local_edges,
                                coord_map,
                            )
                            if (
                                CONFIG[
                                    "check_crossover"
                                ]
                                and CONFIG[
                                    "crossover_weight"
                                ]
                                > 0
                            )
                            else 0.0
                        )

                        cost = (
                            CONFIG[
                                "proximity_weight"
                            ]
                            * proximity
                            + CONFIG[
                                "angular_weight"
                            ]
                            * angular
                            + CONFIG[
                                "length_weight"
                            ]
                            * length
                            + CONFIG[
                                "crossover_weight"
                            ]
                            * crossing
                        )

                        candidate_pairs.append(
                            {
                                "cost": float(
                                    cost
                                ),
                                "move_child": int(
                                    move_child
                                ),
                                "anchor": int(
                                    anchor
                                ),
                                "anchor_root": int(
                                    root
                                ),
                                "anchor_hops": int(
                                    info[
                                        "hops"
                                    ]
                                ),
                                "anchor_path_distance_um": float(
                                    info[
                                        "path_distance"
                                    ]
                                ),
                                "original_edge_um": original_edge,
                                "new_edge_um": new_edge,
                                "angular_cost": float(
                                    angular
                                ),
                                "length_cost": float(
                                    length
                                ),
                                "crossover_cost": float(
                                    crossing
                                ),
                            }
                        )

            if not candidate_pairs:
                raise ValueError(
                    f"No eligible nearby node can resolve "
                    f"Type-5 node {source_id} with "
                    f"{len(child_map.get(source_id, []))} "
                    f"direct children"
                )

            best = min(
                candidate_pairs,
                key=lambda x: (
                    x["cost"],
                    x[
                        "anchor_path_distance_um"
                    ],
                    x["new_edge_um"],
                    x["move_child"],
                    x["anchor"],
                ),
            )

            move_child = best[
                "move_child"
            ]

            anchor = best[
                "anchor"
            ]

            anchor_old_type = (
                type_map[
                    anchor
                ]
            )

            set_parent(
                move_child,
                anchor,
            )

            type_map[
                anchor
            ] = int(
                CONFIG[
                    "new_bifurcation_type"
                ]
            )

            rows[
                anchor
            ]["Type"] = int(
                CONFIG[
                    "new_bifurcation_type"
                ]
            )

            step = {
                "type5_node": int(
                    source_id
                ),
                "step": (
                    len(
                        source_steps
                    )
                    + 1
                ),
                "moved_child": move_child,
                "old_parent": int(
                    source_id
                ),
                "new_parent": anchor,
                "anchor_root": best[
                    "anchor_root"
                ],
                "anchor_hops": best[
                    "anchor_hops"
                ],
                "anchor_path_distance_um": best[
                    "anchor_path_distance_um"
                ],
                "anchor_original_type": int(
                    anchor_old_type
                ),
                "anchor_new_type": int(
                    CONFIG[
                        "new_bifurcation_type"
                    ]
                ),
                "original_edge_um": best[
                    "original_edge_um"
                ],
                "new_edge_um": best[
                    "new_edge_um"
                ],
                "angular_cost": best[
                    "angular_cost"
                ],
                "length_cost": best[
                    "length_cost"
                ],
                "crossover_cost": best[
                    "crossover_cost"
                ],
                "total_cost": best[
                    "cost"
                ],
            }

            source_steps.append(
                step
            )

            step_records.append(
                step
            )

        records.append(
            {
                "type5_node": int(
                    source_id
                ),
                "n_direct_children_before": len(
                    children_before
                ),
                "n_direct_children_after": len(
                    child_map.get(
                        source_id,
                        [],
                    )
                ),
                "children_before": ",".join(
                    map(
                        str,
                        children_before,
                    )
                ),
                "children_after": ",".join(
                    map(
                        str,
                        child_map.get(
                            source_id,
                            [],
                        ),
                    )
                ),
                "n_relinked_branches": len(
                    source_steps
                ),
                "moved_children": ",".join(
                    str(
                        x[
                            "moved_child"
                        ]
                    )
                    for x in source_steps
                ),
                "new_parent_nodes": ",".join(
                    str(
                        x[
                            "new_parent"
                        ]
                    )
                    for x in source_steps
                ),
            }
        )

    result = pd.DataFrame(
        [
            rows[i]
            for i in original_order
        ]
    )

    result = order_parent_first(
        result
    )

    validate_relinked(
        result,
        original_ids,
    )

    info = {
        "n_type5_nodes": len(
            target_nodes
        ),
        "n_type5_relinked": len(
            records
        ),
        "n_relinked_branches": len(
            step_records
        ),
        "n_nodes_before": len(df),
        "n_nodes_after": len(
            result
        ),
    }

    return (
        result,
        pd.DataFrame(records),
        pd.DataFrame(
            step_records
        ),
        info,
    )


def order_parent_first(df):
    ids = (
        df["ID"]
        .astype(int)
        .tolist()
    )

    id_set = set(ids)

    parent_map = dict(
        zip(
            df["ID"].astype(int),
            df["Parent"].astype(int),
        )
    )

    child_map = defaultdict(list)

    for node_id in ids:
        parent = parent_map[
            node_id
        ]

        if parent != -1:
            if parent not in id_set:
                raise ValueError(
                    f"Missing parent {parent} "
                    f"for node {node_id}"
                )

            child_map[
                parent
            ].append(
                node_id
            )

    roots = [
        i
        for i in ids
        if parent_map[i] == -1
    ]

    if not roots:
        raise ValueError(
            "No root node"
        )

    order = []
    seen = set()

    stack = list(
        reversed(
            roots
        )
    )

    while stack:
        node = stack.pop()

        if node in seen:
            raise ValueError(
                "Cycle or repeated traversal detected"
            )

        seen.add(node)
        order.append(node)

        stack.extend(
            reversed(
                child_map.get(
                    node,
                    [],
                )
            )
        )

    if len(order) != len(ids):
        missing = sorted(
            id_set
            - seen
        )

        raise ValueError(
            f"Cycle or unreachable nodes: "
            f"{missing[:10]}"
        )

    return (
        df.set_index(
            "ID",
            drop=False,
        )
        .loc[order]
        .reset_index(
            drop=True
        )
    )


def validate_relinked(
    df,
    original_ids,
):
    ids = set(
        df["ID"].astype(int)
    )

    if ids != original_ids:
        raise ValueError(
            "Original node set changed"
        )

    if df["ID"].duplicated().any():
        raise ValueError(
            "Duplicate node IDs after relinking"
        )

    type_map = dict(
        zip(
            df["ID"].astype(int),
            df["Type"].astype(int),
        )
    )

    child_map = defaultdict(list)

    for node_id, parent in zip(
        df["ID"].astype(int),
        df["Parent"].astype(int),
    ):
        if parent != -1:
            if parent not in ids:
                raise ValueError(
                    f"Missing parent {parent} "
                    f"for node {node_id}"
                )

            child_map[
                parent
            ].append(
                node_id
            )

    invalid = [
        (
            node_id,
            len(
                child_map.get(
                    node_id,
                    [],
                )
            ),
        )
        for (
            node_id,
            node_type,
        ) in type_map.items()
        if (
            node_type
            == int(
                CONFIG[
                    "target_node_type"
                ]
            )
            and len(
                child_map.get(
                    node_id,
                    [],
                )
            )
            > 2
        )
    ]

    if invalid:
        raise ValueError(
            f"Type-5 multifurcations remain: "
            f"{invalid[:10]}"
        )


def output_path_for(
    input_path,
    input_root,
    source_name,
):
    input_path = Path(
        input_path
    )

    input_root = Path(
        input_root
    )

    if CONFIG[
        "preserve_subdirs"
    ]:
        relative_path = (
            input_path.relative_to(
                input_root
            )
        )
    else:
        relative_path = Path(
            input_path.name
        )

    return (
        GLOBAL_OUTPUT_DIR
        / source_name
        / relative_path
    )


def process_one(task):
    (
        path,
        input_root,
        source_name,
    ) = task

    path = Path(path)
    input_root = Path(
        input_root
    )

    relative_path = (
        path.relative_to(
            input_root
        )
    )

    out_path = (
        output_path_for(
            path,
            input_root,
            source_name,
        )
    )

    try:
        df = read_swc(
            path
        )

        (
            cleaned,
            node_df,
            step_df,
            info,
        ) = relink_type5_multifurcations(
            df
        )

        write_swc(
            cleaned,
            out_path,
        )

        return {
            "source": source_name,
            "file": path.name,
            "relative_path": str(
                relative_path
            ),
            "output_path": str(
                out_path.relative_to(
                    GLOBAL_OUTPUT_DIR
                )
            ),
            **info,
            "node_change": (
                len(cleaned)
                - len(df)
            ),
            "fallback_copy": False,
            "node_records": (
                node_df.to_dict(
                    "records"
                )
            ),
            "step_records": (
                step_df.to_dict(
                    "records"
                )
            ),
        }

    except Exception:
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            path,
            out_path,
        )

        return {
            "source": source_name,
            "file": path.name,
            "relative_path": str(
                relative_path
            ),
            "output_path": str(
                out_path.relative_to(
                    GLOBAL_OUTPUT_DIR
                )
            ),
            "n_type5_nodes": np.nan,
            "n_type5_relinked": np.nan,
            "n_relinked_branches": np.nan,
            "n_nodes_before": np.nan,
            "n_nodes_after": np.nan,
            "node_change": np.nan,
            "fallback_copy": True,
            "node_records": [],
            "step_records": [],
        }


def run():
    output_dir = Path(
        CONFIG["output_dir"]
    )

    if CONFIG[
        "clear_output"
    ]:
        print(
            "\nClearing output directory..."
        )

        clear_output_dir(
            output_dir
        )

    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    tasks = []

    print(
        "\n===== Input ====="
    )

    for source in CONFIG[
        "input_dirs"
    ]:
        source_name = source[
            "name"
        ]

        input_root = Path(
            source["path"]
        )

        if not input_root.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{input_root}"
            )

        if not input_root.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: "
                f"{input_root}"
            )

        swc_files = sorted(
            input_root.rglob(
                "*.swc"
            )
        )

        print(
            f"\nSource: {source_name}"
        )

        print(
            "Input dir:",
            input_root,
        )

        print(
            "SWC files:",
            len(
                swc_files
            ),
        )

        tasks.extend(
            (
                str(path),
                str(input_root),
                source_name,
            )
            for path in swc_files
        )

    print(
        "\n===== Configuration ====="
    )

    print(
        "Total SWC files:",
        len(tasks),
    )

    print(
        "Output dir:",
        output_dir,
    )

    print(
        "Target node type:",
        CONFIG[
            "target_node_type"
        ],
    )

    print(
        "Anchor excluded types:",
        CONFIG[
            "anchor_excluded_types"
        ],
    )

    print(
        "Anchor max hops:",
        CONFIG[
            "anchor_max_hops"
        ],
    )

    print(
        "Anchor max distance:",
        CONFIG[
            "anchor_max_distance_um"
        ],
        "um",
    )

    if not tasks:
        print(
            "\nNo SWC files found."
        )

        return pd.DataFrame()

    with Pool(
        processes=CONFIG[
            "max_processes"
        ],
        initializer=init_worker,
        initargs=(
            str(
                output_dir
            ),
        ),
        maxtasksperchild=500,
    ) as pool:
        logs = list(
            tqdm(
                pool.imap_unordered(
                    process_one,
                    tasks,
                    chunksize=CONFIG[
                        "chunksize"
                    ],
                ),
                total=len(tasks),
                desc="Relinking SWC",
                mininterval=0.5,
            )
        )

    summary_rows = []
    node_rows = []
    step_rows = []

    for item in logs:
        node_records = (
            item.pop(
                "node_records",
                [],
            )
        )

        step_records = (
            item.pop(
                "step_records",
                [],
            )
        )

        summary_rows.append(
            item
        )

        for record in node_records:
            node_rows.append(
                {
                    "source": item[
                        "source"
                    ],
                    "file": item[
                        "file"
                    ],
                    "relative_path": item[
                        "relative_path"
                    ],
                    **record,
                }
            )

        for record in step_records:
            step_rows.append(
                {
                    "source": item[
                        "source"
                    ],
                    "file": item[
                        "file"
                    ],
                    "relative_path": item[
                        "relative_path"
                    ],
                    **record,
                }
            )

    df_log = pd.DataFrame(
        summary_rows
    )

    df_nodes = pd.DataFrame(
        node_rows
    )

    df_steps = pd.DataFrame(
        step_rows
    )

    df_log.to_csv(
        output_dir
        / "type5_relink_summary.csv",
        index=False,
    )

    df_nodes.to_csv(
        output_dir
        / "type5_relink_nodes.csv",
        index=False,
    )

    df_steps.to_csv(
        output_dir
        / "type5_relink_steps.csv",
        index=False,
    )

    return df_log


if __name__ == "__main__":
    df_log = run()

    if len(df_log) > 0:
        print(
            "\n===== Summary ====="
        )

        print(
            "Files:",
            len(df_log),
        )

        print(
            "Files with relinking:",
            (
                df_log[
                    "n_type5_relinked"
                ]
                .fillna(0)
                .gt(0)
                .sum()
            ),
        )

        print(
            "Total Type-5 nodes:",
            int(
                df_log[
                    "n_type5_nodes"
                ]
                .fillna(0)
                .sum()
            ),
        )

        print(
            "Total relinked Type-5 nodes:",
            int(
                df_log[
                    "n_type5_relinked"
                ]
                .fillna(0)
                .sum()
            ),
        )

        print(
            "Total relinked branches:",
            int(
                df_log[
                    "n_relinked_branches"
                ]
                .fillna(0)
                .sum()
            ),
        )

        print(
            "Files copied without processing:",
            int(
                df_log[
                    "fallback_copy"
                ]
                .fillna(False)
                .sum()
            ),
        )

        print(
            "\n===== Source Summary ====="
        )

        source_summary = (
            df_log
            .groupby(
                "source",
                dropna=False,
            )
            .agg(
                files=(
                    "file",
                    "size",
                ),
                files_with_relinking=(
                    "n_type5_relinked",
                    lambda x: (
                        x.fillna(0)
                        .gt(0)
                        .sum()
                    ),
                ),
                copied_without_processing=(
                    "fallback_copy",
                    lambda x: int(
                        x.fillna(False)
                        .sum()
                    ),
                ),
                type5_nodes=(
                    "n_type5_nodes",
                    lambda x: int(
                        x.fillna(0)
                        .sum()
                    ),
                ),
                relinked_type5_nodes=(
                    "n_type5_relinked",
                    lambda x: int(
                        x.fillna(0)
                        .sum()
                    ),
                ),
                relinked_branches=(
                    "n_relinked_branches",
                    lambda x: int(
                        x.fillna(0)
                        .sum()
                    ),
                ),
            )
        )

        print(
            source_summary
        )
