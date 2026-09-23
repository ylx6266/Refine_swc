import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

CONFIG = {
    "meta_path": "41586_2024_7686_MOESM5_ESM.tsv",
    "input_dir": "step0_out",
    "output_dir": "step1_out",
    "max_processes": 30,
    "chunksize": 300,
    "meta_scale": [250, 250, 25],
    "swc_scale": [1000, 1000, 1000],
    "trusted_dist": 2,
    "start_thr": 0.15,
    "min_thr": 0.01,
    "step": 0.01,
    "max_volume_loss": 0.10,
    "force_remove_thr": 0.05,
    "repair_trusted_soma": True,
    "repair_soma_need_check": False,
    "reclassify_untrusted_soma": True,
    "use_hardlink_for_unchanged": True,
    "save_history": False,
}

META_SCALE = np.array(CONFIG["meta_scale"], dtype=float)
SWC_SCALE = np.array(CONFIG["swc_scale"], dtype=float)

GLOBAL_META_SOMA = None
GLOBAL_DIRS = None
GLOBAL_LOG_DIR = None


def clear_output_dir(path):
    path = Path(path)

    if path.exists():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)

    path.mkdir(parents=True, exist_ok=True)


def init_worker(meta_soma_dict, dirs, log_dir):
    global GLOBAL_META_SOMA, GLOBAL_DIRS, GLOBAL_LOG_DIR

    GLOBAL_META_SOMA = meta_soma_dict
    GLOBAL_DIRS = {
        k: Path(v)
        for k, v in dirs.items()
    }
    GLOBAL_LOG_DIR = Path(log_dir)


def load_trusted_meta_soma(meta_path):
    df = pd.read_csv(meta_path, sep="\t")

    df = df[
        df["root_id"].notna()
        & df["soma_x"].notna()
        & df["soma_y"].notna()
        & df["soma_z"].notna()
    ].copy()

    df["Name"] = (
        df["root_id"]
        .astype("int64")
        .astype(str)
    )

    meta = {}

    for _, row in df.iterrows():
        meta[row["Name"]] = np.array(
            [
                row["soma_x"],
                row["soma_y"],
                row["soma_z"]
            ],
            dtype=float
        ) / META_SCALE

    print("\n===== Trusted meta soma =====")
    print("Trusted soma count:", len(meta))

    return meta


def check_meta_swc_match(meta_soma_dict, input_dir):
    swc_names = {
        path.stem
        for path in Path(input_dir).glob("*.swc")
    }

    print("\n===== Meta-SWC matching =====")
    print("SWC count:", len(swc_names))
    print("Trusted soma count:", len(meta_soma_dict))
    print(
        "SWC with trusted soma:",
        len(swc_names & set(meta_soma_dict.keys()))
    )


def fast_soma_scan(path):
    soma_xyz = []
    soma_ids = []
    parent_count = defaultdict(int)

    n_nodes = 0
    volume = 0.0

    with open(path, "r") as file:
        for line in file:
            if not line or line[0] == "#":
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            n_nodes += 1

            node_id = int(float(parts[0]))
            node_type = int(float(parts[1]))
            parent_id = int(float(parts[6]))
            radius = max(float(parts[5]), 0)

            parent_count[parent_id] += 1
            volume += radius ** 3

            if node_type == 1:
                soma_ids.append(node_id)
                soma_xyz.append(
                    [
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4])
                    ]
                )

    if len(soma_ids) == 0:
        return {
            "swc_soma_xyz": None,
            "n_soma_nodes": 0,
            "soma_id": None,
            "soma_children": None,
            "n_nodes": n_nodes,
            "volume": volume
        }

    soma_xyz = np.mean(
        np.asarray(soma_xyz, dtype=float),
        axis=0
    ) / SWC_SCALE

    soma_id = int(soma_ids[0])
    soma_children = int(
        parent_count.get(soma_id, 0)
    )

    return {
        "swc_soma_xyz": soma_xyz,
        "n_soma_nodes": len(soma_ids),
        "soma_id": soma_id,
        "soma_children": soma_children,
        "n_nodes": n_nodes,
        "volume": volume
    }


def classify_soma_status(name, scan):
    if scan["swc_soma_xyz"] is None:
        return "swc_no_soma", False, np.nan

    if name not in GLOBAL_META_SOMA:
        return "no_trusted_meta_soma", False, np.nan

    dist = float(
        np.linalg.norm(
            GLOBAL_META_SOMA[name]
            - scan["swc_soma_xyz"]
        )
    )

    if dist <= CONFIG["trusted_dist"]:
        return "trusted_soma", True, dist

    return "soma_need_check", True, dist


def link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        dst.unlink()

    if CONFIG["use_hardlink_for_unchanged"]:
        try:
            os.link(src, dst)
            return "hardlinked"
        except Exception:
            pass

    shutil.copy2(src, dst)

    return "copied"


def reclassify_untrusted_soma_text(src, dst):
    child_count = defaultdict(int)

    with open(src, "r") as fin:
        for line in fin:
            if not line or line[0] == "#":
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            parent_id = int(float(parts[6]))

            if parent_id != -1:
                child_count[parent_id] += 1

    type0_count = 0
    type5_count = 0

    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            if not line or line[0] == "#":
                fout.write(line)
                continue

            parts = line.split()

            if len(parts) < 7:
                fout.write(line)
                continue

            node_id = int(float(parts[0]))
            node_type = int(float(parts[1]))

            if node_type == 1:
                if child_count.get(node_id, 0) >= 2:
                    parts[1] = "5"
                    type5_count += 1
                else:
                    parts[1] = "0"
                    type0_count += 1

                fout.write(
                    " ".join(parts) + "\n"
                )
            else:
                fout.write(line)

    return type0_count, type5_count


def read_swc(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None
    )

    df = df.iloc[:, :7].copy()

    df.columns = [
        "ID",
        "Type",
        "X",
        "Y",
        "Z",
        "Radius",
        "Parent"
    ]

    df["ID"] = df["ID"].astype(int)
    df["Type"] = df["Type"].astype(int)
    df["Parent"] = df["Parent"].astype(int)

    return df


def write_swc(df, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        sep=" ",
        index=False,
        header=False,
        float_format="%.3f"
    )


def build_node_maps(df):
    child_map = defaultdict(list)
    parent_map = {}
    coord_map = {}

    for row in df.itertuples(index=False):
        node_id = int(row.ID)
        parent_id = int(row.Parent)

        parent_map[node_id] = parent_id

        coord_map[node_id] = np.array(
            [
                row.X,
                row.Y,
                row.Z
            ],
            dtype=float
        )

        if parent_id != -1:
            child_map[parent_id].append(node_id)

    return child_map, parent_map, coord_map


def get_subtree_nodes(root, child_map):
    stack = [int(root)]
    nodes = set()

    while stack:
        node = stack.pop()

        if node in nodes:
            continue

        nodes.add(node)
        stack.extend(
            child_map.get(node, [])
        )

    return nodes


def calculate_subtree_cable_length(
    branch_root,
    soma_id,
    child_map,
    parent_map,
    coord_map
):
    branch_root = int(branch_root)
    soma_id = int(soma_id)

    subtree_nodes = get_subtree_nodes(
        branch_root,
        child_map
    )

    total_length = 0.0

    for node_id in subtree_nodes:
        parent_id = parent_map.get(
            node_id,
            -1
        )

        if parent_id == -1:
            continue

        if (
            node_id not in coord_map
            or parent_id not in coord_map
        ):
            continue

        if (
            (
                node_id == branch_root
                and parent_id == soma_id
            )
            or parent_id in subtree_nodes
        ):
            total_length += float(
                np.linalg.norm(
                    coord_map[node_id]
                    - coord_map[parent_id]
                )
            )

    return total_length, subtree_nodes


def estimate_volume(df):
    return float(
        np.sum(
            np.maximum(
                df["Radius"].values.astype(float),
                0
            ) ** 3
        )
    )


def count_soma_children(df):
    soma = df[df["Type"] == 1]

    if soma.empty:
        return None, None

    soma_id = int(
        soma.iloc[0]["ID"]
    )

    n_child = int(
        (df["Parent"] == soma_id).sum()
    )

    return soma_id, n_child


def repair_soma_branches_adaptive(df):
    original_volume = estimate_volume(df)

    best_df = df.copy()
    best_thr = np.nan
    history = []

    thr = CONFIG["start_thr"]

    while thr >= CONFIG["min_thr"]:

        soma_id, n_child = count_soma_children(
            best_df
        )

        if soma_id is None or n_child <= 1:
            break

        (
            child_map,
            parent_map,
            coord_map
        ) = build_node_maps(best_df)

        branches = list(
            child_map.get(soma_id, [])
        )

        if len(branches) <= 1:
            break

        subtrees = {}
        branch_lengths = {}

        for branch_id in branches:
            (
                cable_length,
                subtree_nodes
            ) = calculate_subtree_cable_length(
                branch_root=branch_id,
                soma_id=soma_id,
                child_map=child_map,
                parent_map=parent_map,
                coord_map=coord_map
            )

            subtrees[branch_id] = subtree_nodes
            branch_lengths[branch_id] = cable_length

        if len(branch_lengths) == 0:
            break

        max_length = max(
            branch_lengths.values()
        )

        if max_length <= 0:
            break

        cutoff = max_length * thr

        force_cutoff = (
            max_length
            * CONFIG["force_remove_thr"]
        )

        remove_nodes = set()
        removed_branch_ids = []

        for (
            branch_id,
            cable_length
        ) in branch_lengths.items():

            if cable_length < cutoff:
                remove_nodes.update(
                    subtrees[branch_id]
                )

                removed_branch_ids.append(
                    branch_id
                )

        if len(remove_nodes) == 0:
            thr -= CONFIG["step"]
            continue

        all_removed_are_tiny = all(
            branch_lengths[branch_id]
            < force_cutoff
            for branch_id in removed_branch_ids
        )

        new_df = best_df[
            ~best_df["ID"].isin(
                remove_nodes
            )
        ].copy()

        new_volume = estimate_volume(
            new_df
        )

        volume_loss = (
            1 - new_volume / original_volume
            if original_volume > 0
            else np.nan
        )

        _, new_child = count_soma_children(
            new_df
        )

        removed_total_length = sum(
            branch_lengths[branch_id]
            for branch_id
            in removed_branch_ids
        )

        history.append({
            "threshold": thr,
            "force_remove_thr": CONFIG[
                "force_remove_thr"
            ],
            "before_children": n_child,
            "after_children": new_child,
            "max_branch_length": max_length,
            "length_cutoff": cutoff,
            "force_length_cutoff": force_cutoff,
            "removed_total_length": (
                removed_total_length
            ),
            "removed_nodes": len(
                remove_nodes
            ),
            "removed_branches": len(
                removed_branch_ids
            ),
            "all_removed_are_tiny": (
                all_removed_are_tiny
            ),
            "volume_loss": volume_loss
        })

        if (
            volume_loss
            > CONFIG["max_volume_loss"]
            and not all_removed_are_tiny
        ):
            break

        best_df = new_df
        best_thr = thr

        if (
            new_child is not None
            and new_child <= 1
        ):
            break

        thr -= CONFIG["step"]

    return (
        best_df,
        pd.DataFrame(history),
        best_thr
    )


def write_or_link_to_category(
    path,
    temp_path,
    final_dir
):
    final_path = (
        Path(final_dir)
        / Path(path).name
    )

    if temp_path is not None:
        if final_path.exists():
            final_path.unlink()

        shutil.move(
            str(temp_path),
            final_path
        )

        return "written"

    return link_or_copy(
        path,
        final_path
    )


def process_one(path):
    try:
        path = Path(path)
        name = path.stem

        scan = fast_soma_scan(path)

        (
            soma_status,
            has_trusted_meta_soma,
            soma_dist
        ) = classify_soma_status(
            name,
            scan
        )

        child_before = scan[
            "soma_children"
        ]
        child_after = child_before

        n_nodes_before = scan[
            "n_nodes"
        ]
        n_nodes_after = n_nodes_before

        vol_before = scan[
            "volume"
        ]
        vol_after = vol_before

        best_thr = np.nan
        changed = False
        temp_path = None
        repair_status = None
        reclassified_type0 = 0
        reclassified_type5 = 0

        if soma_status == "swc_no_soma":

            category = "no_soma"
            final_dir = GLOBAL_DIRS[
                "no_soma"
            ]
            repair_status = (
                "linked_swc_no_soma"
            )

        elif (
            soma_status
            == "no_trusted_meta_soma"
        ):

            category = "no_soma"
            final_dir = GLOBAL_DIRS[
                "no_soma"
            ]

            if CONFIG[
                "reclassify_untrusted_soma"
            ]:
                temp_path = (
                    Path("/tmp")
                    / f"{os.getpid()}_{path.name}"
                )

                (
                    reclassified_type0,
                    reclassified_type5
                ) = reclassify_untrusted_soma_text(
                    path,
                    temp_path
                )

                changed = (
                    reclassified_type0
                    + reclassified_type5
                    > 0
                )
                child_after = None
                repair_status = (
                    "reclassified_no_trusted_meta_soma"
                )
            else:
                repair_status = (
                    "linked_no_trusted_meta_soma"
                )

        elif (
            soma_status
            == "soma_need_check"
        ):

            category = "soma_need_check"
            final_dir = GLOBAL_DIRS[
                "soma_need_check"
            ]

            repair_status = (
                "linked_soma_need_check_no_repair"
            )

            changed = False
            temp_path = None
            child_after = child_before
            n_nodes_after = n_nodes_before
            vol_after = vol_before

        elif soma_status == "trusted_soma":

            if (
                CONFIG[
                    "repair_trusted_soma"
                ]
                and child_before
                is not None
                and child_before > 1
            ):
                df = read_swc(path)

                (
                    fixed_df,
                    history,
                    best_thr
                ) = repair_soma_branches_adaptive(
                    df
                )

                (
                    _,
                    child_after
                ) = count_soma_children(
                    fixed_df
                )

                n_nodes_after = len(
                    fixed_df
                )

                vol_after = estimate_volume(
                    fixed_df
                )

                temp_path = (
                    Path("/tmp")
                    / f"{os.getpid()}_{path.name}"
                )

                write_swc(
                    fixed_df,
                    temp_path
                )

                if (
                    CONFIG["save_history"]
                    and len(history) > 0
                ):
                    history.to_csv(
                        GLOBAL_LOG_DIR
                        / f"{name}_trusted_soma_history.csv",
                        index=False
                    )

                changed = (
                    len(fixed_df)
                    != len(df)
                )

                repair_status = (
                    "repaired_trusted_soma"
                )

            else:
                repair_status = (
                    "linked_trusted_soma"
                )

            if child_after == 1:
                category = "unipolar"
                final_dir = GLOBAL_DIRS[
                    "unipolar"
                ]
            else:
                category = "non_unipolar"
                final_dir = GLOBAL_DIRS[
                    "non_unipolar"
                ]

        else:

            category = "soma_need_check"
            final_dir = GLOBAL_DIRS[
                "soma_need_check"
            ]

            if (
                CONFIG[
                    "demote_untrusted_soma"
                ]
                and scan["n_soma_nodes"] > 0
            ):
                temp_path = (
                    Path("/tmp")
                    / f"{os.getpid()}_{path.name}"
                )

                (
                    reclassified_type0,
                    reclassified_type5
                ) = reclassify_untrusted_soma_text(
                    path,
                    temp_path
                )

                changed = (
                    reclassified_type0
                    + reclassified_type5
                    > 0
                )
                child_after = None
                repair_status = (
                    "reclassified_unknown_status"
                )

            else:
                repair_status = (
                    "linked_unknown_status"
                )

        output_method = (
            write_or_link_to_category(
                path,
                temp_path,
                final_dir
            )
        )

        node_loss = (
            1
            - n_nodes_after
            / n_nodes_before
            if n_nodes_before > 0
            else np.nan
        )

        volume_loss = (
            1
            - vol_after
            / vol_before
            if vol_before > 0
            else np.nan
        )

        return {
            "file": path.name,
            "name": name,
            "category": category,
            "repair_status": (
                repair_status
            ),
            "soma_status": soma_status,
            "has_trusted_meta_soma": (
                has_trusted_meta_soma
            ),
            "n_soma_nodes_before": scan[
                "n_soma_nodes"
            ],
            "original_soma_id": scan[
                "soma_id"
            ],
            "soma_dist_before": (
                soma_dist
            ),
            "soma_children_before": (
                child_before
            ),
            "soma_children_after": (
                child_after
            ),
            "n_nodes_before": (
                n_nodes_before
            ),
            "n_nodes_after": (
                n_nodes_after
            ),
            "node_loss": node_loss,
            "volume_before": vol_before,
            "volume_after": vol_after,
            "volume_loss": volume_loss,
            "best_threshold": best_thr,
            "reclassified_type0": (
                reclassified_type0
            ),
            "reclassified_type5": (
                reclassified_type5
            ),
            "changed": changed,
            "output_method": (
                output_method
            ),
            "error": None
        }

    except Exception as error:
        return {
            "file": Path(path).name,
            "category": "error",
            "repair_status": "error",
            "error": str(error)
        }


def run_folder():
    input_dir = Path(
        CONFIG["input_dir"]
    )

    output_dir = Path(
        CONFIG["output_dir"]
    )

    print(
        "\nClearing output directory..."
    )

    clear_output_dir(
        output_dir
    )

    dirs = {
        "unipolar": (
            output_dir / "unipolar"
        ),
        "non_unipolar": (
            output_dir / "non_unipolar"
        ),
        "no_soma": (
            output_dir / "no_soma"
        ),
        "soma_need_check": (
            output_dir
            / "soma_need_check"
        ),
    }

    for directory in dirs.values():
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    log_dir = (
        output_dir / "repair_history"
    )

    if CONFIG["save_history"]:
        log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    meta_soma_dict = (
        load_trusted_meta_soma(
            CONFIG["meta_path"]
        )
    )

    check_meta_swc_match(
        meta_soma_dict,
        input_dir
    )

    files = [
        str(path)
        for path
        in input_dir.glob("*.swc")
    ]

    with Pool(
        processes=CONFIG[
            "max_processes"
        ],
        initializer=init_worker,
        initargs=(
            meta_soma_dict,
            {
                key: str(value)
                for key, value
                in dirs.items()
            },
            str(log_dir)
        )
    ) as pool:

        logs = list(
            tqdm(
                pool.imap_unordered(
                    process_one,
                    files,
                    chunksize=CONFIG[
                        "chunksize"
                    ]
                ),
                total=len(files)
            )
        )

    df_log = pd.DataFrame(logs)

    df_log.to_csv(
        output_dir
        / "repair_log.csv",
        index=False
    )

    return df_log


if __name__ == "__main__":

    df_log = run_folder()

    print("\n===== Category =====")
    print(
        df_log["category"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n===== Repair status ====="
    )
    print(
        df_log["repair_status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n===== Reclassified untrusted soma nodes ====="
    )
    print(
        "Type 1 to Type 0:",
        pd.to_numeric(
            df_log.get(
                "reclassified_type0",
                pd.Series(dtype=float)
            ),
            errors="coerce"
        ).fillna(0).sum()
    )
    print(
        "Type 1 to Type 5:",
        pd.to_numeric(
            df_log.get(
                "reclassified_type5",
                pd.Series(dtype=float)
            ),
            errors="coerce"
        ).fillna(0).sum()
    )

    print(
        "\n===== Soma status ====="
    )
    print(
        df_log["soma_status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n===== Soma children before ====="
    )
    print(
        df_log[
            "soma_children_before"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .head(30)
    )

    print(
        "\n===== Soma children after ====="
    )
    print(
        df_log[
            "soma_children_after"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .head(30)
    )

    print(
        "\n===== Trusted soma fixed to 1 ====="
    )

    trusted = df_log[
        df_log["soma_status"]
        == "trusted_soma"
    ]

    print(
        (
            trusted[
                "soma_children_after"
            ]
            == 1
        ).mean()
        if len(trusted)
        else np.nan
    )

    print(
        "\n===== Soma_need_check kept without repair ====="
    )

    soma_need_check = df_log[
        df_log["soma_status"]
        == "soma_need_check"
    ]

    print(
        soma_need_check["category"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n===== Volume loss > max ====="
    )
    print(
        (
            df_log["volume_loss"]
            > CONFIG[
                "max_volume_loss"
            ]
        ).sum()
    )

    print(
        "\n===== Output SWC count ====="
    )

    output_dir = Path(
        CONFIG["output_dir"]
    )

    total = sum(
        len(
            list(
                (
                    output_dir
                    / category
                ).glob("*.swc")
            )
        )
        for category in [
            "unipolar",
            "non_unipolar",
            "no_soma",
            "soma_need_check",
        ]
    )

    print(total)
