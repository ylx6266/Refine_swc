from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import os
import shutil
import traceback

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D


CFG = {
    "input_root": r"mannual_check/merge/unipolar",
    "output_root": r"soma_pruned",

    "coord_scale": 1000.0,
    "radius_scale": 1000.0,

    "neurite_ratio_threshold": 1.5,
    "soma_ratio_threshold": 2.0,
    "max_soma_distance_um": 20.0,

    "confirmation_window": 5,
    "confirmation_min_valid": 4,
    "confirmation_low_fraction": 0.80,

    "lookahead_nodes": 10,
    "lookahead_min_valid": 6,
    "lookahead_low_fraction": 0.75,

    "candidate_cluster_min_samples": 3,
    "candidate_cluster_nn_quantile": 0.30,
    "candidate_cluster_eps_factor": 2.0,
    "candidate_cluster_core_distance_factor": 8.0,

    "cluster_path_max_neurite_fraction": 0.40,
    "cluster_path_min_neurite_allowance": 2,

    "small_soma_fallback": True,
    "small_soma_max_cable_um": 20,
    "small_soma_max_euclidean_um": 5,
    "small_soma_min_radius_ratio": 1.8,

    "skip_large_soma_span": True,
    "max_soma_to_neuron_span_ratio": 0.10,

    "max_absorb_iterations": 20,
    "max_search_nodes_per_iteration": 10000,

    "retype_0_5": True,
    "remove_stale_output": True,

    "protect_longest_primary_neurite": True,
    "longest_path_tolerance_um": 1e-6,

    "keep_all_major_exits": True,
    "major_exit_ratio_threshold": 0.30,

    "skip_long_submajor_exit": True,
    "submajor_exit_soma_span_factor": 5.0,

    "workers": min(8, os.cpu_count() or 1),
    "progress_every": 100,

    "save_soma_core_csv": True,
    "soma_core_csv_root": r"soma_core_batch_debug",

    "save_plots": True,
    "plot_root": r"soma_pruned_plots",
    "plot_mode": "modified_only",
    "plot_scope": "zoom",
    "zoom_radius_um": 25.0,
    "plot_view": "default",
    "plot_dpi": 300,
}

VIEWS = {
    "default": (22, -58),
    "front": (0, -90),
    "side": (0, 0),
    "top": (90, -90),
    "back": (0, 90),
}


LOG_FIELDS = [
    "relative_path",
    "output_path",
    "nodes_before",
    "nodes_after",
    "deleted_nodes",
    "soma_core_nodes",
    "small_soma_fallback_used",
    "small_soma_fallback_added_nodes",
    "soma_core_max_distance_um",
    "primary_axis_soma_span_um",
    "primary_axis_neuron_span_um",
    "soma_to_neuron_span_ratio",
    "primary_axis_terminal_id",
    "primary_axis_anchor_id",
    "primary_axis_path_length_um",
    "boundary_exit_n",
    "primary_exit_id",
    "primary_exit_parent_id",
    "primary_downstream_cable_um",
    "primary_cable_fraction",
    "retained_major_exit_n",
    "retained_major_exit_ids",
    "second_to_first_exit_ratio",
    "retyped_nodes",
    "search_truncated",
    "growth_limited",
]


def load_swc(path):
    comments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                comments.append(line.rstrip("\n"))

    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[1] < 7:
        raise ValueError("SWC must contain at least 7 columns")

    ids = data[:, 0].astype(np.int64)
    types = data[:, 1].astype(np.int16)
    xyz = data[:, 2:5].astype(float) / CFG["coord_scale"]
    radius = data[:, 5].astype(float) / CFG["radius_scale"]
    parents = data[:, 6].astype(np.int64)

    return data, comments, ids, types, xyz, radius, parents


def build_rooted_tree(ids, xyz, parents, soma):
    n = len(ids)

    id_to_idx = {
        int(v): i
        for i, v in enumerate(ids)
    }

    adj = [[] for _ in range(n)]

    for child, pid in enumerate(parents):
        parent = id_to_idx.get(int(pid))

        if parent is None:
            continue

        adj[parent].append(child)
        adj[child].append(parent)

    tree_parent = np.full(
        n,
        -2,
        dtype=np.int64
    )

    tree_children = [
        []
        for _ in range(n)
    ]

    depth = np.full(
        n,
        -1,
        dtype=np.int32
    )

    edge_length = np.zeros(
        n,
        dtype=float
    )

    order = []

    tree_parent[soma] = -1
    depth[soma] = 0

    q = deque([soma])

    while q:
        node = q.popleft()
        order.append(node)

        for child in adj[node]:
            if tree_parent[child] != -2:
                continue

            tree_parent[child] = node
            tree_children[node].append(child)

            depth[child] = (
                depth[node] + 1
            )

            edge_length[child] = np.linalg.norm(
                xyz[child] - xyz[node]
            )

            q.append(child)

    return (
        tree_parent,
        tree_children,
        depth,
        order,
        edge_length
    )


def calculate_subtree_metrics(
    xyz,
    soma,
    tree_children,
    order,
    edge_length
):
    n = len(xyz)

    subtree_nodes = np.zeros(
        n,
        dtype=np.int64
    )

    subtree_cable = np.zeros(
        n,
        dtype=float
    )

    soma_distance = np.linalg.norm(
        xyz - xyz[soma],
        axis=1
    )

    subtree_max_distance = np.full(
        n,
        np.nan,
        dtype=float
    )

    for node in order:
        subtree_nodes[node] = 1

        subtree_max_distance[node] = (
            soma_distance[node]
        )

    for node in reversed(order):
        for child in tree_children[node]:
            subtree_nodes[node] += (
                subtree_nodes[child]
            )

            subtree_cable[node] += (
                edge_length[child]
                + subtree_cable[child]
            )

            subtree_max_distance[node] = max(
                subtree_max_distance[node],
                subtree_max_distance[child]
            )

    return (
        subtree_nodes,
        subtree_cable,
        subtree_max_distance,
        soma_distance
    )


def classify_radius_state(ratio):
    if not np.isfinite(ratio):
        return "uncertain"

    if ratio <= CFG["neurite_ratio_threshold"]:
        return "neurite_like"

    if ratio >= CFG["soma_ratio_threshold"]:
        return "soma_like"

    return "uncertain"


def get_dominant_path(
    start,
    tree_children,
    subtree_cable,
    edge_length,
    max_nodes
):
    path = []
    node = start

    while len(path) < max_nodes:
        path.append(node)

        children = tree_children[node]

        if not children:
            break

        node = max(
            children,
            key=lambda child:
            edge_length[child]
            + subtree_cable[child]
        )

    return path


def get_exit_evidence(
    start,
    tree_children,
    subtree_cable,
    edge_length,
    radius_ratio,
    radius_state
):
    path = get_dominant_path(
        start,
        tree_children,
        subtree_cable,
        edge_length,
        CFG["lookahead_nodes"]
    )

    values = np.asarray(
        [
            radius_ratio[node]
            for node in path
        ],
        dtype=float
    )

    short_values = (
        values[
            :CFG["confirmation_window"]
        ]
    )

    short_values = (
        short_values[
            np.isfinite(short_values)
        ]
    )

    full_values = values[
        np.isfinite(values)
    ]

    current_state = (
        radius_state[start]
    )

    if (
        len(short_values)
        < CFG["confirmation_min_valid"]
        or
        len(full_values)
        < CFG["lookahead_min_valid"]
    ):
        return {
            "candidate": False,
            "current_state":
                current_state,
            "confirmation_n":
                len(short_values),
            "confirmation_median":
                np.nan,
            "confirmation_low_fraction":
                np.nan,
            "lookahead_n":
                len(full_values),
            "lookahead_median":
                np.nan,
            "lookahead_low_fraction":
                np.nan
        }

    short_median = float(
        np.median(short_values)
    )

    short_low_fraction = float(
        np.mean(
            short_values
            <= CFG["neurite_ratio_threshold"]
        )
    )

    full_median = float(
        np.median(full_values)
    )

    full_low_fraction = float(
        np.mean(
            full_values
            <= CFG["neurite_ratio_threshold"]
        )
    )

    candidate = (
        current_state != "soma_like"
        and
        short_median
        <= CFG["neurite_ratio_threshold"]
        and
        short_low_fraction
        >= CFG["confirmation_low_fraction"]
        and
        full_median
        <= CFG["neurite_ratio_threshold"]
        and
        full_low_fraction
        >= CFG["lookahead_low_fraction"]
    )

    return {
        "candidate":
            candidate,

        "current_state":
            current_state,

        "confirmation_n":
            len(short_values),

        "confirmation_median":
            short_median,

        "confirmation_low_fraction":
            short_low_fraction,

        "lookahead_n":
            len(full_values),

        "lookahead_median":
            full_median,

        "lookahead_low_fraction":
            full_low_fraction
    }


def get_core_frontier(
    soma_core,
    tree_children
):
    frontier = set()

    for node in soma_core:
        for child in tree_children[node]:
            if child not in soma_core:
                frontier.add(child)

    return sorted(frontier)


def scan_from_soma_core(
    soma_core,
    tree_children,
    subtree_cable,
    edge_length,
    radius_ratio,
    radius_state,
    soma_distance
):
    frontier = get_core_frontier(
        soma_core,
        tree_children
    )

    candidate_exits = []
    candidate_stats = {}

    searched_nodes = set(
        soma_core
    )

    stack = list(
        reversed(frontier)
    )

    searched_count = 0
    truncated = False

    while stack:
        node = stack.pop()

        if node in soma_core:
            continue

        if (
            soma_distance[node]
            > CFG["max_soma_distance_um"]
        ):
            continue

        searched_count += 1

        if (
            searched_count
            > CFG[
                "max_search_nodes_per_iteration"
            ]
        ):
            truncated = True
            break

        stats = get_exit_evidence(
            node,
            tree_children,
            subtree_cable,
            edge_length,
            radius_ratio,
            radius_state
        )

        candidate_stats[node] = (
            stats
        )

        if stats["candidate"]:
            candidate_exits.append(
                node
            )
            continue

        searched_nodes.add(
            node
        )

        for child in reversed(
            tree_children[node]
        ):
            if child in soma_core:
                continue

            if (
                soma_distance[child]
                <= CFG["max_soma_distance_um"]
            ):
                stack.append(child)

    return (
        searched_nodes,
        candidate_exits,
        candidate_stats,
        truncated
    )


def estimate_cluster_eps(
    candidate_nodes,
    xyz
):
    if len(candidate_nodes) < 2:
        return np.nan

    points = xyz[
        candidate_nodes
    ]

    tree = cKDTree(
        points
    )

    dists, _ = tree.query(
        points,
        k=2
    )

    nearest = dists[:, 1]

    nearest = nearest[
        np.isfinite(nearest)
        & (nearest > 0)
    ]

    if len(nearest) == 0:
        return np.nan

    base = float(
        np.quantile(
            nearest,
            CFG[
                "candidate_cluster_nn_quantile"
            ]
        )
    )

    return (
        base
        * CFG[
            "candidate_cluster_eps_factor"
        ]
    )


def path_segment_to_core(
    node,
    soma_core,
    tree_parent
):
    segment = []
    current = node

    while current >= 0:
        segment.append(
            current
        )

        parent = tree_parent[
            current
        ]

        if parent in soma_core:
            return (
                segment,
                parent
            )

        current = parent

    return (
        [],
        None
    )


def path_is_soma_compatible(
    segment,
    radius_state,
    soma_distance
):
    if not segment:
        return False

    if any(
        soma_distance[node]
        > CFG["max_soma_distance_um"]
        for node in segment
    ):
        return False

    neurite_like_n = sum(
        radius_state[node]
        == "neurite_like"
        for node in segment
    )

    allowed = max(
        CFG[
            "cluster_path_min_neurite_allowance"
        ],
        int(
            np.ceil(
                len(segment)
                * CFG[
                    "cluster_path_max_neurite_fraction"
                ]
            )
        )
    )

    return (
        neurite_like_n
        <= allowed
    )


def find_absorbable_cluster(
    soma_core,
    candidate_exits,
    xyz,
    tree_parent,
    radius_state,
    soma_distance
):
    label_map = {
        node: -1
        for node
        in candidate_exits
    }

    if (
        len(candidate_exits)
        < CFG[
            "candidate_cluster_min_samples"
        ]
    ):
        return (
            set(),
            None,
            np.nan,
            label_map
        )

    candidate_nodes = np.asarray(
        candidate_exits,
        dtype=int
    )

    candidate_nodes = candidate_nodes[
        soma_distance[candidate_nodes]
        <= CFG["max_soma_distance_um"]
    ]

    if (
        len(candidate_nodes)
        < CFG[
            "candidate_cluster_min_samples"
        ]
    ):
        return (
            set(),
            None,
            np.nan,
            label_map
        )

    eps = estimate_cluster_eps(
        candidate_nodes,
        xyz
    )

    if (
        not np.isfinite(eps)
        or eps <= 0
    ):
        return (
            set(),
            None,
            eps,
            label_map
        )

    labels = DBSCAN(
        eps=eps,
        min_samples=CFG[
            "candidate_cluster_min_samples"
        ]
    ).fit_predict(
        xyz[
            candidate_nodes
        ]
    )

    for node, label in zip(
        candidate_nodes,
        labels
    ):
        label_map[
            int(node)
        ] = int(label)

    valid_labels = [
        int(label)
        for label
        in np.unique(labels)
        if label >= 0
    ]

    if not valid_labels:
        return (
            set(),
            None,
            eps,
            label_map
        )

    core_nodes = np.asarray(
        sorted(
            soma_core
        ),
        dtype=int
    )

    core_tree = cKDTree(
        xyz[
            core_nodes
        ]
    )

    clusters = []

    for label in valid_labels:
        nodes = candidate_nodes[
            labels == label
        ]

        points = xyz[
            nodes
        ]

        core_distance, _ = (
            core_tree.query(
                points,
                k=1
            )
        )

        centroid = np.mean(
            points,
            axis=0
        )

        spread = np.linalg.norm(
            points
            - centroid,
            axis=1
        )

        accepted_nodes = []
        rejected_nodes = []

        for node in nodes:
            segment, core_anchor = (
                path_segment_to_core(
                    int(node),
                    soma_core,
                    tree_parent
                )
            )

            if (
                core_anchor is None
                or
                not path_is_soma_compatible(
                    segment,
                    radius_state,
                    soma_distance
                )
            ):
                rejected_nodes.append(
                    int(node)
                )

            else:
                accepted_nodes.append(
                    int(node)
                )

        min_core_distance = float(
            np.min(
                core_distance
            )
        )

        median_core_distance = float(
            np.median(
                core_distance
            )
        )

        max_allowed_core_distance = (
            CFG[
                "candidate_cluster_core_distance_factor"
            ]
            * eps
        )

        eligible = (
            len(accepted_nodes)
            >= CFG[
                "candidate_cluster_min_samples"
            ]
            and
            min_core_distance
            <= max_allowed_core_distance
        )

        clusters.append({
            "label":
                int(label),

            "nodes":
                set(
                    int(x)
                    for x
                    in nodes.tolist()
                ),

            "accepted_nodes":
                set(
                    accepted_nodes
                ),

            "rejected_nodes":
                set(
                    rejected_nodes
                ),

            "n":
                int(
                    len(nodes)
                ),

            "accepted_n":
                int(
                    len(
                        accepted_nodes
                    )
                ),

            "min_core_distance_um":
                min_core_distance,

            "median_core_distance_um":
                median_core_distance,

            "median_spread_um":
                float(
                    np.median(
                        spread
                    )
                ),

            "max_spread_um":
                float(
                    np.max(
                        spread
                    )
                ),

            "max_allowed_core_distance_um":
                float(
                    max_allowed_core_distance
                ),

            "eligible":
                bool(
                    eligible
                )
        })

    eligible_clusters = [
        cluster
        for cluster
        in clusters
        if cluster["eligible"]
    ]

    if not eligible_clusters:
        return (
            set(),
            None,
            eps,
            label_map
        )

    eligible_clusters.sort(
        key=lambda cluster: (
            cluster[
                "median_core_distance_um"
            ],
            -cluster[
                "accepted_n"
            ],
            cluster[
                "median_spread_um"
            ]
        )
    )

    best = (
        eligible_clusters[0]
    )

    return (
        set(
            best[
                "accepted_nodes"
            ]
        ),
        best,
        eps,
        label_map
    )


def absorb_cluster_into_core(
    soma_core,
    cluster_nodes,
    tree_parent,
    soma_distance
):
    new_core = set(
        soma_core
    )

    added_nodes = set()

    for node in cluster_nodes:
        if (
            soma_distance[node]
            > CFG["max_soma_distance_um"]
        ):
            continue

        segment, core_anchor = (
            path_segment_to_core(
                node,
                new_core,
                tree_parent
            )
        )

        if core_anchor is None:
            continue

        if any(
            soma_distance[current]
            > CFG["max_soma_distance_um"]
            for current in segment
        ):
            continue

        for current in segment:
            if current not in new_core:
                new_core.add(
                    current
                )

                added_nodes.add(
                    current
                )

    return (
        new_core,
        added_nodes
    )


def iterative_soma_growth(
    soma,
    tree_parent,
    tree_children,
    subtree_cable,
    edge_length,
    radius_ratio,
    radius_state,
    xyz,
    soma_distance
):
    soma_core = {
        soma
    }

    absorbed_cluster_nodes = set()
    absorbed_path_nodes = set()

    history = []

    truncated_any = False

    for iteration in range(
        1,
        CFG[
            "max_absorb_iterations"
        ] + 1
    ):
        (
            searched_nodes,
            candidate_exits,
            candidate_stats,
            truncated
        ) = scan_from_soma_core(
            soma_core,
            tree_children,
            subtree_cable,
            edge_length,
            radius_ratio,
            radius_state,
            soma_distance
        )

        truncated_any = (
            truncated_any
            or truncated
        )

        (
            cluster_nodes,
            cluster_info,
            cluster_eps,
            cluster_labels
        ) = find_absorbable_cluster(
            soma_core,
            candidate_exits,
            xyz,
            tree_parent,
            radius_state,
            soma_distance
        )

        history.append({
            "iteration":
                iteration,

            "soma_core_n_before":
                len(
                    soma_core
                ),

            "soma_core_max_distance_um_before":
                float(
                    np.max(
                        soma_distance[
                            list(soma_core)
                        ]
                    )
                ),

            "searched_n":
                len(
                    searched_nodes
                ),

            "candidate_n":
                len(
                    candidate_exits
                ),

            "cluster_eps_um":
                cluster_eps,

            "absorb_cluster_found":
                bool(
                    cluster_nodes
                ),

            "absorb_cluster_n":
                len(
                    cluster_nodes
                ),

            "cluster_label":
                (
                    cluster_info[
                        "label"
                    ]
                    if cluster_info
                    is not None
                    else np.nan
                ),

            "cluster_total_n":
                (
                    cluster_info[
                        "n"
                    ]
                    if cluster_info
                    is not None
                    else 0
                ),

            "cluster_accepted_n":
                (
                    cluster_info[
                        "accepted_n"
                    ]
                    if cluster_info
                    is not None
                    else 0
                ),

            "cluster_min_core_distance_um":
                (
                    cluster_info[
                        "min_core_distance_um"
                    ]
                    if cluster_info
                    is not None
                    else np.nan
                ),

            "cluster_median_core_distance_um":
                (
                    cluster_info[
                        "median_core_distance_um"
                    ]
                    if cluster_info
                    is not None
                    else np.nan
                ),

            "max_soma_distance_um":
                CFG["max_soma_distance_um"],

            "search_truncated":
                truncated
        })

        if not cluster_nodes:
            break

        (
            new_core,
            newly_added
        ) = absorb_cluster_into_core(
            soma_core,
            cluster_nodes,
            tree_parent,
            soma_distance
        )

        if not newly_added:
            break

        absorbed_cluster_nodes.update(
            cluster_nodes
        )

        absorbed_path_nodes.update(
            newly_added
        )

        soma_core = (
            new_core
        )

    (
        searched_nodes,
        candidate_exits,
        candidate_stats,
        final_truncated
    ) = scan_from_soma_core(
        soma_core,
        tree_children,
        subtree_cable,
        edge_length,
        radius_ratio,
        radius_state,
        soma_distance
    )

    truncated_any = (
        truncated_any
        or final_truncated
    )

    (
        final_absorbable,
        _,
        final_cluster_eps,
        cluster_labels
    ) = find_absorbable_cluster(
        soma_core,
        candidate_exits,
        xyz,
        tree_parent,
        radius_state,
        soma_distance
    )

    growth_limit_reached = (
        bool(
            final_absorbable
        )
        and
        len(history)
        >= CFG[
            "max_absorb_iterations"
        ]
    )

    if growth_limit_reached:
        truncated_any = True

    too_far = {
        node
        for node in soma_core
        if (
            soma_distance[node]
            > CFG["max_soma_distance_um"]
        )
    }

    if too_far:
        raise RuntimeError(
            f"{len(too_far)} soma-core nodes exceed "
            f"max_soma_distance_um="
            f"{CFG['max_soma_distance_um']}"
        )

    return (
        soma_core,
        absorbed_cluster_nodes,
        absorbed_path_nodes,
        searched_nodes,
        candidate_exits,
        candidate_stats,
        cluster_labels,
        final_cluster_eps,
        history,
        truncated_any
    )




def expand_small_soma_fallback(
    soma,
    soma_core,
    tree_children,
    edge_length,
    radius_ratio,
    soma_distance,
):
    if not CFG["small_soma_fallback"]:
        return set(soma_core), set()

    if len(soma_core) != 1 or soma not in soma_core:
        return set(soma_core), set()

    max_cable = float(
        CFG["small_soma_max_cable_um"]
    )
    max_euclidean = float(
        CFG["small_soma_max_euclidean_um"]
    )
    min_ratio = float(
        CFG["small_soma_min_radius_ratio"]
    )

    new_core = set(soma_core)
    added = set()
    queue = []

    for child in tree_children[soma]:
        child = int(child)
        step = float(edge_length[child])

        if (
            not np.isfinite(step)
            or step < 0
        ):
            continue

        queue.append((child, step))

    while queue:
        node, cable_distance = queue.pop(0)

        if node in new_core:
            continue

        if (
            not np.isfinite(cable_distance)
            or cable_distance > max_cable
        ):
            continue

        if (
            not np.isfinite(soma_distance[node])
            or soma_distance[node] > max_euclidean
        ):
            continue

        ratio = radius_ratio[node]

        if (
            not np.isfinite(ratio)
            or ratio < min_ratio
        ):
            continue

        new_core.add(node)
        added.add(node)

        for child in tree_children[node]:
            child = int(child)
            step = float(edge_length[child])

            if (
                not np.isfinite(step)
                or step < 0
            ):
                continue

            next_cable = cable_distance + step

            if next_cable <= max_cable:
                queue.append(
                    (
                        child,
                        next_cable,
                    )
                )

    return new_core, added

def estimate_axis_span(
    points,
    origin,
    axis,
    radii=None,
):
    points = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)

    if len(points) == 0:
        return 0.0

    finite_points = np.all(np.isfinite(points), axis=1)

    if radii is not None:
        radii = np.asarray(radii, dtype=float)

        if len(radii) != len(points):
            raise ValueError(
                "radii length must match points length"
            )

        finite = finite_points
    else:
        finite = finite_points

    points = points[finite]

    if len(points) == 0:
        return 0.0

    norm = float(np.linalg.norm(axis))

    if not np.isfinite(norm) or norm <= 0:
        return np.nan

    axis = axis / norm

    projections = (
        points - origin
    ) @ axis

    if radii is None:
        return float(
            np.max(projections)
            - np.min(projections)
        )

    radii = radii[finite]

    safe_radii = np.where(
        np.isfinite(radii) & (radii > 0),
        radii,
        0.0,
    )

    lower = projections - safe_radii
    upper = projections + safe_radii

    return float(
        np.max(upper)
        - np.min(lower)
    )

def estimate_radius_aware_3d_span(
    points,
    radii,
):
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if len(points) == 0:
        return 0.0

    if len(radii) != len(points):
        raise ValueError(
            "radii length must match points length"
        )

    finite_points = np.all(
        np.isfinite(points),
        axis=1,
    )

    points = points[finite_points]
    radii = radii[finite_points]

    if len(points) == 0:
        return 0.0

    safe_radii = np.where(
        np.isfinite(radii) & (radii > 0),
        radii,
        0.0,
    )

    if len(points) == 1:
        return float(
            2.0 * safe_radii[0]
        )

    best = 0.0

    for i in range(len(points) - 1):
        distances = np.linalg.norm(
            points[i + 1:] - points[i],
            axis=1,
        )

        candidate = (
            distances
            + safe_radii[i]
            + safe_radii[i + 1:]
        )

        if len(candidate):
            best = max(
                best,
                float(np.max(candidate)),
            )

    best = max(
        best,
        float(
            2.0 * np.max(safe_radii)
        ),
    )

    return float(best)


def get_primary_neurite_axis(
    soma,
    primary_exit,
    xyz,
    tree_children,
    subtree_cable,
    edge_length,
):
    start = int(primary_exit["exit_idx"])
    path = [start]
    node = start
    path_length_um = float(edge_length[start])

    while tree_children[node]:
        child = max(
            tree_children[node],
            key=lambda x: edge_length[x] + subtree_cable[x],
        )
        child = int(child)
        path.append(child)
        path_length_um += float(edge_length[child])
        node = child

    path_idx = np.asarray(path, dtype=int)
    soma_xyz = xyz[soma]

    displacement = xyz[path_idx] - soma_xyz
    radial_distance = np.linalg.norm(
        displacement,
        axis=1,
    )

    if not len(radial_distance):
        return {
            "axis": None,
            "terminal_idx": None,
            "anchor_idx": None,
            "path_length_um": np.nan,
        }

    anchor_local = int(np.argmax(radial_distance))
    anchor = int(path_idx[anchor_local])
    anchor_distance = float(radial_distance[anchor_local])

    if not np.isfinite(anchor_distance) or anchor_distance <= 0:
        return {
            "axis": None,
            "terminal_idx": int(path_idx[-1]),
            "anchor_idx": anchor,
            "path_length_um": float(path_length_um),
        }

    axis = xyz[anchor] - soma_xyz
    axis_norm = float(np.linalg.norm(axis))

    if not np.isfinite(axis_norm) or axis_norm <= 0:
        return {
            "axis": None,
            "terminal_idx": int(path_idx[-1]),
            "anchor_idx": anchor,
            "path_length_um": float(path_length_um),
        }

    axis = axis / axis_norm

    return {
        "axis": axis,
        "terminal_idx": int(path_idx[-1]),
        "anchor_idx": anchor,
        "path_length_um": float(path_length_um),
    }
def calculate_root_distance(
    soma,
    tree_children,
    edge_length,
):
    root_distance = np.full(len(edge_length), np.nan, dtype=float)
    root_distance[soma] = 0.0
    stack = [soma]

    while stack:
        node = stack.pop()
        for child in tree_children[node]:
            root_distance[child] = root_distance[node] + edge_length[child]
            stack.append(child)

    return root_distance


def get_longest_primary_paths(
    soma,
    tree_parent,
    tree_children,
    edge_length,
):
    root_distance = calculate_root_distance(
        soma,
        tree_children,
        edge_length,
    )

    terminals = [
        node
        for node, children in enumerate(tree_children)
        if not children and np.isfinite(root_distance[node])
    ]

    if not terminals:
        return [], np.nan

    max_length = max(root_distance[node] for node in terminals)
    tol = CFG["longest_path_tolerance_um"]

    longest_terminals = [
        node
        for node in terminals
        if abs(root_distance[node] - max_length) <= tol
    ]

    paths = []
    for terminal in longest_terminals:
        path = set(get_path_to_soma(terminal, soma, tree_parent))
        paths.append(
            {
                "terminal_idx": terminal,
                "path_nodes": path,
                "path_length_um": float(root_distance[terminal]),
            }
        )

    return paths, float(max_length)


def longest_primary_neurite_would_be_pruned(
    soma,
    tree_parent,
    tree_children,
    edge_length,
    keep_nodes,
):
    if not CFG["protect_longest_primary_neurite"]:
        return False, [], np.nan

    longest_paths, max_length = get_longest_primary_paths(
        soma,
        tree_parent,
        tree_children,
        edge_length,
    )

    threatened = []

    for info in longest_paths:
        deleted_on_path = info["path_nodes"] - keep_nodes
        if deleted_on_path:
            threatened.append(
                {
                    "terminal_idx": info["terminal_idx"],
                    "path_length_um": info["path_length_um"],
                    "deleted_path_nodes": deleted_on_path,
                }
            )

    return bool(threatened), threatened, max_length


def find_boundary_exits(
    soma_core,
    tree_children,
    subtree_nodes,
    subtree_cable,
    edge_length,
):
    exits = []

    for parent in soma_core:
        for child in tree_children[parent]:
            if child in soma_core:
                continue

            cable = edge_length[child] + subtree_cable[child]
            exits.append(
                {
                    "parent_idx": parent,
                    "exit_idx": child,
                    "downstream_nodes": int(subtree_nodes[child]),
                    "downstream_cable_um": float(cable),
                }
            )

    exits.sort(
        key=lambda x: (
            x["downstream_cable_um"],
            x["downstream_nodes"],
        ),
        reverse=True,
    )

    return exits


def select_major_exits(exits):
    if not exits:
        return [], np.nan

    top_cable = float(exits[0]["downstream_cable_um"])

    if len(exits) >= 2 and top_cable > 0:
        second_to_first = float(
            exits[1]["downstream_cable_um"] / top_cable
        )
    else:
        second_to_first = np.nan

    if not CFG["keep_all_major_exits"] or top_cable <= 0:
        return [exits[0]], second_to_first

    major_exits = []

    for i, item in enumerate(exits):
        if i == 0:
            ratio_to_top = 1.0
        else:
            ratio_to_top = float(
                item["downstream_cable_um"] / top_cable
            )

        item["ratio_to_top"] = ratio_to_top

        if ratio_to_top >= CFG["major_exit_ratio_threshold"]:
            major_exits.append(item)

    if not major_exits:
        major_exits = [exits[0]]

    return major_exits, second_to_first


def find_long_submajor_exits(
    exits,
    soma_span_um,
    soma,
    tree_children,
    edge_length,
):
    if (
        not CFG["skip_long_submajor_exit"]
        or len(exits) < 2
        or not np.isfinite(soma_span_um)
        or soma_span_um <= 0
    ):
        return []

    top_cable = float(exits[0]["downstream_cable_um"])

    if not np.isfinite(top_cable) or top_cable <= 0:
        return []

    required_path_distance_um = (
        float(CFG["submajor_exit_soma_span_factor"])
        * float(soma_span_um)
    )

    flagged = []

    for item in exits[1:]:
        cable_um = float(item["downstream_cable_um"])

        ratio_to_top = (
            cable_um / top_cable
            if top_cable > 0
            else np.nan
        )

        if (
            not np.isfinite(ratio_to_top)
            or ratio_to_top
            >= CFG["major_exit_ratio_threshold"]
        ):
            continue

        exit_idx = int(item["exit_idx"])

        stack = [
            (
                exit_idx,
                float(edge_length[exit_idx]),
            )
        ]

        max_path_distance_um = 0.0
        farthest_terminal_idx = exit_idx
        terminal_n = 0

        while stack:
            node, path_distance = stack.pop()
            children = tree_children[node]

            if not children:
                terminal_n += 1

                if path_distance > max_path_distance_um:
                    max_path_distance_um = float(path_distance)
                    farthest_terminal_idx = int(node)

                continue

            for child in children:
                child = int(child)
                step = float(edge_length[child])

                if not np.isfinite(step) or step < 0:
                    continue

                stack.append(
                    (
                        child,
                        path_distance + step,
                    )
                )

        if (
            max_path_distance_um
            >= required_path_distance_um
        ):
            flagged.append(
                {
                    **item,
                    "ratio_to_top": float(ratio_to_top),
                    "soma_span_um": float(soma_span_um),
                    "required_path_distance_um": float(
                        required_path_distance_um
                    ),
                    "max_path_distance_um": float(
                        max_path_distance_um
                    ),
                    "farthest_terminal_idx": int(
                        farthest_terminal_idx
                    ),
                    "terminal_n": int(terminal_n),
                }
            )

    return flagged

def get_path_to_soma(node, soma, tree_parent):
    path = []
    current = node

    while True:
        path.append(current)

        if current == soma:
            break

        current = int(tree_parent[current])
        if current < 0:
            raise ValueError("Cannot trace primary path back to Type-1 soma")

    path.reverse()
    return path


def get_descendants(root, tree_children):
    result = set()
    stack = [root]

    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(tree_children[node])

    return result


def build_keep_delete_sets(
    soma,
    retained_exits,
    soma_core,
    tree_parent,
    tree_children,
    n_nodes,
):
    backbone = set()
    retained_subtrees = set()

    for exit_info in retained_exits:
        exit_idx = exit_info["exit_idx"]
        exit_parent = exit_info["parent_idx"]

        exit_backbone = set(
            get_path_to_soma(
                exit_parent,
                soma,
                tree_parent,
            )
        )

        if not exit_backbone.issubset(soma_core):
            raise ValueError(
                "Retained soma backbone is not fully contained in soma_core"
            )

        backbone.update(exit_backbone)
        retained_subtrees.update(
            get_descendants(exit_idx, tree_children)
        )

    keep_nodes = backbone | retained_subtrees
    all_nodes = set(range(n_nodes))
    delete_nodes = all_nodes - keep_nodes

    return backbone, retained_subtrees, keep_nodes, delete_nodes


def validate_pruned_tree(soma, keep_nodes, tree_parent):
    for node in keep_nodes:
        if node == soma:
            continue
        parent = int(tree_parent[node])
        if parent not in keep_nodes:
            raise ValueError(
                f"Pruning would leave orphan node index {node}"
            )


def retype_topology(data, keep_mask):
    output = data[keep_mask].copy()
    ids = output[:, 0].astype(np.int64)
    types_before = output[:, 1].astype(np.int16).copy()
    parents = output[:, 6].astype(np.int64)

    child_count = {int(node): 0 for node in ids}
    kept_ids = set(int(x) for x in ids)

    for parent in parents:
        parent = int(parent)
        if parent in kept_ids:
            child_count[parent] += 1

    if CFG["retype_0_5"]:
        for i, node in enumerate(ids):
            if int(output[i, 1]) not in (0, 5):
                continue
            output[i, 1] = 5 if child_count[int(node)] >= 2 else 0

    types_after = output[:, 1].astype(np.int16)
    retyped_nodes = int(np.sum(types_before != types_after))

    return output, retyped_nodes


def save_swc(path, data, comments):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for line in comments:
            f.write(line + "\n")

        for row in data:
            values = [
                str(int(row[0])),
                str(int(row[1])),
                f"{row[2]:.6f}",
                f"{row[3]:.6f}",
                f"{row[4]:.6f}",
                f"{row[5]:.6f}",
                str(int(row[6])),
            ]

            if len(row) > 7:
                values.extend(f"{x:.6f}" for x in row[7:])

            f.write(" ".join(values) + "\n")


def build_segments(ids, xyz, parents, node_set):
    id_to_idx = {int(node): i for i, node in enumerate(ids)}
    segments = []

    for child_id, parent_id in zip(ids, parents):
        child_id = int(child_id)
        parent_id = int(parent_id)

        if parent_id == -1:
            continue

        if child_id not in node_set or parent_id not in node_set:
            continue

        child = id_to_idx[child_id]
        parent = id_to_idx[parent_id]
        segments.append([xyz[parent], xyz[child]])

    return segments


def save_pruning_plot(
    relative,
    ids,
    xyz,
    parents,
    soma,
    soma_core,
    backbone,
    retained_exits,
    keep_nodes,
    delete_nodes,
):
    if not CFG["save_plots"]:
        return

    if CFG["plot_mode"] != "modified_only":
        return

    plot_root = Path(CFG["plot_root"]).resolve()
    plot_path = plot_root / relative
    plot_path = plot_path.with_suffix('.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    id_to_idx = {int(node): i for i, node in enumerate(ids)}
    all_nodes = set(int(i) for i in ids)
    kept_id_nodes = set(int(ids[i]) for i in keep_nodes)
    deleted_id_nodes = all_nodes - kept_id_nodes
    backbone_id_nodes = set(int(ids[i]) for i in backbone)

    original_segments = build_segments(ids, xyz, parents, all_nodes)
    kept_segments = build_segments(ids, xyz, parents, kept_id_nodes)
    deleted_segments = []

    for child_id, parent_id in zip(ids, parents):
        child_id = int(child_id)
        parent_id = int(parent_id)
        if parent_id == -1 or child_id not in deleted_id_nodes:
            continue
        if parent_id not in id_to_idx:
            continue
        deleted_segments.append([xyz[id_to_idx[parent_id]], xyz[id_to_idx[child_id]]])

    backbone_segments = build_segments(ids, xyz, parents, backbone_id_nodes)

    fig = plt.figure(figsize=(6.8, 5.9), dpi=150)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if original_segments:
        ax.add_collection3d(Line3DCollection(
            original_segments, colors='#D0D0D0', linewidths=0.4, alpha=0.28, zorder=1
        ))

    if kept_segments:
        ax.add_collection3d(Line3DCollection(
            kept_segments, colors='#808080', linewidths=0.6, alpha=0.55, zorder=3
        ))

    if deleted_segments:
        ax.add_collection3d(Line3DCollection(
            deleted_segments, colors='#CC79A7', linewidths=1.2, alpha=0.85, zorder=5
        ))

    if backbone_segments:
        ax.add_collection3d(Line3DCollection(
            backbone_segments, colors='#E69F00', linewidths=2.4, alpha=1.0, zorder=10
        ))

    core_idx = np.asarray(sorted(soma_core), dtype=int)
    if len(core_idx):
        ax.scatter(
            xyz[core_idx, 0], xyz[core_idx, 1], xyz[core_idx, 2],
            s=13, color='#E69F00', alpha=0.75, depthshade=False, zorder=9
        )

    ax.scatter(
        [xyz[soma, 0]], [xyz[soma, 1]], [xyz[soma, 2]],
        s=58, color='#D55E00', edgecolor='white', linewidth=0.7, depthshade=False, zorder=15
    )

    exit_indices = [item["exit_idx"] for item in retained_exits]
    if exit_indices:
        ax.scatter(
            xyz[exit_indices, 0], xyz[exit_indices, 1], xyz[exit_indices, 2],
            s=48, color='#009E73', edgecolor='white', linewidth=0.7,
            depthshade=False, zorder=16
        )

    if CFG['plot_scope'] == 'zoom':
        center = xyz[soma]
        radius_plot = CFG['zoom_radius_um']
        ax.set_xlim(center[0]-radius_plot, center[0]+radius_plot)
        ax.set_ylim(center[1]-radius_plot, center[1]+radius_plot)
        ax.set_zlim(center[2]-radius_plot, center[2]+radius_plot)
    else:
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        radius_plot = max(np.ptp(xyz, axis=0).max() / 2, 1e-6)
        ax.set_xlim(center[0]-radius_plot, center[0]+radius_plot)
        ax.set_ylim(center[1]-radius_plot, center[1]+radius_plot)
        ax.set_zlim(center[2]-radius_plot, center[2]+radius_plot)

    ax.set_box_aspect((1, 1, 1))
    elev, azim = VIEWS[CFG['plot_view']]
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.text2D(0.02, 0.98, str(relative), transform=ax.transAxes, ha='left', va='top', fontsize=8)

    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#D55E00', markeredgecolor='white', markersize=7, label='Type-1 soma'),
        Line2D([0], [0], color='#E69F00', lw=2.5, label='Retained soma backbone'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#009E73', markeredgecolor='white', markersize=7, label='Retained major exit(s)'),
        Line2D([0], [0], color='#CC79A7', lw=2, label='Pruned branches'),
        Line2D([0], [0], color='#808080', lw=1, label='Retained neurite'),
    ]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=CFG['plot_dpi'], bbox_inches='tight')
    plt.close(fig)


def remove_stale_output(path):
    if CFG["remove_stale_output"] and path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def copy_original(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def save_soma_core_csv(relative, ids, xyz, radius, radius_ratio, radius_state, soma_core):
    if not CFG["save_soma_core_csv"]:
        return

    root = Path(CFG["soma_core_csv_root"])
    out_path = (root / relative).with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "x_um",
                "y_um",
                "z_um",
                "radius_um",
                "radius_ratio",
                "radius_state",
            ]
        )

        for node in sorted(soma_core):
            writer.writerow(
                [
                    int(ids[node]),
                    float(xyz[node, 0]),
                    float(xyz[node, 1]),
                    float(xyz[node, 2]),
                    float(radius[node]),
                    (
                        float(radius_ratio[node])
                        if np.isfinite(radius_ratio[node])
                        else ""
                    ),
                    str(radius_state[node]),
                ]
            )


def process_one_file(path_str):
    input_root = Path(CFG["input_root"]).resolve()
    output_root = Path(CFG["output_root"]).resolve()
    path = Path(path_str).resolve()
    relative = path.relative_to(input_root)
    output_path = output_root / relative

    try:
        (
            data,
            comments,
            ids,
            types,
            xyz,
            radius,
            parents,
        ) = load_swc(path)

        soma_indices = np.flatnonzero(types == 1)
        if len(soma_indices) != 1:
            copy_original(path, output_path)
            return {
                "status": "skipped",
                "path": str(relative),
                "reason": f"Type-1 count = {len(soma_indices)}",
            }

        soma = int(soma_indices[0])

        if int(parents[soma]) != -1:
            copy_original(path, output_path)
            return {
                "status": "skipped",
                "path": str(relative),
                "reason": "Type-1 soma is not root",
            }

        (
            tree_parent,
            tree_children,
            depth,
            order,
            edge_length,
        ) = build_rooted_tree(
            ids,
            xyz,
            parents,
            soma,
        )

        connected_mask = np.zeros(
            len(ids),
            dtype=bool,
        )
        connected_mask[order] = True

        (
            subtree_nodes,
            subtree_cable,
            subtree_max_distance,
            soma_distance,
        ) = calculate_subtree_metrics(
            xyz,
            soma,
            tree_children,
            order,
            edge_length,
        )

        valid_radius = (
            connected_mask
            & np.isfinite(radius)
            & (radius > 0)
            & (types != 1)
        )

        if not np.any(valid_radius):
            copy_original(path, output_path)
            return {
                "status": "skipped",
                "path": str(relative),
                "reason": "No valid non-Type-1 radius values",
            }

        neurite_baseline = float(np.median(radius[valid_radius]))

        radius_ratio = np.full(len(ids), np.nan, dtype=float)
        radius_ratio[valid_radius] = radius[valid_radius] / neurite_baseline

        radius_state = np.asarray(
            [
                "soma_seed"
                if i == soma
                else classify_radius_state(radius_ratio[i])
                for i in range(len(ids))
            ],
            dtype=object,
        )

        (
            soma_core,
            absorbed_cluster_nodes,
            absorbed_path_nodes,
            searched_nodes,
            candidate_exits,
            candidate_stats,
            candidate_cluster_labels,
            final_cluster_eps,
            growth_history,
            search_truncated,
        ) = iterative_soma_growth(
            soma,
            tree_parent,
            tree_children,
            subtree_cable,
            edge_length,
            radius_ratio,
            radius_state,
            xyz,
            soma_distance,
        )

        small_soma_fallback_used = False
        small_soma_fallback_added = set()

        if CFG["small_soma_fallback"] and len(soma_core) == 1:
            soma_core, small_soma_fallback_added = (
                expand_small_soma_fallback(
                    soma,
                    soma_core,
                    tree_children,
                    edge_length,
                    radius_ratio,
                    soma_distance,
                )
            )
            small_soma_fallback_used = bool(
                small_soma_fallback_added
            )

        growth_limited = bool(
            growth_history
            and len(growth_history) >= CFG["max_absorb_iterations"]
            and any(
                bool(item.get("absorb_cluster_found"))
                for item in growth_history[-1:]
            )
        )

        soma_core_idx = np.asarray(
            sorted(soma_core),
            dtype=int,
        )

        save_soma_core_csv(
            relative,
            ids,
            xyz,
            radius,
            radius_ratio,
            radius_state,
            soma_core,
        )

        exits = find_boundary_exits(
            soma_core,
            tree_children,
            subtree_nodes,
            subtree_cable,
            edge_length,
        )

        if not exits:
            copy_original(path, output_path)
            return {
                "status": "skipped",
                "path": str(relative),
                "reason": "No soma-core boundary exit",
            }

        primary_exit = exits[0]

        primary_axis_info = get_primary_neurite_axis(
            soma,
            primary_exit,
            xyz,
            tree_children,
            subtree_cable,
            edge_length,
        )

        primary_axis = primary_axis_info["axis"]

        soma_span_um = estimate_radius_aware_3d_span(
            xyz[soma_core_idx],
            radius[soma_core_idx],
        )

        primary_path_length_um = (
            float(primary_axis_info["path_length_um"])
            if np.isfinite(primary_axis_info["path_length_um"])
            else np.nan
        )

        soma_to_neuron_span_ratio = (
            soma_span_um / primary_path_length_um
            if (
                np.isfinite(soma_span_um)
                and np.isfinite(primary_path_length_um)
                and primary_path_length_um > 0
            )
            else np.nan
        )

        if (
            CFG["skip_large_soma_span"]
            and np.isfinite(soma_to_neuron_span_ratio)
            and soma_to_neuron_span_ratio
            > CFG["max_soma_to_neuron_span_ratio"]
        ):
            copy_original(path, output_path)

            return {
                "status": "skipped",
                "path": str(relative),
                "reason": (
                    "Radius-aware 3D soma span / primary-path length ratio exceeds threshold: "
                    f"{soma_to_neuron_span_ratio:.6f} > "
                    f"{CFG['max_soma_to_neuron_span_ratio']:.6f}"
                ),
                "skip_category": "large_soma_primary_neurite_axis_span",
                "soma_span_um": float(soma_span_um),
                "primary_path_length_um": float(
                    primary_path_length_um
                ),
                "soma_to_neuron_span_ratio": float(
                    soma_to_neuron_span_ratio
                ),
                "primary_axis_terminal_id": (
                    int(ids[primary_axis_info["terminal_idx"]])
                    if primary_axis_info["terminal_idx"] is not None
                    else ""
                ),
                "primary_axis_anchor_id": (
                    int(ids[primary_axis_info["anchor_idx"]])
                    if primary_axis_info["anchor_idx"] is not None
                    else ""
                ),
                "primary_path_length_um": (
                    float(primary_axis_info["path_length_um"])
                    if np.isfinite(primary_axis_info["path_length_um"])
                    else np.nan
                ),
                "soma_core_nodes": int(len(soma_core)),
            }

        search_truncated = bool(search_truncated)

        long_submajor_exits = (
            find_long_submajor_exits(
                exits,
                soma_span_um,
                soma,
                tree_children,
                edge_length,
            )
        )

        if long_submajor_exits:
            copy_original(
                path,
                output_path,
            )

            flagged_ids = [
                int(ids[item["exit_idx"]])
                for item in long_submajor_exits
            ]

            flagged_ratios = [
                float(item["ratio_to_top"])
                for item in long_submajor_exits
            ]

            flagged_path_distances = [
                float(
                    item[
                        "max_path_distance_um"
                    ]
                )
                for item in long_submajor_exits
            ]

            flagged_terminal_ids = [
                int(
                    ids[
                        item[
                            "farthest_terminal_idx"
                        ]
                    ]
                )
                for item in long_submajor_exits
            ]

            flagged_terminal_counts = [
                int(
                    item[
                        "terminal_n"
                    ]
                )
                for item in long_submajor_exits
            ]

            required_path_distance_um = float(
                CFG[
                    "submajor_exit_soma_span_factor"
                ]
                * soma_span_um
            )

            return {
                "status": "skipped",
                "path": str(relative),
                "reason": (
                    "Submajor boundary exit is below "
                    f"major-exit ratio threshold "
                    f"({CFG['major_exit_ratio_threshold']:.6f}) "
                    "but its farthest terminal-to-soma Euclidean "
                    "distance is >= "
                    f"{CFG['submajor_exit_soma_span_factor']:.6f} "
                    f"x soma span "
                    f"({required_path_distance_um:.6f} um)"
                ),
                "skip_category":
                    "long_submajor_exit",
                "soma_span_um":
                    float(soma_span_um),
                "major_exit_ratio_threshold":
                    float(
                        CFG[
                            "major_exit_ratio_threshold"
                        ]
                    ),
                "soma_span_factor":
                    float(
                        CFG[
                            "submajor_exit_soma_span_factor"
                        ]
                    ),
                "required_path_distance_um":
                    required_path_distance_um,
                "flagged_exit_n":
                    int(
                        len(
                            long_submajor_exits
                        )
                    ),
                "flagged_exit_ids":
                    "|".join(
                        str(x)
                        for x in flagged_ids
                    ),
                "flagged_exit_ratios":
                    "|".join(
                        f"{x:.6f}"
                        for x in flagged_ratios
                    ),
                "flagged_terminal_ids":
                    "|".join(
                        str(x)
                        for x in flagged_terminal_ids
                    ),
                "flagged_terminal_counts":
                    "|".join(
                        str(x)
                        for x in flagged_terminal_counts
                    ),
                "flagged_max_path_distance_um":
                    "|".join(
                        f"{x:.6f}"
                        for x in flagged_path_distances
                    ),
            }

        total_exit_cable = sum(x["downstream_cable_um"] for x in exits)
        primary_fraction = (
            primary_exit["downstream_cable_um"] / total_exit_cable
            if total_exit_cable > 0
            else np.nan
        )

        retained_exits, second_to_first_exit_ratio = select_major_exits(exits)

        backbone, retained_subtrees, keep_nodes, delete_nodes = build_keep_delete_sets(
            soma,
            retained_exits,
            soma_core,
            tree_parent,
            tree_children,
            len(ids),
        )

        if not delete_nodes:
            copy_original(path, output_path)
            return {
                "status": "unchanged",
                "path": str(relative),
                "reason": "No nodes require pruning",
            }

        primary_guard_triggered, threatened_paths, longest_path_um = (
            longest_primary_neurite_would_be_pruned(
                soma,
                tree_parent,
                tree_children,
                edge_length,
                keep_nodes,
            )
        )

        if primary_guard_triggered:
            copy_original(path, output_path)
            threatened_terminal_ids = [
                int(ids[item["terminal_idx"]])
                for item in threatened_paths
            ]
            return {
                "status": "unchanged",
                "path": str(relative),
                "reason": (
                    "Primary-neurite guard: pruning would remove "
                    f"a longest soma-to-terminal path ({longest_path_um:.6f} um); "
                    f"terminal IDs={threatened_terminal_ids}"
                ),
            }

        validate_pruned_tree(
            soma,
            keep_nodes,
            tree_parent,
        )

        keep_mask = np.asarray(
            [i in keep_nodes for i in range(len(ids))],
            dtype=bool,
        )

        output_data, retyped_nodes = retype_topology(
            data,
            keep_mask,
        )

        output_ids = set(output_data[:, 0].astype(np.int64))
        output_parents = output_data[:, 6].astype(np.int64)
        output_node_ids = output_data[:, 0].astype(np.int64)

        for node_id, parent_id in zip(output_node_ids, output_parents):
            if int(node_id) == int(ids[soma]):
                continue
            if int(parent_id) not in output_ids:
                raise RuntimeError(
                    f"Orphan node after pruning: {int(node_id)}"
                )

        save_swc(
            output_path,
            output_data,
            comments,
        )

        save_pruning_plot(
            relative,
            ids,
            xyz,
            parents,
            soma,
            soma_core,
            backbone,
            retained_exits,
            keep_nodes,
            delete_nodes,
        )

        core_idx = np.asarray(sorted(soma_core), dtype=int)
        core_max_distance = float(np.max(soma_distance[core_idx]))

        return {
            "status": "modified",
            "relative_path": str(relative),
            "output_path": str(Path(CFG["output_root"]) / relative),
            "nodes_before": int(len(ids)),
            "nodes_after": int(len(output_data)),
            "deleted_nodes": int(len(delete_nodes)),
            "soma_core_nodes": int(len(soma_core)),
            "small_soma_fallback_used": bool(
                small_soma_fallback_used
            ),
            "small_soma_fallback_added_nodes": int(
                len(small_soma_fallback_added)
            ),
            "soma_core_max_distance_um": core_max_distance,
            "primary_axis_soma_span_um": (
                float(soma_span_um)
                if np.isfinite(soma_span_um)
                else np.nan
            ),
            "primary_axis_neuron_span_um": (
                float(primary_path_length_um)
                if np.isfinite(primary_path_length_um)
                else np.nan
            ),
            "soma_to_neuron_span_ratio": (
                float(soma_to_neuron_span_ratio)
                if np.isfinite(soma_to_neuron_span_ratio)
                else np.nan
            ),
            "primary_axis_terminal_id": (
                int(ids[primary_axis_info["terminal_idx"]])
                if primary_axis_info["terminal_idx"] is not None
                else ""
            ),
            "primary_axis_anchor_id": (
                int(ids[primary_axis_info["anchor_idx"]])
                if primary_axis_info["anchor_idx"] is not None
                else ""
            ),
            "primary_axis_path_length_um": (
                float(primary_axis_info["path_length_um"])
                if np.isfinite(primary_axis_info["path_length_um"])
                else np.nan
            ),
            "boundary_exit_n": int(len(exits)),
            "primary_exit_id": int(ids[primary_exit["exit_idx"]]),
            "primary_exit_parent_id": int(ids[primary_exit["parent_idx"]]),
            "primary_downstream_cable_um": float(
                primary_exit["downstream_cable_um"]
            ),
            "primary_cable_fraction": float(primary_fraction),
            "retained_major_exit_n": int(len(retained_exits)),
            "retained_major_exit_ids": ";".join(
                str(int(ids[item["exit_idx"]]))
                for item in retained_exits
            ),
            "second_to_first_exit_ratio": (
                float(second_to_first_exit_ratio)
                if np.isfinite(second_to_first_exit_ratio)
                else np.nan
            ),
            "retyped_nodes": int(retyped_nodes),
            "search_truncated": bool(search_truncated),
            "growth_limited": bool(growth_limited),
        }

    except Exception as e:
        remove_stale_output(output_path)
        return {
            "status": "error",
            "path": str(relative),
            "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def is_inside(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_swc_files():
    input_root = Path(CFG["input_root"]).resolve()
    output_root = Path(CFG["output_root"]).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    files = []

    for path in input_root.rglob("*.swc"):
        if is_inside(path, output_root):
            continue
        files.append(path.resolve())

    files.sort()
    return files


def write_log(rows):
    output_root = Path(CFG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "soma_pruning_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LOG_FIELDS})

    return log_path


def write_large_soma_skip_log(rows):
    output_root = Path(CFG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "primary_neurite_axis_span_skip_log.csv"

    fields = [
        "path",
        "soma_core_nodes",
        "soma_span_um",
        "neuron_span_um",
        "soma_to_neuron_span_ratio",
        "primary_axis_terminal_id",
        "primary_axis_anchor_id",
        "primary_path_length_um",
        "reason",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in fields
                }
            )

    return log_path


def write_long_submajor_exit_skip_log(rows):
    output_root = Path(
        CFG["output_root"]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        output_root
        / "long_submajor_exit_skip_log.csv"
    )

    fields = [
        "path",
        "soma_span_um",
        "major_exit_ratio_threshold",
        "soma_span_factor",
        "required_path_distance_um",
        "flagged_exit_n",
        "flagged_exit_ids",
        "flagged_exit_ratios",
        "flagged_terminal_ids",
        "flagged_terminal_counts",
        "flagged_max_path_distance_um",
        "reason",
    ]

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in fields
                }
            )

    return log_path


def main():
    files = discover_swc_files()
    total = len(files)

    print(f"Found {total} SWC files")
    print(f"Input root:  {Path(CFG['input_root']).resolve()}")
    print(f"Output root: {Path(CFG['output_root']).resolve()}")
    print(f"Workers:     {CFG['workers']}")
    print(
        f"Fixed soma search distance: "
        f"{CFG['max_soma_distance_um']} μm"
    )
    print(
        "Small-soma fallback: "
        f"{CFG['small_soma_fallback']} "
        f"(cable <= {CFG['small_soma_max_cable_um']} μm, "
        f"Euclidean <= {CFG['small_soma_max_euclidean_um']} μm, "
        f"radius ratio >= {CFG['small_soma_min_radius_ratio']})"
    )
    print("Soma detector: exact detect(1).py iterative logic")
    print(f"Primary-neurite guard: {CFG['protect_longest_primary_neurite']}")
    print(
        f"Skip large soma span: "
        f"{CFG['skip_large_soma_span']} "
        f"(ratio > "
        f"{CFG['max_soma_to_neuron_span_ratio']})"
    )
    print(f"Keep all major exits: {CFG['keep_all_major_exits']}")
    print(f"Major-exit ratio threshold: {CFG['major_exit_ratio_threshold']}")
    print(
        "Skip long submajor exits: "
        f"{CFG['skip_long_submajor_exit']} "
        f"(longest single topological path distance >= "
        f"{CFG['submajor_exit_soma_span_factor']} "
        "x soma span)"
    )
    if CFG["save_plots"]:
        print(f"Plot root:   {Path(CFG['plot_root']).resolve()}")

    modified_rows = []
    large_soma_skip_rows = []
    long_submajor_exit_skip_rows = []
    skipped = 0
    unchanged = 0
    errors = 0
    error_examples = []

    def update_result(result, pbar):
        nonlocal skipped, unchanged, errors

        status = result["status"]

        if status == "modified":
            modified_rows.append(result)
        elif status == "unchanged":
            unchanged += 1
        elif status == "skipped":
            skipped += 1

            if (
                result.get("skip_category")
                == "large_soma_primary_neurite_axis_span"
            ):
                large_soma_skip_rows.append(
                    result
                )
            elif (
                result.get("skip_category")
                == "long_submajor_exit"
            ):
                long_submajor_exit_skip_rows.append(
                    result
                )
        else:
            errors += 1
            if len(error_examples) < 20:
                error_examples.append(result)

        pbar.update(1)
        pbar.set_postfix(
            modified=len(modified_rows),
            unchanged=unchanged,
            skipped=skipped,
            errors=errors,
            refresh=False,
        )

    if total == 0:
        log_path = write_log([])
        skip_log_path = write_large_soma_skip_log([])
        submajor_skip_log_path = (
            write_long_submajor_exit_skip_log([])
        )
        print("\n===== Finished =====")
        print("Total:     0")
        print("Modified:  0")
        print("Unchanged: 0")
        print("Skipped:   0")
        print("Errors:    0")
        print(f"Log:       {log_path.resolve()}")
        print(f"Skip log:  {skip_log_path.resolve()}")
        print(
            "Submajor skip log: "
            f"{submajor_skip_log_path.resolve()}"
        )
        return

    with tqdm(
        total=total,
        desc="Processing SWC",
        unit="file",
        dynamic_ncols=True,
        mininterval=0.2,
    ) as pbar:

        if CFG["workers"] <= 1:
            for path in files:
                result = process_one_file(str(path))
                update_result(result, pbar)

        else:
            max_pending = max(CFG["workers"] * 4, 1)

            with ProcessPoolExecutor(max_workers=CFG["workers"]) as executor:
                file_iter = iter(files)
                pending = set()

                for _ in range(min(max_pending, total)):
                    try:
                        path = next(file_iter)
                    except StopIteration:
                        break

                    pending.add(
                        executor.submit(
                            process_one_file,
                            str(path)
                        )
                    )

                while pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED
                    )

                    for future in done:
                        try:
                            result = future.result()
                        except Exception as e:
                            result = {
                                "status": "error",
                                "path": "<worker future>",
                                "reason": f"{type(e).__name__}: {e}",
                            }

                        update_result(result, pbar)

                        try:
                            path = next(file_iter)
                        except StopIteration:
                            path = None

                        if path is not None:
                            pending.add(
                                executor.submit(
                                    process_one_file,
                                    str(path)
                                )
                            )

    modified_rows.sort(key=lambda x: x["relative_path"])
    large_soma_skip_rows.sort(
        key=lambda x: x["path"]
    )
    long_submajor_exit_skip_rows.sort(
        key=lambda x: x["path"]
    )

    log_path = write_log(modified_rows)
    skip_log_path = write_large_soma_skip_log(
        large_soma_skip_rows
    )
    submajor_skip_log_path = (
        write_long_submajor_exit_skip_log(
            long_submajor_exit_skip_rows
        )
    )

    print("\n===== Finished =====")
    print(f"Total:     {total}")
    print(f"Modified:  {len(modified_rows)}")
    print(f"Unchanged: {unchanged}")
    print(f"Skipped:   {skipped}")
    print(
        f"Primary-axis span skips: "
        f"{len(large_soma_skip_rows)}"
    )
    print(
        f"Long-submajor-exit skips: "
        f"{len(long_submajor_exit_skip_rows)}"
    )
    print(f"Errors:    {errors}")
    print(f"Log:       {log_path.resolve()}")
    print(f"Skip log:  {skip_log_path.resolve()}")
    print(
        "Submajor skip log: "
        f"{submajor_skip_log_path.resolve()}"
    )

    if error_examples:
        print("\nFirst error examples:")
        for item in error_examples:
            print(f"- {item.get('path', '<unknown>')}: {item.get('reason', '')}")


if __name__ == "__main__":
    main()