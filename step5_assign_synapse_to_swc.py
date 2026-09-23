import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


CONFIG = {
    "swc_dir": "step4_out",
    "pre_dir": "pre_neurites",
    "post_dir": "post_neurites",
    "output_dir": "step5_out",

    "max_processes": 30,
    "chunksize": 200,

    "coord_scale": [1000, 1000, 1000],

    "clear_output": True,
    "preserve_subdirs": True,

    "detail_dirname": "synapse_assignment_details",
    "plot_bins": 200,
    "plot_dpi": 300,
}


COORD_SCALE = np.array(CONFIG["coord_scale"], dtype=float)

GLOBAL_SWC_DIR = None
GLOBAL_OUT_DIR = None
GLOBAL_PRE_MAP = None
GLOBAL_POST_MAP = None


def clear_output_dir(path):
    path = Path(path)
    if path.exists():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)
    path.mkdir(parents=True, exist_ok=True)


def init_worker(swc_dir, out_dir, pre_map, post_map):
    global GLOBAL_SWC_DIR, GLOBAL_OUT_DIR, GLOBAL_PRE_MAP, GLOBAL_POST_MAP
    GLOBAL_SWC_DIR = Path(swc_dir)
    GLOBAL_OUT_DIR = Path(out_dir)
    GLOBAL_PRE_MAP = pre_map
    GLOBAL_POST_MAP = post_map


def build_file_map(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}

    return {
        f.stem: f
        for f in folder.rglob("*")
        if f.is_file()
    }


def read_swc(path):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)

    if df.shape[1] < 7:
        raise ValueError(f"Invalid SWC columns: {path}")

    base = df.iloc[:, :7].copy()
    base.columns = ["ID", "Type", "X", "Y", "Z", "Radius", "Parent"]

    base["ID"] = base["ID"].astype(int)
    base["Type"] = base["Type"].astype(int)
    base["Parent"] = base["Parent"].astype(int)

    return base


def read_syn_xyz(path):
    if path is None or not Path(path).exists():
        return np.empty((0, 3), dtype=float)

    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    except Exception:
        return np.empty((0, 3), dtype=float)

    if df.shape[1] < 5:
        return np.empty((0, 3), dtype=float)

    xyz = df.iloc[:, 2:5].values.astype(float)
    return xyz / COORD_SCALE


def assign_synapses_to_nodes(node_xyz, syn_xyz):
    counts = np.zeros(len(node_xyz), dtype=int)

    if len(syn_xyz) == 0 or len(node_xyz) == 0:
        return counts, np.empty(0, dtype=int), np.empty(0, dtype=float)

    tree = cKDTree(node_xyz)
    dist, idx = tree.query(syn_xyz, k=1)

    idx = np.asarray(idx, dtype=int)
    dist = np.asarray(dist, dtype=float)

    counts += np.bincount(idx, minlength=len(node_xyz))

    return counts, idx, dist


def output_path_for(swc_path):
    swc_path = Path(swc_path)

    if CONFIG["preserve_subdirs"]:
        rel = swc_path.relative_to(GLOBAL_SWC_DIR)
        return GLOBAL_OUT_DIR / rel

    return GLOBAL_OUT_DIR / swc_path.name


def detail_path_for(swc_path):
    swc_path = Path(swc_path)

    if CONFIG["preserve_subdirs"]:
        rel = swc_path.relative_to(GLOBAL_SWC_DIR)
        return (
            GLOBAL_OUT_DIR
            / CONFIG["detail_dirname"]
            / rel.parent
            / f"{swc_path.stem}_synapse_assignment_details.csv"
        )

    return (
        GLOBAL_OUT_DIR
        / CONFIG["detail_dirname"]
        / f"{swc_path.stem}_synapse_assignment_details.csv"
    )


def build_syn_detail_df(syn_xyz, nearest_idx, dist_um, swc_df, syn_type):
    if len(syn_xyz) == 0:
        return pd.DataFrame(
            columns=[
                "syn_type",
                "syn_index",
                "syn_x_um",
                "syn_y_um",
                "syn_z_um",
                "assigned_node_row",
                "assigned_node_id",
                "assigned_node_type",
                "assigned_x_um",
                "assigned_y_um",
                "assigned_z_um",
                "distance_um",
            ]
        )

    assigned = swc_df.iloc[nearest_idx].reset_index(drop=True)

    detail = pd.DataFrame({
        "syn_type": syn_type,
        "syn_index": np.arange(len(syn_xyz), dtype=int),
        "syn_x_um": syn_xyz[:, 0],
        "syn_y_um": syn_xyz[:, 1],
        "syn_z_um": syn_xyz[:, 2],
        "assigned_node_row": nearest_idx,
        "assigned_node_id": assigned["ID"].values.astype(int),
        "assigned_node_type": assigned["Type"].values.astype(int),
        "assigned_x_um": assigned["X"].values.astype(float) / COORD_SCALE[0],
        "assigned_y_um": assigned["Y"].values.astype(float) / COORD_SCALE[1],
        "assigned_z_um": assigned["Z"].values.astype(float) / COORD_SCALE[2],
        "distance_um": dist_um,
    })

    return detail


def process_one(swc_path):
    try:
        swc_path = Path(swc_path)
        name = swc_path.stem

        df = read_swc(swc_path)
        node_xyz = df[["X", "Y", "Z"]].values.astype(float) / COORD_SCALE

        pre_xyz = read_syn_xyz(GLOBAL_PRE_MAP.get(name))
        post_xyz = read_syn_xyz(GLOBAL_POST_MAP.get(name))

        pre_count, pre_idx, pre_dist = assign_synapses_to_nodes(node_xyz, pre_xyz)
        post_count, post_idx, post_dist = assign_synapses_to_nodes(node_xyz, post_xyz)

        df["pre_syn_count"] = pre_count
        df["post_syn_count"] = post_count

        out_path = output_path_for(swc_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            out_path,
            sep=" ",
            index=False,
            header=False,
            float_format="%.3f"
        )

        pre_detail = build_syn_detail_df(pre_xyz, pre_idx, pre_dist, df, "pre")
        post_detail = build_syn_detail_df(post_xyz, post_idx, post_dist, df, "post")
        detail_df = pd.concat([pre_detail, post_detail], ignore_index=True)

        detail_path = detail_path_for(swc_path)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_path, index=False)

        return {
            "file": swc_path.name,
            "relative_path": str(swc_path.relative_to(GLOBAL_SWC_DIR)),
            "n_nodes": len(df),
            "n_pre_syn": int(len(pre_xyz)),
            "n_post_syn": int(len(post_xyz)),
            "assigned_pre_syn": int(pre_count.sum()),
            "assigned_post_syn": int(post_count.sum()),
            "nodes_with_pre": int((pre_count > 0).sum()),
            "nodes_with_post": int((post_count > 0).sum()),
            "nodes_with_both": int(((pre_count > 0) & (post_count > 0)).sum()),
            "has_pre_file": name in GLOBAL_PRE_MAP,
            "has_post_file": name in GLOBAL_POST_MAP,
            "pre_dist_mean_um": float(pre_dist.mean()) if len(pre_dist) else np.nan,
            "pre_dist_median_um": float(np.median(pre_dist)) if len(pre_dist) else np.nan,
            "pre_dist_p95_um": float(np.percentile(pre_dist, 95)) if len(pre_dist) else np.nan,
            "pre_dist_max_um": float(pre_dist.max()) if len(pre_dist) else np.nan,
            "post_dist_mean_um": float(post_dist.mean()) if len(post_dist) else np.nan,
            "post_dist_median_um": float(np.median(post_dist)) if len(post_dist) else np.nan,
            "post_dist_p95_um": float(np.percentile(post_dist, 95)) if len(post_dist) else np.nan,
            "post_dist_max_um": float(post_dist.max()) if len(post_dist) else np.nan,
            "detail_file": str(detail_path.relative_to(GLOBAL_OUT_DIR)),
            "error": None,
        }

    except Exception as e:
        return {
            "file": Path(swc_path).name,
            "relative_path": str(Path(swc_path)),
            "error": str(e),
        }


def summarize_and_plot_distance_distribution(out_dir):
    out_dir = Path(out_dir)
    detail_root = out_dir / CONFIG["detail_dirname"]

    detail_files = sorted(detail_root.rglob("*_synapse_assignment_details.csv"))

    if len(detail_files) == 0:
        print("\nNo synapse detail files found, skip summary plot.")
        return

    dist_frames = []
    for f in tqdm(detail_files, desc="Reading detail files"):
        try:
            df = pd.read_csv(f, usecols=["syn_type", "distance_um"])
            if len(df) > 0:
                dist_frames.append(df)
        except Exception:
            continue

    if len(dist_frames) == 0:
        print("\nNo valid synapse detail data found, skip summary plot.")
        return

    dist_df = pd.concat(dist_frames, ignore_index=True)

    summary = (
        dist_df
        .groupby("syn_type")["distance_um"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            min="min",
            p90=lambda x: np.percentile(x, 90),
            p95=lambda x: np.percentile(x, 95),
            p99=lambda x: np.percentile(x, 99),
            max="max",
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "synapse_distance_summary.csv", index=False)
    dist_df.to_csv(out_dir / "synapse_distance_all.csv", index=False)

    plt.figure(figsize=(8, 5))

    for syn_type, sub_df in dist_df.groupby("syn_type"):
        plt.hist(
            sub_df["distance_um"].values,
            bins=CONFIG["plot_bins"],
            alpha=0.6,
            density=True,
            label=syn_type,
        )

    plt.xlabel("Assignment distance (μm)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "synapse_assignment_distance_distribution.png",
        dpi=CONFIG["plot_dpi"]
    )
    plt.close()


def run():
    swc_dir = Path(CONFIG["swc_dir"])
    out_dir = Path(CONFIG["output_dir"])

    if CONFIG["clear_output"]:
        print("Clearing output directory...")
        clear_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    swc_files = list(swc_dir.rglob("*.swc"))

    pre_map = build_file_map(CONFIG["pre_dir"])
    post_map = build_file_map(CONFIG["post_dir"])

    print("\n===== Input =====")
    print("SWC files:", len(swc_files))
    print("Pre files:", len(pre_map))
    print("Post files:", len(post_map))

    with Pool(
        processes=CONFIG["max_processes"],
        initializer=init_worker,
        initargs=(str(swc_dir), str(out_dir), pre_map, post_map)
    ) as pool:
        logs = list(
            tqdm(
                pool.imap_unordered(
                    process_one,
                    [str(f) for f in swc_files],
                    chunksize=CONFIG["chunksize"]
                ),
                total=len(swc_files),
                desc="Assigning synapses"
            )
        )

    df_log = pd.DataFrame(logs)
    df_log.to_csv(out_dir / "synapse_assignment_log.csv", index=False)

    summarize_and_plot_distance_distribution(out_dir)

    return df_log


if __name__ == "__main__":
    df_log = run()

    print("\n===== Summary =====")
    print("Files:", len(df_log))
    print("Errors:", df_log["error"].notna().sum())

    print("\n===== Synapse assignment =====")
    print("Total pre:", df_log["assigned_pre_syn"].fillna(0).sum())
    print("Total post:", df_log["assigned_post_syn"].fillna(0).sum())

    print("\n===== Node usage =====")
    print("Nodes with pre:", df_log["nodes_with_pre"].fillna(0).sum())
    print("Nodes with post:", df_log["nodes_with_post"].fillna(0).sum())
    print("Nodes with both:", df_log["nodes_with_both"].fillna(0).sum())

    print("\n===== Distance summary =====")
    print("Pre mean distance (um):", df_log["pre_dist_mean_um"].mean(skipna=True))
    print("Pre median distance (um):", df_log["pre_dist_median_um"].median(skipna=True))
    print("Pre p95 distance (um):", df_log["pre_dist_p95_um"].median(skipna=True))
    print("Post mean distance (um):", df_log["post_dist_mean_um"].mean(skipna=True))
    print("Post median distance (um):", df_log["post_dist_median_um"].median(skipna=True))
    print("Post p95 distance (um):", df_log["post_dist_p95_um"].median(skipna=True))
