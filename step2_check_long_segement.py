from pathlib import Path
from multiprocessing import Pool
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

CFG = {
    "src_dir": "step1_out",
    "dst_dir": "step2_out",
    "check_dir_name": "check",
    "coord_scale": [1000, 1000, 1000],
    "length_threshold_um": 5.0,
    "min_total_children": 2,
    "min_long_children": 2,
    "processes": min(24, os.cpu_count() or 1),
    "chunksize": 20,
    "maxtasksperchild": 2000,
    "clear_output": True
}

ROOT = None
SCALE = np.asarray(CFG["coord_scale"], dtype=float)


def init_worker(root):
    global ROOT
    ROOT = Path(root)


def read_swc_fast(path):
    lines = []

    with open(path, "rb") as f:
        for line in f:
            s = line.lstrip()
            if s and not s.startswith(b"#"):
                lines.append(line)

    if not lines:
        raise ValueError("Empty SWC")

    try:
        data = np.fromstring(
            b"".join(lines).decode("ascii", errors="ignore"),
            sep=" "
        )

        n_rows = len(lines)

        if data.size % n_rows != 0:
            raise ValueError

        n_cols = data.size // n_rows

        if n_cols < 7:
            raise ValueError

        data = data.reshape(n_rows, n_cols)

        return data[:, [0, 2, 3, 4, 6]]

    except Exception:
        return np.loadtxt(
            path,
            comments="#",
            usecols=(0, 2, 3, 4, 6),
            ndmin=2
        )


def process_one(path):
    path = Path(path)
    relative_path = path.relative_to(ROOT)

    try:
        data = read_swc_fast(path)

        ids = data[:, 0].astype(np.int64)
        xyz = data[:, 1:4].astype(float) / SCALE
        parents = data[:, 4].astype(np.int64)

        n_nodes = len(ids)

        if n_nodes == 0:
            raise ValueError("No valid nodes")

        order = np.argsort(ids)
        sorted_ids = ids[order]

        if np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError("Duplicate node IDs")

        nonroot = np.flatnonzero(parents != -1)

        if not len(nonroot):
            return {
                "file": path.name,
                "relative_path": str(relative_path),
                "n_nodes": n_nodes,
                "n_valid_segments": 0,
                "n_branch_parents": 0,
                "n_matching_parents": 0,
                "n_matching_long_segments": 0,
                "max_long_children_same_parent": 0,
                "matched": False,
                "error": ""
            }

        parent_pos = np.searchsorted(
            sorted_ids,
            parents[nonroot]
        )

        valid = parent_pos < n_nodes
        valid_idx = np.flatnonzero(valid)

        valid[valid_idx] = (
            sorted_ids[parent_pos[valid_idx]]
            == parents[nonroot[valid_idx]]
        )

        child_idx = nonroot[valid]
        parent_idx = order[parent_pos[valid]]

        if not len(child_idx):
            return {
                "file": path.name,
                "relative_path": str(relative_path),
                "n_nodes": n_nodes,
                "n_valid_segments": 0,
                "n_branch_parents": 0,
                "n_matching_parents": 0,
                "n_matching_long_segments": 0,
                "max_long_children_same_parent": 0,
                "matched": False,
                "error": ""
            }

        lengths = np.linalg.norm(
            xyz[child_idx] - xyz[parent_idx],
            axis=1
        )

        finite = np.isfinite(lengths)

        child_idx = child_idx[finite]
        parent_idx = parent_idx[finite]
        lengths = lengths[finite]

        child_count = np.bincount(
            parent_idx,
            minlength=n_nodes
        ).astype(np.int32)

        long_mask = (
            lengths > CFG["length_threshold_um"]
        )

        long_child_count = np.bincount(
            parent_idx[long_mask],
            minlength=n_nodes
        ).astype(np.int32)

        branch_parent_mask = (
            child_count >= CFG["min_total_children"]
        )

        matching_parent_mask = (
            branch_parent_mask
            & (
                long_child_count
                >= CFG["min_long_children"]
            )
        )

        matching_parents = np.flatnonzero(
            matching_parent_mask
        )

        matching_long_segments = int(
            long_child_count[matching_parents].sum()
        )

        return {
            "file": path.name,
            "relative_path": str(relative_path),
            "n_nodes": n_nodes,
            "n_valid_segments": len(lengths),
            "n_branch_parents": int(
                branch_parent_mask.sum()
            ),
            "n_matching_parents": len(
                matching_parents
            ),
            "n_matching_long_segments": (
                matching_long_segments
            ),
            "max_long_children_same_parent": (
                int(long_child_count.max())
                if len(long_child_count)
                else 0
            ),
            "matched": bool(
                len(matching_parents)
            ),
            "error": ""
        }

    except Exception as e:
        return {
            "file": path.name,
            "relative_path": str(relative_path),
            "n_nodes": np.nan,
            "n_valid_segments": np.nan,
            "n_branch_parents": np.nan,
            "n_matching_parents": 0,
            "n_matching_long_segments": 0,
            "max_long_children_same_parent": 0,
            "matched": False,
            "error": str(e)
        }


def run():
    src = Path(CFG["src_dir"]).resolve()
    dst = Path(CFG["dst_dir"]).resolve()
    check = dst / CFG["check_dir_name"]

    if not src.exists():
        raise FileNotFoundError(src)

    if src == dst:
        raise ValueError(
            "Source and destination directories must be different"
        )

    if src in dst.parents:
        raise ValueError(
            "Destination directory cannot be inside source directory"
        )

    if CFG["clear_output"] and dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    check.mkdir(parents=True, exist_ok=True)

    files = sorted(src.rglob("*.swc"))

    print("Source:", src)
    print("Destination:", dst)
    print("Check directory:", check)
    print("SWC files:", len(files))
    print(
        "Criterion:",
        f">={CFG['min_long_children']} child segments "
        f"> {CFG['length_threshold_um']} μm "
        f"from the same parent"
    )

    results = []

    with Pool(
        processes=CFG["processes"],
        initializer=init_worker,
        initargs=(str(src),),
        maxtasksperchild=CFG["maxtasksperchild"]
    ) as pool:

        iterator = pool.imap_unordered(
            process_one,
            map(str, files),
            chunksize=CFG["chunksize"]
        )

        for result in tqdm(
            iterator,
            total=len(files),
            desc="Screening SWC files"
        ):
            results.append(result)

            relative_path = Path(
                result["relative_path"]
            )

            source_file = src / relative_path

            if result["matched"]:
                target_file = (
                    check / relative_path
                )
            else:
                target_file = (
                    dst / relative_path
                )

            target_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                source_file,
                target_file
            )

    df = pd.DataFrame(results)

    matched = df[
        df["matched"] == True
    ].copy()

    unmatched = df[
        df["matched"] == False
    ].copy()

    errors = df[
        df["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    matched = matched.sort_values(
        [
            "n_matching_parents",
            "n_matching_long_segments",
            "max_long_children_same_parent"
        ],
        ascending=[False, False, False]
    )

    df.to_csv(
        dst / "screening_summary.csv",
        index=False
    )

    matched.to_csv(
        dst / "matching_files.csv",
        index=False
    )

    unmatched.to_csv(
        dst / "non_matching_files.csv",
        index=False
    )

    matched[
        ["file", "relative_path"]
    ].to_csv(
        dst / "matching_swc_filenames.csv",
        index=False
    )

    if len(errors):
        errors.to_csv(
            dst / "screening_errors.csv",
            index=False
        )

    folder_summary = (
        df.assign(
            folder=df["relative_path"].map(
                lambda x: str(Path(x).parent)
            )
        )
        .groupby("folder")
        .agg(
            total_files=("file", "size"),
            matching_files=("matched", "sum")
        )
        .reset_index()
    )

    folder_summary["non_matching_files"] = (
        folder_summary["total_files"]
        - folder_summary["matching_files"]
    )

    folder_summary.to_csv(
        dst / "folder_summary.csv",
        index=False
    )

    print("\n===== Summary =====")

    print(
        "Total SWC files:",
        len(df)
    )

    print(
        "Successfully processed:",
        len(df) - len(errors)
    )

    print(
        "Errors:",
        len(errors)
    )

    print(
        "Files moved to check:",
        len(matched)
    )

    print(
        "Files kept outside check:",
        len(unmatched)
    )

    if len(df):
        print(
            "Matching fraction:",
            f"{len(matched) / len(df) * 100:.4f}%"
        )

    print(
        "Matching parent nodes:",
        int(
            matched[
                "n_matching_parents"
            ].sum()
        )
    )

    print(
        "Matching long segments:",
        int(
            matched[
                "n_matching_long_segments"
            ].sum()
        )
    )

    print("\nOutput:")
    print(dst)

    print("\nMatching SWCs:")
    print(check)


if __name__ == "__main__":
    run()
