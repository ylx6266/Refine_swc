import shutil
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


CONFIG = {
    "input_dir": "step5_out",
    "output_dir": "step6_out",
    "max_processes": 30,
    "chunksize": 300,
    "clear_output": True,
}


INPUT_DIR = None
OUTPUT_DIR = None


def clear_output_dir(path):
    path = Path(path)

    if path.exists():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)

    path.mkdir(parents=True, exist_ok=True)


def write_visual_swc_with_header(df_out, out_path):
    header = """# Enhanced SWC for synapse polarity visualization
#
# Columns
# ----------------------------------------------------
# 1  ID
# 2  Type
# 3  X
# 4  Y
# 5  Z
# 6  Radius
# 7  Parent
#
# Type definitions
# ----------------------------------------------------
# 0 = unlabeled neurite
# 1 = soma
# 2 = pre-synapse-rich node
# 3 = post-synapse-rich node
# 4 = mixed pre/post synaptic node
#
# Generation rule
# ----------------------------------------------------
# pre>0  and post==0  -> Type 2
# pre==0 and post>0   -> Type 3
# pre>0  and post>0   -> Type 4
#
# Original synapse count columns are removed
# in this visualization version.
#
"""

    with open(out_path, "w") as f:
        f.write(header)
        df_out.to_csv(
            f,
            sep=" ",
            index=False,
            header=False,
            float_format="%.3f"
        )


def process_one(path):
    try:
        path = Path(path)
        rel = path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None
        )

        if df.shape[1] < 9:
            raise ValueError(f"{path.name} has fewer than 9 columns")

        pre = df.iloc[:, 7].astype(float)
        post = df.iloc[:, 8].astype(float)

        pre_only = (pre > 0) & (post == 0)
        post_only = (pre == 0) & (post > 0)
        mixed = (pre > 0) & (post > 0)

        # Preserve Type-1 soma nodes
        is_soma = df.iloc[:, 1].astype(int) == 1

        df.loc[pre_only & ~is_soma, 1] = 2
        df.loc[post_only & ~is_soma, 1] = 3
        df.loc[mixed & ~is_soma, 1] = 4

        df_out = df.iloc[:, :7].copy()
        write_visual_swc_with_header(df_out, out_path)

        return {
            "file": path.name,
            "relative_path": str(rel),
            "n_nodes": len(df),
            "pre_rich_nodes": int(pre_only.sum()),
            "post_rich_nodes": int(post_only.sum()),
            "mixed_nodes": int(mixed.sum()),
            "syn_nodes": int(((pre > 0) | (post > 0)).sum()),
            "error": None
        }

    except Exception as e:
        return {
            "file": Path(path).name,
            "relative_path": str(path),
            "error": str(e)
        }


def init_worker(input_dir, output_dir):
    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR = Path(input_dir)
    OUTPUT_DIR = Path(output_dir)


def run():
    input_dir = Path(CONFIG["input_dir"])
    output_dir = Path(CONFIG["output_dir"])

    if CONFIG["clear_output"]:
        print("Clearing output directory...")
        clear_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.rglob("*.swc"))

    print("\n===== Input =====")
    print("SWC files:", len(files))

    with Pool(
        processes=CONFIG["max_processes"],
        initializer=init_worker,
        initargs=(str(input_dir), str(output_dir))
    ) as pool:
        logs = list(
            tqdm(
                pool.imap_unordered(
                    process_one,
                    files,
                    chunksize=CONFIG["chunksize"]
                ),
                total=len(files)
            )
        )

    df_log = pd.DataFrame(logs)
    df_log.to_csv(
        output_dir / "syn_type_visual_log.csv",
        index=False
    )

    return df_log


if __name__ == "__main__":
    df_log = run()

    print("\n===== Summary =====")
    print("Files:", len(df_log))
    print("Errors:", df_log["error"].notna().sum())

    print("\n===== Node type assignment =====")
    print("Pre-rich nodes:", df_log["pre_rich_nodes"].fillna(0).sum())
    print("Post-rich nodes:", df_log["post_rich_nodes"].fillna(0).sum())
    print("Mixed nodes:", df_log["mixed_nodes"].fillna(0).sum())
    print("Synaptic nodes:", df_log["syn_nodes"].fillna(0).sum())
