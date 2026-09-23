from pathlib import Path
import gc, pickle
import navis
import numpy as np
import pandas as pd
from datasketch import MinHash, LeanMinHash
from tqdm.auto import tqdm

INPUT_UNIT = "nm"
VOXEL_UM = 5.0
NUM_PERM = 128
N_HARD = 30
RESAMPLE_UM = 2.0
DOTPROP_K = 5

COLS = ["node_id", "type", "x", "y", "z", "radius", "parent_id"]


def read_swc(path):
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=COLS,
        usecols=range(7), engine="python"
    ).apply(pd.to_numeric, errors="coerce").dropna(subset=COLS)


def xyz_um(path):
    xyz = read_swc(path)[["x", "y", "z"]].to_numpy(float)
    return xyz / 1000 if INPUT_UNIT == "nm" else xyz


def make_minhash(path):
    vox = np.unique(np.floor(xyz_um(path) / VOXEL_UM).astype("<i4"), axis=0)
    mh = MinHash(num_perm=NUM_PERM)
    for v in vox:
        mh.update(v.tobytes())
    return LeanMinHash(mh)


def build_index(raw_dir, cache):
    raw_dir, cache = Path(raw_dir), Path(cache)
    paths = sorted(raw_dir.glob("*.swc"))
    fingerprint = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]

    if cache.exists():
        with open(cache, "rb") as f:
            x = pickle.load(f)
        if x["fingerprint"] == fingerprint:
            print(f"Loaded MinHash index: {len(x['paths']):,} neurons")
            return x

    sig, path_map, failed = {}, {}, []
    for p in tqdm(paths, desc="Building raw MinHash index"):
        try:
            sig[p.name] = make_minhash(p)
            path_map[p.name] = str(p)
        except Exception as e:
            failed.append((p.name, repr(e)))

    x = {"fingerprint": fingerprint, "signatures": sig, "paths": path_map}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if failed:
        pd.DataFrame(failed, columns=["filename", "error"]).to_csv(
            cache.parent/"index_failures_clean.csv", index=False
        )

    print(f"Indexed: {len(path_map):,}")
    return x


def to_dp(path, neuron_id):
    n = navis.read_swc(str(path))
    if isinstance(n, navis.NeuronList):
        n = n[0]

    scale = 1 / 1000 if INPUT_UNIT == "nm" else 1
    n.nodes.loc[:, ["x", "y", "z"]] *= scale

    if "radius" in n.nodes:
        n.nodes.loc[:, "radius"] = (
            pd.to_numeric(n.nodes["radius"], errors="coerce").fillna(0) * scale
        )

    n.id = n.name = neuron_id
    n.units = "1 micrometer"
    n = navis.resample_skeleton(n, resample_to=RESAMPLE_UM, inplace=False)

    dp = navis.make_dotprops(n, k=DOTPROP_K, resample=False)
    dp.id = dp.name = neuron_id
    dp.units = "1 micrometer"
    return dp


def exact_scores(query_path, names, path_map):
    q = to_dp(query_path, f"query::{query_path.name}")
    targets = navis.NeuronList([to_dp(path_map[n], n) for n in names])

    score = navis.nblast(
        navis.NeuronList([q]), targets,
        scores="mean", normalized=True, progress=False
    )

    return pd.DataFrame({
        "candidate_filename": names,
        "nblast_score": np.asarray(score, float).reshape(-1)
    })


def search_one(repair_path, index):
    expected = repair_path.name
    sig, path_map = index["signatures"], index["paths"]

    if expected not in path_map:
        raise FileNotFoundError(f"Missing same-name raw: {expected}")

    q = make_minhash(repair_path)

    approx = pd.DataFrame([
        (name, q.jaccard(s))
        for name, s in sig.items()
    ], columns=["candidate_filename", "minhash_jaccard"])

    approx = approx.sort_values(
        ["minhash_jaccard", "candidate_filename"],
        ascending=[False, True]
    )

    # Always select exactly N_HARD unmatched competitors
    hard = (
        approx[approx["candidate_filename"] != expected]
        .head(N_HARD)
        .copy()
    )

    if len(hard) < N_HARD:
        raise ValueError(f"{expected}: only {len(hard)} unmatched candidates")

    names = [expected] + hard["candidate_filename"].tolist()

    exact = exact_scores(repair_path, names, path_map).merge(
        approx, on="candidate_filename", how="left"
    )

    exact["is_same_name_original"] = exact["candidate_filename"].eq(expected)
    exact = exact.sort_values(
        ["nblast_score", "minhash_jaccard"],
        ascending=[False, False]
    ).reset_index(drop=True)

    exact["exact_rank"] = np.arange(1, len(exact) + 1)
    exact["query_filename"] = expected

    paired = exact[exact["is_same_name_original"]].iloc[0]
    unmatched = exact[~exact["is_same_name_original"]].iloc[0]

    summary = {
        "query_filename": expected,
        "paired_nblast_score": paired["nblast_score"],
        "paired_exact_rank": int(paired["exact_rank"]),
        "best_unmatched_filename": unmatched["candidate_filename"],
        "best_unmatched_nblast_score": unmatched["nblast_score"],
        "delta_nblast": paired["nblast_score"] - unmatched["nblast_score"],
        "top1_is_same_name": int(paired["exact_rank"]) == 1,
        "paired_minhash_jaccard": paired["minhash_jaccard"],
        "n_hard_negatives": N_HARD
    }

    return summary, exact


def append(df, path):
    df.to_csv(path, mode="a", index=False, header=not Path(path).exists())


def run_all(original_dir, repaired_dir, output_dir, resume=True, limit=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_file = out/"nblast_summary_clean.csv"
    candidate_file = out/"nblast_candidates_clean.csv"
    error_file = out/"nblast_errors_clean.csv"
    cache_file = out/"raw_minhash_index_clean.pkl"

    index = build_index(original_dir, cache_file)

    completed = set()
    if resume and summary_file.exists():
        completed = set(pd.read_csv(summary_file)["query_filename"].astype(str))

        # Remove partial candidate records from interrupted runs
        if candidate_file.exists():
            c = pd.read_csv(candidate_file)
            c = c[c["query_filename"].astype(str).isin(completed)]
            c.drop_duplicates(
                ["query_filename", "candidate_filename"], keep="last"
            ).to_csv(candidate_file, index=False)

    paths = [
        p for p in sorted(Path(repaired_dir).glob("*.swc"))
        if p.name not in completed
    ]

    if limit is not None:
        paths = paths[:limit]

    for p in tqdm(paths, desc="NBLAST"):
        try:
            summary, candidates = search_one(p, index)

            # Candidates first: avoids summary marking an incomplete query
            append(candidates, candidate_file)
            append(pd.DataFrame([summary]), summary_file)

        except Exception as e:
            append(pd.DataFrame([{
                "query_filename": p.name,
                "error": repr(e)
            }]), error_file)

        gc.collect()

    r = pd.read_csv(summary_file)

    print(f"Completed: {len(r):,}")
    print(f"Top-1 retention: {r['top1_is_same_name'].mean()*100:.2f}%")
    print(f"Median paired NBLAST: {r['paired_nblast_score'].median():.4f}")
    print(f"Median delta NBLAST: {r['delta_nblast'].median():.4f}")
    print(f"Min hard negatives: {r['n_hard_negatives'].min()}")

    return r


if __name__ == "__main__":
    results = run_all(
        original_dir="/data/project_backup/swc_fix/sample1/raw",
        repaired_dir="/data/project_backup/swc_fix/sample1/repair",
        output_dir="/data/project_backup/swc_fix/sample1",
        resume=True,
        limit=None,
    )
