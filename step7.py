import shutil
import csv
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm


CONFIG = {
    "input_dir": "step6_out",
    "output_dir": "step7_out",
    "max_processes": 30,
    "chunksize": 300,
    "clear_output": True,
}


INPUT_DIR = None
OUTPUT_DIR = None
GENERATED_DATE = None


def clear_output_dir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def make_header():
    return f"""# Enhanced SWC for neuronal morphology and synapse polarity visualization
#
# Author: Longxiao Yuan
# Generated: {GENERATED_DATE}
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
# 5 = branch point
# 6 = terminal point
#
"""


def process_one(path):
    try:
        path = Path(path)
        rel = path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Identify top-level group
        group = rel.parts[0] if len(rel.parts) > 1 else ""
        is_soma_pruned = group == "soma_pruned"

        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                x = line.split()

                if len(x) < 7:
                    raise ValueError(
                        f"{path.name}: invalid SWC row with {len(x)} columns"
                    )

                rows.append(x)

        if not rows:
            raise ValueError(f"{path.name}: no SWC data")

        ids = [int(float(x[0])) for x in rows]
        types = [int(float(x[1])) for x in rows]
        parents = [int(float(x[6])) for x in rows]

        id_set = set(ids)

        # Number of children for every node
        child_count = {node: 0 for node in ids}

        for parent in parents:
            if parent in id_set:
                child_count[parent] += 1

        roots = [
            i for i, parent in enumerate(parents)
            if parent == -1
        ]

        n_root_to_1 = 0
        n_type1_removed = 0
        n_type0_to_5 = 0

        # ------------------------------------------------
        # Rule 1: soma_pruned root -> Type 1
        # ------------------------------------------------
        if is_soma_pruned:
            if len(roots) != 1:
                raise ValueError(
                    f"{path.name}: expected 1 root in soma_pruned, "
                    f"found {len(roots)}"
                )

            r = roots[0]

            if types[r] != 1:
                types[r] = 1
                n_root_to_1 += 1

        # ------------------------------------------------
        # Rule 2: non-soma_pruned must not contain Type 1
        # ------------------------------------------------
        else:
            for i, node in enumerate(ids):

                if types[i] != 1:
                    continue

                if child_count[node] >= 2:
                    types[i] = 5
                else:
                    types[i] = 0

                n_type1_removed += 1

        # ------------------------------------------------
        # Rule 3: Type 0 branch point -> Type 5
        # ------------------------------------------------
        for i, node in enumerate(ids):

            if types[i] == 0 and child_count[node] >= 2:
                types[i] = 5
                n_type0_to_5 += 1

        # Update only Type column
        for i, t in enumerate(types):
            rows[i][1] = str(t)

        # Final validation
        final_type1 = sum(t == 1 for t in types)

        if is_soma_pruned:
            root_type = types[roots[0]]

            if root_type != 1:
                raise RuntimeError(
                    f"{path.name}: root is not Type 1 after correction"
                )

        else:
            if final_type1 != 0:
                raise RuntimeError(
                    f"{path.name}: Type 1 remains outside soma_pruned"
                )

        # Check no Type-0 branch points remain
        remaining_type0_branch = sum(
            types[i] == 0 and child_count[node] >= 2
            for i, node in enumerate(ids)
        )

        if remaining_type0_branch:
            raise RuntimeError(
                f"{path.name}: {remaining_type0_branch} "
                f"Type-0 branch points remain"
            )

        # Write
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(make_header())

            for row in rows:
                f.write(" ".join(row) + "\n")

        return {
            "relative_path": str(rel),
            "group": group,
            "n_nodes": len(rows),
            "n_roots": len(roots),
            "type1_final": final_type1,
            "root_to_type1": n_root_to_1,
            "type1_removed": n_type1_removed,
            "type0_branch_to_type5": n_type0_to_5,
            "error": "",
        }

    except Exception as e:
        return {
            "relative_path": str(path),
            "group": "",
            "n_nodes": "",
            "n_roots": "",
            "type1_final": "",
            "root_to_type1": "",
            "type1_removed": "",
            "type0_branch_to_type5": "",
            "error": str(e),
        }


def init_worker(input_dir, output_dir, generated_date):
    global INPUT_DIR, OUTPUT_DIR, GENERATED_DATE

    INPUT_DIR = Path(input_dir)
    OUTPUT_DIR = Path(output_dir)
    GENERATED_DATE = generated_date


def run():
    input_dir = Path(CONFIG["input_dir"]).resolve()
    output_dir = Path(CONFIG["output_dir"]).resolve()

    generated_date = datetime.now().strftime("%Y-%m-%d")

    if CONFIG["clear_output"]:
        print("Clearing output directory...")
        clear_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".swc"
    ]

    print("\n===== Input =====")
    print("Input :", input_dir)
    print("Output:", output_dir)
    print("Date  :", generated_date)
    print("SWCs  :", f"{len(files):,}")

    with Pool(
        processes=CONFIG["max_processes"],
        initializer=init_worker,
        initargs=(
            str(input_dir),
            str(output_dir),
            generated_date,
        ),
    ) as pool:

        logs = list(
            tqdm(
                pool.imap_unordered(
                    process_one,
                    files,
                    chunksize=CONFIG["chunksize"],
                ),
                total=len(files),
                desc="Processing",
            )
        )

    log_path = output_dir / "release_processing_log.csv"

    fields = [
        "relative_path",
        "group",
        "n_nodes",
        "n_roots",
        "type1_final",
        "root_to_type1",
        "type1_removed",
        "type0_branch_to_type5",
        "error",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(logs)

    return logs


if __name__ == "__main__":

    logs = run()

    good = [x for x in logs if not x["error"]]
    bad = [x for x in logs if x["error"]]

    print("\n===== Summary =====")
    print("Files       :", f"{len(logs):,}")
    print("Successful  :", f"{len(good):,}")
    print("Errors      :", f"{len(bad):,}")

    print("\n===== Corrections =====")
    print(
        "Root -> Type 1:",
        f"{sum(int(x['root_to_type1']) for x in good):,}"
    )
    print(
        "Type 1 removed outside soma_pruned:",
        f"{sum(int(x['type1_removed']) for x in good):,}"
    )
    print(
        "Type 0 branch -> Type 5:",
        f"{sum(int(x['type0_branch_to_type5']) for x in good):,}"
    )

    if bad:
        print("\nFirst errors:")
        for x in bad[:20]:
            print(
                x["relative_path"],
                "->",
                x["error"]
            )
