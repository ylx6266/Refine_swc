from pathlib import Path
from multiprocessing import Pool
import csv, os, shutil
import numpy as np
from tqdm import tqdm

CFG = {
    "input_dir": "sk_lod1_783_healed_260609",
    "output_dir": "step0_out",
    "processes": min(8, os.cpu_count() or 1),
    "chunksize": 1,
    "maxtasksperchild": 500,
    "resume": True,
    "hardlink_unchanged": True
}

ROOT = OUT = None
FIELDS = [
    "file", "relative_path", "n_nodes", "type0_before", "type5_before",
    "retyped_type0_to_type5", "type0_after", "type5_after",
    "output_method", "error"
]


def init_worker(root, out):
    global ROOT, OUT
    ROOT, OUT = Path(root), Path(out)


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if CFG["hardlink_unchanged"]:
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def parse_swc(path):
    rows, ids, types, parents = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                rows.append((line, None))
                continue
            p = line.split()
            if len(p) < 7:
                rows.append((line, None))
                continue
            try:
                node_id = int(float(p[0]))
                node_type = int(float(p[1]))
                parent_id = int(float(p[6]))
            except ValueError:
                rows.append((line, None))
                continue
            rows.append((line, p))
            ids.append(node_id)
            types.append(node_type)
            parents.append(parent_id)
    return rows, np.asarray(ids, np.int64), np.asarray(types, np.int16), np.asarray(parents, np.int64)


def process_one(path):
    path = Path(path)
    rel = path.relative_to(ROOT)
    dst = OUT / rel
    try:
        rows, ids, types, parents = parse_swc(path)
        if not ids.size:
            raise ValueError("No valid SWC nodes")
        if np.unique(ids).size != ids.size:
            raise ValueError("Duplicate node IDs")

        parent_ids, counts = np.unique(parents[parents != -1], return_counts=True)
        n_children = np.zeros(ids.size, np.int32)
        if parent_ids.size:
            pos = np.searchsorted(parent_ids, ids)
            valid = pos < parent_ids.size
            matched = np.zeros(ids.size, bool)
            matched[valid] = parent_ids[pos[valid]] == ids[valid]
            n_children[matched] = counts[pos[matched]]

        change = (types == 0) & (n_children >= 2)
        changed_ids = set(ids[change].tolist())

        if changed_ids:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for line, parts in rows:
                    if parts is None:
                        f.write(line)
                    elif int(float(parts[0])) in changed_ids and int(float(parts[1])) == 0:
                        parts[1] = "5"
                        f.write(" ".join(parts) + "\n")
                    else:
                        f.write(line)
            os.replace(tmp, dst)
            method = "rewritten"
        else:
            method = link_or_copy(path, dst)

        n_changed = int(change.sum())
        return {
            "file": path.name,
            "relative_path": str(rel),
            "n_nodes": int(ids.size),
            "type0_before": int((types == 0).sum()),
            "type5_before": int((types == 5).sum()),
            "retyped_type0_to_type5": n_changed,
            "type0_after": int((types == 0).sum() - n_changed),
            "type5_after": int((types == 5).sum() + n_changed),
            "output_method": method,
            "error": ""
        }
    except Exception as e:
        return {
            **{k: "" for k in FIELDS},
            "file": path.name,
            "relative_path": str(rel),
            "error": str(e)
        }


def completed(log_path):
    if not CFG["resume"] or not log_path.exists():
        return set()
    done = set()
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("relative_path") and not r.get("error"):
                    done.add(r["relative_path"])
    except Exception:
        pass
    return done


def run():
    root, out = Path(CFG["input_dir"]), Path(CFG["output_dir"])
    if root.resolve() == out.resolve():
        raise ValueError("input_dir and output_dir must be different")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "type0_to_type5_log.csv"
    done = completed(log_path)
    files = [p for p in sorted(root.rglob("*.swc")) if str(p.relative_to(root)) not in done]
    print("Remaining:", len(files))

    exists = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        with Pool(
            CFG["processes"], initializer=init_worker,
            initargs=(str(root), str(out)),
            maxtasksperchild=CFG["maxtasksperchild"]
        ) as pool:
            it = pool.imap_unordered(process_one, map(str, files), chunksize=CFG["chunksize"])
            for i, result in enumerate(tqdm(it, total=len(files), desc="Relabeling Type-0 branches"), 1):
                writer.writerow(result)
                if i % 100 == 0:
                    f.flush()

    rows, changed, errors = 0, 0, 0
    with log_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows += 1
            errors += bool(r.get("error"))
            try:
                changed += int(r.get("retyped_type0_to_type5") or 0)
            except ValueError:
                pass
    print("SWC files:", rows)
    print("Errors:", errors)
    print("Type-0 nodes relabeled to Type-5:", changed)
    print("Output:", out)


if __name__ == "__main__":
    run()
