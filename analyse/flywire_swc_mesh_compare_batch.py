from pathlib import Path
import gc, re, time, warnings

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SWC_COLS = ["node_id", "type", "x", "y", "z", "radius", "parent_id"]


def read_swc(path, xyz_unit="nm", radius_unit=None):
    swc = pd.read_csv(
        path, sep=r"\s+", comment="#", names=SWC_COLS,
        usecols=range(7), engine="python"
    ).apply(pd.to_numeric, errors="coerce").dropna()

    if swc.empty:
        raise ValueError(f"No valid SWC rows: {path}")

    swc[["node_id", "type", "parent_id"]] = swc[
        ["node_id", "type", "parent_id"]
    ].astype(np.int64)

    if swc["node_id"].duplicated().any():
        dup = swc.loc[swc["node_id"].duplicated(), "node_id"].iloc[0]
        raise ValueError(f"Duplicate node_id={dup}: {path}")

    scale = {"nm": 1.0, "um": 1000.0, "voxel": np.array([4.0, 4.0, 40.0])}
    if xyz_unit not in scale:
        raise ValueError("xyz_unit must be 'nm', 'um' or 'voxel'")

    swc[["x", "y", "z"]] = swc[["x", "y", "z"]].to_numpy(float) * scale[xyz_unit]

    radius_unit = radius_unit or (None if xyz_unit == "voxel" else xyz_unit)
    if radius_unit not in {"nm", "um"}:
        raise ValueError("radius_unit must be specified as 'nm' or 'um'")

    r = swc["radius"].to_numpy(float) * (1000.0 if radius_unit == "um" else 1.0)
    if np.any(r < 0):
        warnings.warn("Negative radii were clipped to zero.", RuntimeWarning)
    swc["radius"] = np.maximum(r, 0)

    return swc


def sample_edges(swc, spacing_nm=250, max_samples=128):
    if spacing_nm <= 0 or max_samples < 1:
        raise ValueError("spacing_nm must be >0 and max_samples must be >=1")

    lookup = swc.set_index("node_id")[["x", "y", "z", "radius", "type"]]
    pts, radii, weights, edge_ids, soma_flags, rows = [], [], [], [], [], []

    for eid, row in enumerate(swc.itertuples(index=False)):
        pid = int(row.parent_id)
        if pid < 0 or pid not in lookup.index:
            continue

        p = lookup.loc[pid]
        p0 = p[["x", "y", "z"]].to_numpy(float)
        p1 = np.array([row.x, row.y, row.z], float)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))

        if not np.isfinite(length) or length <= 0:
            continue

        n = min(max_samples, max(1, int(np.ceil(length / spacing_nm))))
        t = (np.arange(n) + 0.5) / n
        r0, r1 = float(p.radius), float(row.radius)
        soma = int(p.type) == 1 or int(row.type) == 1

        pts.append(p0 + t[:, None] * vec)
        radii.append(r0 + t * (r1 - r0))
        weights.append(np.full(n, length / n))
        edge_ids.append(np.full(n, len(rows), dtype=np.int64))
        soma_flags.append(np.full(n, soma, dtype=bool))
        rows.append({
            "edge_id": len(rows),
            "child_id": int(row.node_id),
            "parent_id": pid,
            "child_type": int(row.type),
            "parent_type": int(p.type),
            "length_nm": length,
            "n_samples": n,
            "soma_edge": soma,
        })

    if not pts:
        raise ValueError("No valid parent-child edges found")

    samples = {
        "points": np.vstack(pts),
        "radius": np.concatenate(radii),
        "weight": np.concatenate(weights),
        "edge_id": np.concatenate(edge_ids),
        "soma_edge": np.concatenate(soma_flags),
    }
    return samples, pd.DataFrame(rows)


def weighted_quantile(values, q, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    q = np.atleast_1d(q).astype(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not valid.any():
        return np.full(len(q), np.nan)

    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(q, cdf, values)


def kd_query(tree, points):
    try:
        return tree.query(points, k=1, workers=-1)
    except TypeError:
        return tree.query(points, k=1)


def fetch_mesh(root_id, dataset="flat_783", lod=2, threads=5, retries=3):
    from fafbseg import flywire

    last_error = None
    for attempt in range(retries):
        try:
            m = flywire.get_mesh_neuron(
                int(root_id), dataset=dataset, lod=lod,
                threads=threads, progress=False
            )
            v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
            if v.ndim != 2 or v.shape[1] != 3 or not len(v):
                raise ValueError("Invalid mesh vertices")
            if f.ndim != 2 or f.shape[1] != 3 or not len(f):
                raise ValueError("Invalid mesh faces")
            return trimesh.Trimesh(v, f, process=False, validate=False), m
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to fetch mesh: root_id={root_id}, dataset={dataset}, lod={lod}"
    ) from last_error


def sample_surface(mesh, count=80000, seed=42):
    try:
        return trimesh.sample.sample_surface(mesh, int(count), seed=int(seed))[0]
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return trimesh.sample.sample_surface(mesh, int(count))[0]
        finally:
            np.random.set_state(state)


def mesh_surface_distance(mesh, points, exact=True, batch_size=10000):
    if exact:
        try:
            query = trimesh.proximity.ProximityQuery(mesh)
            out = []
            for i in range(0, len(points), batch_size):
                _, d, _ = query.on_surface(points[i:i + batch_size])
                out.append(np.asarray(d, float))
            return np.concatenate(out), "triangle"
        except Exception as exc:
            reason = type(exc).__name__
    else:
        reason = "disabled"

    d, _ = kd_query(cKDTree(np.asarray(mesh.vertices, float)), points)
    return np.asarray(d, float), f"nearest_vertex_fallback:{reason}"


def validate_overlap(points, bounds, margin_nm=1000):
    inside = np.all(
        (points >= bounds[0] - margin_nm) & (points <= bounds[1] + margin_nm),
        axis=1
    )
    frac = float(inside.mean())

    if frac < 0.1:
        raise ValueError(
            "SWC and mesh barely overlap; check root_id, units, dataset and coordinates"
        )
    if frac < 0.9:
        warnings.warn(f"Only {frac:.1%} of SWC samples fall in the mesh bounding box")

    return inside


def edge_diagnostics(edges, samples, center_err, center_excess, surface_err, nearest_edge):
    center = pd.DataFrame({
        "edge_id": samples["edge_id"],
        "weight": samples["weight"],
        "error": center_err,
        "excess": center_excess,
    })

    center_rows = []
    for eid, g in center.groupby("edge_id", sort=False):
        w, e, x = g["weight"].to_numpy(), g["error"].to_numpy(), g["excess"].to_numpy()
        center_rows.append({
            "edge_id": int(eid),
            "center_shell_error_mean_nm": np.average(e, weights=w),
            "center_shell_error_p90_nm": weighted_quantile(e, 0.9, w)[0],
            "center_excess_mean_nm": np.average(x, weights=w),
            "center_fit_fraction_500nm": np.average(e <= 500, weights=w),
        })

    surface = pd.DataFrame({"edge_id": nearest_edge, "error": surface_err})
    surface_rows = []
    for eid, g in surface.groupby("edge_id", sort=False):
        e = g["error"].to_numpy()
        surface_rows.append({
            "edge_id": int(eid),
            "surface_points_assigned": len(e),
            "surface_shell_error_mean_nm": e.mean(),
            "surface_shell_error_p90_nm": np.quantile(e, 0.9),
            "surface_fit_fraction_500nm": np.mean(e <= 500),
        })

    return (
        edges
        .merge(pd.DataFrame(center_rows), on="edge_id", how="left")
        .merge(pd.DataFrame(surface_rows), on="edge_id", how="left")
        .sort_values(
            ["center_shell_error_p90_nm", "surface_shell_error_p90_nm"],
            ascending=False, na_position="last"
        )
        .reset_index(drop=True)
    )


def compare_one(
    name, swc, mesh, surface_points, spacing_nm=250,
    max_samples=128, thresholds=(250, 500, 1000), exact=True
):
    s, edges = sample_edges(swc, spacing_nm, max_samples)
    p, r, w = s["points"], s["radius"], s["weight"]

    inside = validate_overlap(p, np.asarray(mesh.bounds, float))
    center_dist, method = mesh_surface_distance(mesh, p, exact=exact)
    center_err = np.abs(center_dist - r)
    center_excess = np.maximum(center_dist - r, 0)

    surface_dist, idx = kd_query(cKDTree(p), surface_points)
    nearest_r = r[idx]
    nearest_edge = s["edge_id"][idx]
    nearest_soma = s["soma_edge"][idx]
    surface_err = np.abs(surface_dist - nearest_r)
    surface_excess = np.maximum(surface_dist - nearest_r, 0)

    rows = []
    scopes = {
        "all": (np.ones(len(p), bool), np.ones(len(surface_points), bool))
    }

    for scope, (cm, sm) in scopes.items():
        if not cm.any() or not sm.any():
            continue

        cq = weighted_quantile(center_err[cm], [0.5, 0.9, 0.95], w[cm])
        sq = np.quantile(surface_err[sm], [0.5, 0.9, 0.95])

        row = {
            "swc": name,
            "scope": scope,
            "n_nodes": len(swc),
            "n_edges": len(edges),
            "cable_length_um": edges["length_nm"].sum() / 1000,
            "n_center_samples": cm.sum(),
            "n_surface_samples": sm.sum(),
            "center_distance_method": method,
            "center_in_mesh_aabb_fraction": np.average(inside[cm], weights=w[cm]),
            "center_to_surface_p50_nm": weighted_quantile(center_dist[cm], 0.5, w[cm])[0],
            "center_shell_error_p50_nm": cq[0],
            "center_shell_error_p90_nm": cq[1],
            "center_shell_error_p95_nm": cq[2],
            "surface_shell_error_p50_nm": sq[0],
            "surface_shell_error_p90_nm": sq[1],
            "surface_shell_error_p95_nm": sq[2],
            "center_excess_p90_nm": weighted_quantile(center_excess[cm], 0.9, w[cm])[0],
            "surface_excess_p90_nm": np.quantile(surface_excess[sm], 0.9),
        }

        for t in map(int, thresholds):
            cable_fit = np.average(center_err[cm] <= t, weights=w[cm])
            surface_fit = np.mean(surface_err[sm] <= t)
            f1 = 0 if cable_fit + surface_fit == 0 else (
                2 * cable_fit * surface_fit / (cable_fit + surface_fit)
            )
            row.update({
                f"cable_fit_fraction_{t}nm": cable_fit,
                f"surface_fit_fraction_{t}nm": surface_fit,
                f"bidirectional_f1_{t}nm": f1,
                f"match_score_{t}nm": 100 * f1,
            })

        rows.append(row)

    diag = edge_diagnostics(
        edges, s, center_err, center_excess, surface_err, nearest_edge
    )
    return pd.DataFrame(rows), diag


def root_id_from_filename(path):
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)

    matches = re.findall(r"(?<!\d)(\d{15,21})(?!\d)", stem)
    if len(matches) == 1:
        return int(matches[0])
    if not matches:
        raise ValueError(f"No FlyWire root ID found in filename: {Path(path).name}")
    raise ValueError(f"Multiple possible root IDs found in filename: {Path(path).name}")


def compare_swc_to_online_mesh(
    swc_path,
    root_id=None,
    xyz_unit="nm",
    radius_unit=None,
    dataset="flat_783",
    lod=2,
    mesh_threads=5,
    surface_samples=80000,
    spacing_nm=250,
    max_samples_per_edge=128,
    thresholds_nm=(250, 500, 1000),
    exact_center_distance=True,
    random_seed=42,
    diagnostics_path=None,
):
    swc_path = Path(swc_path)
    root_id = root_id_from_filename(swc_path) if root_id is None else int(root_id)
    swc = read_swc(swc_path, xyz_unit, radius_unit)

    mesh, mesh_neuron = fetch_mesh(root_id, dataset, lod, mesh_threads)
    surface_points = sample_surface(mesh, surface_samples, random_seed)

    summary, diagnostics = compare_one(
        swc_path.name, swc, mesh, surface_points,
        spacing_nm, max_samples_per_edge, thresholds_nm,
        exact_center_distance
    )

    mesh_info = {
        "root_id": str(root_id),
        "swc_file": str(swc_path),
        "dataset": dataset,
        "lod_requested": int(lod),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "surface_samples": int(len(surface_points)),
        "mesh_units_reported": str(getattr(mesh_neuron, "units", "unknown")),
    }

    for key, value in reversed(list(mesh_info.items())):
        summary.insert(0, key, value)

    diagnostics.insert(0, "root_id", str(root_id))
    diagnostics.insert(1, "swc_file", str(swc_path))

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(diagnostics_path, index=False)

    del surface_points, mesh, mesh_neuron
    gc.collect()

    return {
        "mesh_info": mesh_info,
        "summary": summary,
        "edge_diagnostics": diagnostics,
    }


def atomic_to_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def append_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def checkpoint_name(swc_path, root_id, base_dir):
    import hashlib

    try:
        rel = Path(swc_path).resolve().relative_to(Path(base_dir).resolve())
    except ValueError:
        rel = Path(swc_path).resolve()

    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"{int(root_id)}_{digest}.csv"


def load_checkpoints(checkpoint_dir):
    files = sorted(Path(checkpoint_dir).glob("*.csv"))
    tables = []
    for path in files:
        try:
            table = pd.read_csv(path, dtype={"root_id": str, "swc_file": str})
            if not table.empty:
                tables.append(table)
        except Exception as exc:
            warnings.warn(f"Ignoring unreadable checkpoint {path.name}: {exc}")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def compare_swc_folder_to_online_mesh(
    swc_dir,
    output_dir="mesh_validation",
    recursive=False,
    xyz_unit="nm",
    radius_unit=None,
    dataset="flat_783",
    lod=2,
    mesh_threads=5,
    surface_samples=20000,
    spacing_nm=500,
    max_samples_per_edge=64,
    thresholds_nm=(250, 500, 1000),
    exact_center_distance=False,
    random_seed=42,
    resume=True,
    limit=None,
    save_edge_diagnostics=False,
    retry_errors=True,
):
    swc_dir = Path(swc_dir).resolve()
    output_dir = Path(output_dir).resolve()
    pattern = "**/*.swc" if recursive else "*.swc"
    swc_paths = sorted(swc_dir.glob(pattern))

    if not swc_paths:
        raise FileNotFoundError(f"No SWC files found in: {swc_dir}")

    if limit is not None:
        swc_paths = swc_paths[:int(limit)]

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    diagnostics_dir = output_dir / "edge_diagnostics"
    summary_path = output_dir / "mesh_match_summary.csv"
    error_path = output_dir / "mesh_match_errors.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_table = load_checkpoints(checkpoint_dir) if resume else pd.DataFrame()
    completed = set()
    if not checkpoint_table.empty and "swc_file" in checkpoint_table:
        completed = set(Path(x).resolve().as_posix() for x in checkpoint_table["swc_file"])

    failed = set()
    if resume and not retry_errors and error_path.exists():
        errors = pd.read_csv(error_path, dtype={"swc_file": str})
        if "swc_file" in errors:
            failed = set(Path(x).resolve().as_posix() for x in errors["swc_file"])

    pending = [
        p for p in swc_paths
        if p.resolve().as_posix() not in completed
        and p.resolve().as_posix() not in failed
    ]

    print(f"Total SWCs: {len(swc_paths):,}")
    print(f"Already completed: {len(completed):,}")
    print(f"Pending: {len(pending):,}")

    for swc_path in tqdm(pending, desc="Comparing SWCs with FlyWire meshes"):
        try:
            root_id = root_id_from_filename(swc_path)
            ckpt_path = checkpoint_dir / checkpoint_name(swc_path, root_id, swc_dir)
            diag_path = (
                diagnostics_dir / f"{root_id}_edge_diagnostics.csv"
                if save_edge_diagnostics else None
            )

            result = compare_swc_to_online_mesh(
                swc_path=swc_path,
                root_id=root_id,
                xyz_unit=xyz_unit,
                radius_unit=radius_unit,
                dataset=dataset,
                lod=lod,
                mesh_threads=mesh_threads,
                surface_samples=surface_samples,
                spacing_nm=spacing_nm,
                max_samples_per_edge=max_samples_per_edge,
                thresholds_nm=thresholds_nm,
                exact_center_distance=exact_center_distance,
                random_seed=random_seed,
                diagnostics_path=diag_path,
            )

            result["summary"]["swc_file"] = str(swc_path.resolve())
            atomic_to_csv(result["summary"], ckpt_path)

            # Rebuild the consolidated table from intact checkpoints.
            atomic_to_csv(load_checkpoints(checkpoint_dir), summary_path)

        except KeyboardInterrupt:
            print("\nInterrupted safely. Restart with resume=True to continue.")
            raise

        except Exception as exc:
            append_csv(pd.DataFrame([{
                "swc_file": str(swc_path.resolve()),
                "root_id": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "time": pd.Timestamp.now().isoformat(),
            }]), error_path)

        finally:
            gc.collect()

    summary = load_checkpoints(checkpoint_dir)
    if not summary.empty:
        atomic_to_csv(summary, summary_path)
        print(f"Completed SWCs: {summary['swc_file'].nunique():,}")
        print(f"Summary: {summary_path}")
    else:
        print("No SWCs completed successfully.")

    if error_path.exists():
        errors = pd.read_csv(error_path)
        print(f"Error records: {len(errors):,} -> {error_path}")

    return summary


# Backward-compatible alias for a single SWC.
def compare_swc_pair_to_online_flywire_mesh(
    root_id,
    repaired_swc_path,
    original_swc_path=None,
    **kwargs,
):
    if original_swc_path is not None:
        warnings.warn(
            "Pairwise comparison is deprecated. Only repaired_swc_path will be processed.",
            DeprecationWarning,
        )
    return compare_swc_to_online_mesh(
        swc_path=repaired_swc_path,
        root_id=root_id,
        **kwargs,
    )
