"""Build the compact FC npy caches under `Conn2Conn_data/fc/` from the xcp-d relmat TSVs.

One cache folder per (condition, parcellation), from the xcp-d combined-run relmat (LR+RL concatenated
before correlation; for rest all four runs):

    fc/parc-{P}_hemi-both/              rest    <- sub-X_task-rest_space-fsLR_seg-{P}_stat-pearsoncorrelation_relmat.tsv
    fc/parc-{P}_hemi-both_task-{T}/     task T  <- sub-X_task-{T}_space-fsLR_seg-{P}_stat-pearsoncorrelation_relmat.tsv

Each folder: subject_ids.npy (N,) int64 sorted; matrices.npy (N,P,P) float32; upper_triangles.npy (N,P(P-1)/2)
float32 (triu k=1); manifest.json (provenance + validation + sha256). Parsing mirrors `dataset_utils.load_fc`
exactly (same read_csv call, float64 -> float32), so a rebuilt rest cache is bit-identical to the existing one.

Standalone on purpose (no imports from data/): running jobs import data/dataset_utils.py live.

Usage (inside kraken_env):
    python data/data_caching/build_fc_cache.py build --condition wm --parcellation Glasser --out-root <Conn2Conn_data> --workers 8
    python data/data_caching/build_fc_cache.py compare --a <cache_dir> --b <cache_dir>
    python data/data_caching/build_fc_cache.py spotcheck --cache-dir <cache_dir> --condition wm --parcellation Glasser --n 20
    python data/data_caching/build_fc_cache.py catalog --out-root <Conn2Conn_data>
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

XCPD_DIR = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_fMRI/xcpd-0-9-1"
DEFAULT_OUT_ROOT = "/scratch/asr655/neuroinformatics/Conn2Conn_data"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATLAS_INFO_DIR = os.path.join(REPO_DIR, "data", "atlas_info")
EXCLUSIONS_TSV = os.path.join(REPO_DIR, "data", "data_caching", "fc_cache_exclusions.tsv")
CONDITIONS = ("rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm")
PARCELLATIONS = ("4S456Parcels", "Glasser")
HEMI = "both"
CACHE_FILES = ("subject_ids.npy", "matrices.npy", "upper_triangles.npy")
ACL_XATTR = "system.nfs4_acl"


def _get_acl(path):
    try:
        return os.getxattr(path, ACL_XATTR)
    except OSError:  # local disk / no NFSv4 ACL support
        return None


def acl_refs(fc_root):
    """(dir ACL, file ACL) to give new cache folders/files, copied from fc/ and an existing cache file.

    On the /scratch VAST mount a new entry inherits only the inheritable ACEs of its parent (not OWNER@),
    which leaves it inaccessible to its owner; so every created folder/file gets these ACLs explicitly.
    """
    dir_acl = _get_acl(fc_root)
    file_acl = None
    for d in sorted(os.listdir(fc_root)) if os.path.isdir(fc_root) else []:
        f = os.path.join(fc_root, d, "matrices.npy")
        if not d.endswith(".partial") and os.path.isfile(f) and os.lstat(f).st_mode & 0o400:
            file_acl = _get_acl(f)
            break
    return dir_acl, file_acl


def _set_acl(path, acl):
    if acl is not None:
        os.setxattr(path, ACL_XATTR, acl)


def relmat_name(subj_folder, condition, parcellation):
    return (f"{subj_folder}_task-{condition}_space-fsLR_seg-{parcellation}"
            f"_stat-pearsoncorrelation_relmat.tsv")


def relmat_path(xcpd_dir, subj_folder, condition, parcellation):
    return os.path.join(xcpd_dir, subj_folder, "func", relmat_name(subj_folder, condition, parcellation))


def cache_dir_name(condition, parcellation):
    suffix = "" if condition == "rest" else f"_task-{condition}"
    return f"parc-{parcellation}_hemi-{HEMI}{suffix}"


def subject_folders(xcpd_dir):
    return sorted(d for d in os.listdir(xcpd_dir) if d.startswith("sub-"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_relmat(path):
    """Same parse as dataset_utils._load_single_fc_file; returns (labels, float32 matrix)."""
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    labels = [str(c) for c in df.columns]
    if [str(i) for i in df.index] != labels:
        raise ValueError(f"row/column labels differ in {path}")
    return labels, df.values.astype(float).astype(np.float32)


def _load_one(args):
    path, subj_folder = args
    if not os.path.exists(path):
        return subj_folder, None, None, None
    try:
        labels, mat = _read_relmat(path)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] != len(labels):
            raise ValueError(f"matrix shape {mat.shape} does not match {len(labels)} labels")
        if not np.isfinite(mat).all():
            raise ValueError(f"{int((~np.isfinite(mat)).sum())} non-finite values")
        return subj_folder, labels, mat, None
    except Exception as e:  # never silently dropped: must be listed in the exclusions TSV
        return subj_folder, None, None, f"{type(e).__name__}: {e}"


def load_exclusions(path, condition, parcellation):
    """{sub-folder: reason} for this condition/parcellation from the tracked exclusions TSV."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    df = df[(df["condition"] == condition) & (df["parcellation"] == parcellation)]
    return {f"sub-{s}": r for s, r in zip(df["subject"], df["reason"])}


def validate(matrices, subject_ids):
    """Structural checks on a stacked cache; returns a dict of results, raises on failure."""
    checks = {
        "n_subjects": int(matrices.shape[0]),
        "n_nodes": int(matrices.shape[1]),
        "subject_ids_sorted_unique": bool(np.all(np.diff(subject_ids) > 0)),
        "all_finite": bool(np.isfinite(matrices).all()),
        "symmetric_max_abs_diff": float(np.abs(matrices - matrices.transpose(0, 2, 1)).max()),
        "diag_min": float(np.einsum("nii->ni", matrices).min()),
        "diag_max": float(np.einsum("nii->ni", matrices).max()),
        "value_min": float(matrices.min()),
        "value_max": float(matrices.max()),
    }
    problems = []
    if not checks["subject_ids_sorted_unique"]:
        problems.append("subject ids not sorted/unique")
    if not checks["all_finite"]:
        problems.append("non-finite values")
    if checks["symmetric_max_abs_diff"] > 1e-6:
        problems.append("not symmetric")
    if checks["value_min"] < -1 - 1e-6 or checks["value_max"] > 1 + 1e-6:
        problems.append("values outside [-1, 1]")
    if problems:
        raise ValueError("validation failed: " + "; ".join(problems))
    return checks


def atlas_label_match(parcellation, labels):
    """Which atlas_info CSV column (if any) lists exactly these labels in this order."""
    path = os.path.join(ATLAS_INFO_DIR, f"{parcellation}_dseg_reformatted.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in df.columns:
        if [str(x) for x in df[col].tolist()] == labels:
            return col
    return None


def cmd_build(a):
    t0 = datetime.datetime.now()
    name = cache_dir_name(a.condition, a.parcellation)
    final_dir = os.path.join(a.out_root, "fc", name)
    if os.path.exists(final_dir):
        sys.exit(f"ABORT: {final_dir} already exists; refusing to overwrite (build elsewhere and compare)")
    tmp_dir = final_dir + ".partial"
    fc_root = os.path.join(a.out_root, "fc")
    os.makedirs(fc_root, exist_ok=True)
    dir_acl, file_acl = acl_refs(fc_root)
    if os.path.exists(tmp_dir):  # leftover from a crashed build of this same folder only
        _set_acl(tmp_dir, dir_acl)
        shutil.rmtree(tmp_dir)

    folders = subject_folders(a.xcpd_dir)
    if a.subjects:
        folders = [f for f in folders if f in set(a.subjects)]
    args = [(relmat_path(a.xcpd_dir, f, a.condition, a.parcellation), f) for f in folders]
    print(f"[build] {name}: {len(folders)} subject folders, {a.workers} workers", flush=True)

    exclusions = load_exclusions(a.exclusions, a.condition, a.parcellation)
    ids, mats, missing, errors, excluded, label_ref = [], [], [], {}, {}, None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for subj_folder, labels, mat, err in ex.map(_load_one, args, chunksize=8):
            if subj_folder in exclusions:  # listed bad source file: leave out, record why
                excluded[subj_folder] = {"reason": exclusions[subj_folder], "load_result": err or "loaded OK"}
                continue
            if err is not None:
                errors[subj_folder] = err
                continue
            if mat is None:
                missing.append(subj_folder)
                continue
            if label_ref is None:
                label_ref = labels
            elif labels != label_ref:
                errors[subj_folder] = "node labels differ from first subject"
                continue
            ids.append(int(subj_folder.replace("sub-", "")))
            mats.append(mat)
    if errors:
        for k, v in errors.items():
            print(f"  ERROR {k}: {v}")
        sys.exit(f"ABORT: {len(errors)} subjects failed to load and are not in {a.exclusions}; nothing written")
    if not mats:
        sys.exit("ABORT: no subjects loaded")

    subject_ids = np.asarray(ids, dtype=np.int64)
    matrices = np.stack(mats, axis=0)
    del mats
    tri = np.triu_indices(matrices.shape[1], k=1)
    upper = matrices[:, tri[0], tri[1]]
    checks = validate(matrices, subject_ids)

    os.makedirs(tmp_dir)
    _set_acl(tmp_dir, dir_acl)
    for fname, arr in (("subject_ids.npy", subject_ids), ("matrices.npy", matrices), ("upper_triangles.npy", upper)):
        np.save(os.path.join(tmp_dir, fname), arr)
        _set_acl(os.path.join(tmp_dir, fname), file_acl)

    manifest = {
        "cache_dir": name,
        "condition": a.condition,
        "parcellation": a.parcellation,
        "hemi": HEMI,
        "source": {
            "xcpd_dir": a.xcpd_dir,
            "pattern": "sub-{id}/func/" + relmat_name("sub-{id}", a.condition, a.parcellation),
            "description": ("xcp-d 0.9.1 combined-run Pearson relmat (runs concatenated before correlation; "
                            "rest = 4 runs, tasks = LR+RL)"),
        },
        "n_subject_folders": len(folders),
        "n_subjects": int(len(subject_ids)),
        "subjects_missing_file": [int(f.replace("sub-", "")) for f in missing],
        "subjects_excluded": {f.replace("sub-", ""): v for f, v in excluded.items()},
        "exclusions_file": os.path.relpath(a.exclusions, REPO_DIR) if a.exclusions else None,
        "node_labels": label_ref,
        "node_labels_match_atlas_info_column": atlas_label_match(a.parcellation, label_ref),
        "arrays": {
            "subject_ids": {"shape": list(subject_ids.shape), "dtype": str(subject_ids.dtype)},
            "matrices": {"shape": list(matrices.shape), "dtype": str(matrices.dtype)},
            "upper_triangles": {"shape": list(upper.shape), "dtype": str(upper.dtype), "triu_k": 1},
        },
        "validation": checks,
        "sha256": {f: sha256_file(os.path.join(tmp_dir, f)) for f in CACHE_FILES},
        "build": {
            "builder": os.path.relpath(os.path.abspath(__file__), REPO_DIR),
            "builder_sha256": sha256_file(os.path.abspath(__file__)),
            "git_commit": a.git_commit,
            "command": " ".join(sys.argv),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "started": t0.isoformat(timespec="seconds"),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(os.path.join(tmp_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    _set_acl(os.path.join(tmp_dir, "manifest.json"), file_acl)
    os.rename(tmp_dir, final_dir)  # folder only appears once complete
    print(f"[build] wrote {final_dir}: N={len(subject_ids)} missing={len(missing)} excluded={len(excluded)} "
          f"shape={matrices.shape} ({(datetime.datetime.now() - t0).total_seconds():.0f}s)")


def cmd_compare(a):
    """Exact (bitwise) comparison of two cache folders' npy arrays."""
    ok = True
    for f in CACHE_FILES:
        pa, pb = os.path.join(a.a, f), os.path.join(a.b, f)
        if not os.path.exists(pa) or not os.path.exists(pb):
            print(f"[compare] {f}: missing in {'a' if not os.path.exists(pa) else 'b'}")
            ok = False
            continue
        xa, xb = np.load(pa, mmap_mode="r"), np.load(pb, mmap_mode="r")
        same = xa.shape == xb.shape and xa.dtype == xb.dtype and np.array_equal(xa, xb)
        detail = "" if same else (f" shape {xa.shape}/{xb.shape} dtype {xa.dtype}/{xb.dtype}"
                                  + (f" max|diff| {float(np.abs(np.asarray(xa, np.float64) - xb).max()):.3g}"
                                     if xa.shape == xb.shape else ""))
        print(f"[compare] {f}: {'IDENTICAL' if same else 'DIFFERENT' + detail}")
        ok &= same
    print("COMPARE PASSED" if ok else "COMPARE FAILED")
    sys.exit(0 if ok else 1)


def cmd_spotcheck(a):
    """Re-read random subjects' source TSVs and check they equal the cached rows exactly."""
    ids = np.load(os.path.join(a.cache_dir, "subject_ids.npy"))
    mats = np.load(os.path.join(a.cache_dir, "matrices.npy"), mmap_mode="r")
    upper = np.load(os.path.join(a.cache_dir, "upper_triangles.npy"), mmap_mode="r")
    tri = np.triu_indices(mats.shape[1], k=1)
    rng = random.Random(a.seed)
    picks = sorted(rng.sample(range(len(ids)), min(a.n, len(ids))))
    bad = 0
    for i in picks:
        _, m = _read_relmat(relmat_path(a.xcpd_dir, f"sub-{ids[i]}", a.condition, a.parcellation))
        if not (np.array_equal(m, mats[i]) and np.array_equal(m[tri], upper[i])):
            print(f"[spotcheck] MISMATCH sub-{ids[i]}")
            bad += 1
    print(f"[spotcheck] {a.cache_dir}: {len(picks) - bad}/{len(picks)} subjects match source TSVs exactly")
    sys.exit(1 if bad else 0)


def cmd_catalog(a):
    """Write fc/catalog.tsv (one row per cache folder) and fc/availability.tsv (subject x condition)."""
    fc_root = os.path.join(a.out_root, "fc")
    rows = []
    for cond in CONDITIONS:
        for parc in PARCELLATIONS:
            d = os.path.join(fc_root, cache_dir_name(cond, parc))
            mpath = os.path.join(d, "manifest.json")
            if not os.path.exists(mpath):
                rows.append(dict(cache_dir=cache_dir_name(cond, parc), condition=cond, parcellation=parc,
                                 n_subjects="", n_nodes="", git_commit="", built="",
                                 sha256_matrices="", note="no manifest (pre-builder cache)"
                                 if os.path.isdir(d) else "missing"))
                continue
            with open(mpath) as fh:
                m = json.load(fh)
            rows.append(dict(cache_dir=m["cache_dir"], condition=cond, parcellation=parc,
                             n_subjects=m["n_subjects"], n_nodes=m["validation"]["n_nodes"],
                             git_commit=m["build"]["git_commit"], built=m["build"]["finished"],
                             sha256_matrices=m["sha256"]["matrices.npy"], note=""))
    _, file_acl = acl_refs(fc_root)
    pd.DataFrame(rows).to_csv(os.path.join(fc_root, "catalog.tsv"), sep="\t", index=False)
    _set_acl(os.path.join(fc_root, "catalog.tsv"), file_acl)

    folders = subject_folders(a.xcpd_dir)

    def present(f):  # one listdir per subject instead of one stat per file
        func = os.path.join(a.xcpd_dir, f, "func")
        names = set(os.listdir(func)) if os.path.isdir(func) else set()
        return [int(relmat_name(f, c, p) in names) for c in CONDITIONS for p in PARCELLATIONS]

    with ThreadPoolExecutor(max_workers=16) as tp:
        flags = list(tp.map(present, folders))
    cols = [f"{c}_{p}" for c in CONDITIONS for p in PARCELLATIONS]
    avail = {"subject": [int(f.replace("sub-", "")) for f in folders]}
    avail.update({col: [row[j] for row in flags] for j, col in enumerate(cols)})
    pd.DataFrame(avail).to_csv(os.path.join(fc_root, "availability.tsv"), sep="\t", index=False)
    _set_acl(os.path.join(fc_root, "availability.tsv"), file_acl)
    print(f"[catalog] wrote {fc_root}/catalog.tsv ({len(rows)} rows) and availability.tsv ({len(folders)} subjects)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("build")
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--parcellation", required=True, choices=PARCELLATIONS)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--xcpd-dir", default=XCPD_DIR)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--subjects", nargs="*", help="restrict to these sub-* folders (testing)")
    p.add_argument("--git-commit", default=None)
    p.add_argument("--exclusions", default=EXCLUSIONS_TSV, help="TSV of known-bad source files to leave out")
    p.set_defaults(func=cmd_build)

    p = sp.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(func=cmd_compare)

    p = sp.add_parser("spotcheck")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--parcellation", required=True, choices=PARCELLATIONS)
    p.add_argument("--xcpd-dir", default=XCPD_DIR)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_spotcheck)

    p = sp.add_parser("catalog")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--xcpd-dir", default=XCPD_DIR)
    p.set_defaults(func=cmd_catalog)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
