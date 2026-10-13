"""Spec v3 E1.1: build the unnormalised Glasser SC cache (sift_radius2_count, log1p) and run the cache gate.

Reads the raw qsirecon .mat files through data.dataset_utils.load_sc, writes
Conn2Conn_data/sc/parc-Glasser_hemi-both_metric-sift_radius2_count_connectivity_log1p-1/ (via a .partial folder,
renamed at the end; refuses to overwrite), copying the NFSv4 ACLs from the sibling invnodevol cache (on the /scratch
VAST mount new entries otherwise come out owner-locked). Then checks the gate against the invnodevol cache and
writes the result to runs/cache_gate.json.

  python scripts/experiments/covariate_ablation/build_sc_cache.py [--workers 8]
"""
import argparse
import json
import os
import shutil
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from data.dataset_utils import DEFAULT_CONN2CONN_CACHE_ROOT, _cache_dir_sc, load_sc  # noqa: E402

ACL_XATTR = "system.nfs4_acl"
PARC, HEMI = "Glasser", "both"
NEW_METRIC = "sift_radius2_count_connectivity"
REF_METRIC = "sift_invnodevol_radius2_count_connectivity"


def _get_acl(path):
    try:
        return os.getxattr(path, ACL_XATTR)
    except OSError:
        return None


def _set_acl(path, acl):
    if acl is not None:
        os.setxattr(path, ACL_XATTR, acl)


def build(cache_root, workers):
    ref_dir = _cache_dir_sc(cache_root, PARC, HEMI, REF_METRIC, True)
    out_dir = _cache_dir_sc(cache_root, PARC, HEMI, NEW_METRIC, True)
    if os.path.exists(out_dir):
        print(f"[build] {out_dir} exists; not overwriting", flush=True)
        return out_dir
    dir_acl, file_acl = _get_acl(ref_dir), _get_acl(os.path.join(ref_dir, "matrices.npy"))
    tmp_dir = out_dir + ".partial"
    if os.path.exists(tmp_dir):
        _set_acl(tmp_dir, dir_acl)
        shutil.rmtree(tmp_dir)
    ids, mats, tris, r2t = load_sc(PARC, HEMI, NEW_METRIC, apply_log1p=True, n_jobs=workers, write_cache=False)
    os.makedirs(tmp_dir)
    _set_acl(tmp_dir, dir_acl)
    arrays = {"subject_ids": np.asarray(ids, dtype=np.int64), "matrices": mats.astype(np.float32),
              "upper_triangles": tris.astype(np.float32),
              "r2t_matrices": None if r2t is None else r2t.astype(np.float32)}
    for name, arr in arrays.items():
        if arr is None:
            continue
        path = os.path.join(tmp_dir, f"{name}.npy")
        np.save(path, arr)
        _set_acl(path, file_acl)
    os.rename(tmp_dir, out_dir)
    print(f"[build] wrote {out_dir}: {len(ids)} subjects, matrices {mats.shape}", flush=True)
    return out_dir


def gate(cache_root):
    ref_dir = _cache_dir_sc(cache_root, PARC, HEMI, REF_METRIC, True)
    new_dir = _cache_dir_sc(cache_root, PARC, HEMI, NEW_METRIC, True)
    ref_ids = np.load(os.path.join(ref_dir, "subject_ids.npy"))
    new_ids = np.load(os.path.join(new_dir, "subject_ids.npy"))
    new_mats = np.load(os.path.join(new_dir, "matrices.npy"), mmap_mode="r")
    new_tri = np.load(os.path.join(new_dir, "upper_triangles.npy"))
    ref_tri = np.load(os.path.join(ref_dir, "upper_triangles.npy"))
    common = np.intersect1d(ref_ids, new_ids)
    ri = {s: i for i, s in enumerate(ref_ids)}
    ni = {s: i for i, s in enumerate(new_ids)}
    rows_r = [ri[s] for s in common]
    rows_n = [ni[s] for s in common]
    r = np.array([np.corrcoef(ref_tri[i], new_tri[j])[0, 1] for i, j in zip(rows_r, rows_n)])
    sym = max(float(np.abs(new_mats[k] - new_mats[k].T).max()) for k in range(0, len(new_ids), 25))
    res = {
        "n_ref": int(len(ref_ids)), "n_new": int(len(new_ids)), "n_common": int(len(common)),
        "missing_from_new": [int(s) for s in np.setdiff1d(ref_ids, new_ids)],
        "finite": bool(np.isfinite(new_tri).all()), "max_asym_sampled": sym,
        "corr_with_invnodevol": {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())},
    }
    res["pass"] = bool(res["n_common"] == res["n_ref"] and res["finite"] and sym < 1e-5
                       and res["corr_with_invnodevol"]["median"] > 0.8 and res["corr_with_invnodevol"]["max"] < 0.9999)
    os.makedirs(os.path.join(EXP_DIR, "runs"), exist_ok=True)
    with open(os.path.join(EXP_DIR, "runs", "cache_gate.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache_root", default=DEFAULT_CONN2CONN_CACHE_ROOT)
    a = p.parse_args()
    build(a.cache_root, a.workers)
    sys.exit(0 if gate(a.cache_root)["pass"] else 1)
