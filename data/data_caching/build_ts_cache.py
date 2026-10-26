"""Build the regional timeseries caches under `Conn2Conn_data/timeseries/` from the xcp-d per-run mean timeseries TSVs.

One cache folder per (condition, parcellation), named as in fc/ (layout and reader: data/timeseries.py):

    timeseries/parc-{P}_hemi-both[_task-{T}]/  timeseries.npy (T_total, P) float32, subject_ids.npy, offsets.npy,
                                                runs.tsv, manifest.json

A subject is included when xcp-d wrote its combined-run timeseries (the fc/ inclusion rule). Its rows are the per-run
TSVs listed in the combined file's JSON sidecar ("Sources"), in that order, i.e. xcp-d's combined timeseries,
rebuilt from the runs so run boundaries are known (and a damaged combined TSV does not matter). Parsing as
build_fc_cache (read_csv, float64 -> float32).

Validation (in build, every subject): the Pearson correlation of the subject's rows must equal the fc/ cache matrix of
the same condition (xcp-d combined relmat) within --corr-tol. `spotcheck`: for random subjects, the rows equal the
parsed combined TSV exactly, and each run's correlation equals the per-run fc/ cache matrix within --corr-tol.

Usage (inside kraken_env):
    python data/data_caching/build_ts_cache.py build --condition wm --parcellation Glasser --workers 8
    python data/data_caching/build_ts_cache.py spotcheck --condition wm --parcellation Glasser --n 25
    python data/data_caching/build_ts_cache.py catalog
"""
import argparse
import datetime
import json
import os
import random
import shutil
import socket
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fc_cache import (CONDITIONS, DEFAULT_OUT_ROOT, PARCELLATIONS, REPO_DIR, RUNS, XCPD_DIR,  # noqa: E402
                            _set_acl, acl_refs, atlas_label_match, cache_dir_name, sha256_file, subject_folders)

TR = 0.72
EXCLUSIONS_TSV = os.path.join(REPO_DIR, "data", "data_caching", "ts_cache_exclusions.tsv")
CACHE_FILES = ("timeseries.npy", "subject_ids.npy", "offsets.npy", "runs.tsv")


def ts_name(subj_folder, condition, parcellation, run=None):
    ent = "" if run is None else f"_dir-{run[:2]}_run-{run[2:]}"
    return f"{subj_folder}_task-{condition}{ent}_space-fsLR_seg-{parcellation}_stat-mean_timeseries"


def run_label(source):
    """'bids::sub-X/func/sub-X_task-rest_dir-LR_run-1_..._timeseries.tsv' -> 'LR1'."""
    name = os.path.basename(source)
    d = name.split("_dir-")[1][:2]
    r = name.split("_run-")[1].split("_")[0]
    return f"{d}{r}"


def _read_ts(path):
    df = pd.read_csv(path, sep="\t", header=0)
    return [str(c) for c in df.columns], df.values.astype(float).astype(np.float32)


def _load_one(args):
    """-> (subj_folder, labels, [(run, array)], error). labels None and no error = no combined file."""
    xcpd_dir, subj_folder, condition, parcellation = args
    base = os.path.join(xcpd_dir, subj_folder, "func", ts_name(subj_folder, condition, parcellation))
    if not os.path.exists(base + ".tsv"):
        return subj_folder, None, None, None
    try:
        with open(base + ".json") as fh:
            sources = json.load(fh)["Sources"]
        runs, labels = [], None
        for src in sources:
            lab, arr = _read_ts(os.path.join(xcpd_dir, src.replace("bids::", "", 1)))
            if labels is None:
                labels = lab
            elif lab != labels:
                raise ValueError(f"node labels differ between runs ({src})")
            if arr.ndim != 2 or arr.shape[1] != len(labels) or arr.shape[0] < 2:
                raise ValueError(f"run {src} has shape {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{int((~np.isfinite(arr)).sum())} non-finite values in {src}")
            runs.append((run_label(src), arr))
        if [r for r, _ in runs] != [r for r in RUNS[condition] if r in {x for x, _ in runs}]:
            raise ValueError(f"unexpected run order {[r for r, _ in runs]}")
        return subj_folder, labels, runs, None
    except Exception as e:  # never silently dropped: must be listed in the exclusions TSV
        return subj_folder, None, None, f"{type(e).__name__}: {e}"


def load_exclusions(path, condition, parcellation):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#", keep_default_na=False)
    df = df[(df["condition"] == condition) & (df["parcellation"] == parcellation) & (df["file"] != "combined")]
    return {f"sub-{s}": f"{f}: {r}" for s, f, r in zip(df["subject"], df["file"], df["reason"])}


def corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    x = x / np.sqrt(np.square(x).sum(0))
    return x.T @ x


def fc_cache(out_root, condition, parcellation, run=None):
    """(subject_ids, matrices memmap) of the matching fc/ cache folder, or None."""
    d = os.path.join(out_root, "fc", cache_dir_name(condition, parcellation, run))
    if not os.path.exists(os.path.join(d, "matrices.npy")):
        return None
    return np.load(os.path.join(d, "subject_ids.npy")), np.load(os.path.join(d, "matrices.npy"), mmap_mode="r")


def cmd_build(a):
    t0 = datetime.datetime.now()
    name = cache_dir_name(a.condition, a.parcellation)
    ts_root = os.path.join(a.out_root, "timeseries")
    final_dir = os.path.join(ts_root, name)
    if os.path.exists(final_dir):
        sys.exit(f"ABORT: {final_dir} already exists; refusing to overwrite")
    tmp_dir = final_dir + ".partial"
    dir_acl, file_acl = acl_refs(os.path.join(a.out_root, "fc"))  # fc/'s ACLs (see build_fc_cache.acl_refs)
    if not os.path.exists(ts_root):
        os.makedirs(ts_root)
        _set_acl(ts_root, dir_acl)
    if os.path.exists(tmp_dir):
        _set_acl(tmp_dir, dir_acl)
        shutil.rmtree(tmp_dir)

    folders = subject_folders(a.xcpd_dir)
    if a.subjects:
        folders = [f for f in folders if f in set(a.subjects)]
    print(f"[build] {name}: {len(folders)} subject folders, {a.workers} workers", flush=True)
    exclusions = load_exclusions(a.exclusions, a.condition, a.parcellation)
    loaded, missing, errors, excluded, label_ref = [], [], {}, {}, None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [(a.xcpd_dir, f, a.condition, a.parcellation) for f in folders]
        for subj_folder, labels, runs, err in ex.map(_load_one, jobs, chunksize=4):
            if subj_folder in exclusions:
                excluded[subj_folder] = {"reason": exclusions[subj_folder], "load_result": err or "loaded OK"}
                continue
            if err is not None:
                errors[subj_folder] = err
                continue
            if runs is None:
                missing.append(subj_folder)
                continue
            if label_ref is None:
                label_ref = labels
            elif labels != label_ref:
                errors[subj_folder] = "node labels differ from first subject"
                continue
            loaded.append((int(subj_folder.replace("sub-", "")), runs))
    if errors:
        for k, v in errors.items():
            print(f"  ERROR {k}: {v}")
        sys.exit(f"ABORT: {len(errors)} subjects failed to load and are not in {a.exclusions}; nothing written")
    if not loaded:
        sys.exit("ABORT: no subjects loaded")
    loaded.sort(key=lambda t: t[0])

    subject_ids = np.asarray([s for s, _ in loaded], dtype=np.int64)
    lengths = np.asarray([sum(len(r) for _, r in runs) for _, runs in loaded], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    P = len(label_ref)

    # validation against the combined-run fc/ cache: corr(rows) == xcp-d combined relmat
    ref = fc_cache(a.fc_root, a.condition, a.parcellation)
    diffs, not_in_fc = {}, []
    if ref is not None:
        ref_row = {int(s): i for i, s in enumerate(ref[0])}
    for sid, runs in loaded:
        if ref is None or sid not in ref_row:
            not_in_fc.append(sid)
            continue
        diffs[sid] = float(np.abs(corr(np.concatenate([r for _, r in runs])) - ref[1][ref_row[sid]]).max())
    bad = {s: d for s, d in diffs.items() if not d <= a.corr_tol}
    if bad:
        for s, d in sorted(bad.items())[:20]:
            print(f"  MISMATCH sub-{s}: max|corr - fc| = {d:.3g}")
        sys.exit(f"ABORT: {len(bad)} subjects' correlation differs from the fc/ cache by > {a.corr_tol}")

    os.makedirs(tmp_dir)
    _set_acl(tmp_dir, dir_acl)
    ts_path = os.path.join(tmp_dir, "timeseries.npy")
    open(ts_path, "wb").close()  # create + fix the ACL first: open_memmap reopens the file it creates
    _set_acl(ts_path, file_acl)
    out = np.lib.format.open_memmap(ts_path, mode="w+", dtype=np.float32, shape=(int(offsets[-1]), P))
    run_rows = []
    for i, (sid, runs) in enumerate(loaded):
        pos = int(offsets[i])
        for run, arr in runs:
            out[pos:pos + len(arr)] = arr
            run_rows.append(dict(subject=sid, run=run, start=pos, n_tr=len(arr)))
            pos += len(arr)
    out.flush()
    del out, loaded
    np.save(os.path.join(tmp_dir, "subject_ids.npy"), subject_ids)
    np.save(os.path.join(tmp_dir, "offsets.npy"), offsets)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(tmp_dir, "runs.tsv"), sep="\t", index=False)
    for f in CACHE_FILES:
        _set_acl(os.path.join(tmp_dir, f), file_acl)

    run_sets = runs_df.groupby("subject")["run"].apply(lambda r: " ".join(r)).value_counts()
    manifest = {
        "cache_dir": name,
        "condition": a.condition,
        "parcellation": a.parcellation,
        "hemi": "both",
        "tr": TR,
        "source": {
            "xcpd_dir": a.xcpd_dir,
            "pattern": "sub-{id}/func/" + ts_name("sub-{id}", a.condition, a.parcellation, "{D}{R}") + ".tsv",
            "run_order": "JSON sidecar 'Sources' of the combined-run timeseries (rest LR1 RL1 LR2 RL2; tasks LR1 RL1)",
            "description": ("xcp-d 0.9.1 denoised, band-pass filtered parcel-mean BOLD (not z-scored, no censored "
                            "volumes); a subject's runs concatenated = xcp-d's combined-run timeseries"),
        },
        "n_subject_folders": len(folders),
        "n_subjects": int(len(subject_ids)),
        "n_tr_total": int(offsets[-1]),
        "n_tr_per_subject": {str(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))},
        "run_sets": {k: int(v) for k, v in run_sets.items()},
        "subjects_missing_file": [int(f.replace("sub-", "")) for f in missing],
        "subjects_excluded": {f.replace("sub-", ""): v for f, v in excluded.items()},
        "exclusions_file": os.path.relpath(a.exclusions, REPO_DIR) if a.exclusions else None,
        "node_labels": label_ref,
        "node_labels_match_atlas_info_column": atlas_label_match(a.parcellation, label_ref),
        "arrays": {
            "timeseries": {"shape": [int(offsets[-1]), P], "dtype": "float32"},
            "subject_ids": {"shape": list(subject_ids.shape), "dtype": "int64"},
            "offsets": {"shape": list(offsets.shape), "dtype": "int64"},
        },
        "validation": {
            "fc_reference": None if ref is None else os.path.join("fc", name),
            "corr_tol": a.corr_tol,
            "n_compared": len(diffs),
            "max_abs_corr_minus_fc": max(diffs.values()) if diffs else None,
            "subjects_not_in_fc_cache": not_in_fc,
        },
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
    os.rename(tmp_dir, final_dir)
    print(f"[build] wrote {final_dir}: N={len(subject_ids)} T_total={int(offsets[-1])} missing={len(missing)} "
          f"excluded={len(excluded)} max|corr-fc|={manifest['validation']['max_abs_corr_minus_fc']} "
          f"({(datetime.datetime.now() - t0).total_seconds():.0f}s)")


def cmd_spotcheck(a):
    d = os.path.join(a.out_root, "timeseries", cache_dir_name(a.condition, a.parcellation))
    ids = np.load(os.path.join(d, "subject_ids.npy"))
    offsets = np.load(os.path.join(d, "offsets.npy"))
    data = np.load(os.path.join(d, "timeseries.npy"), mmap_mode="r")
    runs = pd.read_csv(os.path.join(d, "runs.tsv"), sep="\t")
    known_bad = {(int(s), r) for s, c, p, r in pd.read_csv(a.known_bad, sep="\t", comment="#", dtype=str,
                                                             keep_default_na=False)[["subject", "condition",
                                                                                     "parcellation", "file"]].values
                 if c == a.condition and p == a.parcellation} if os.path.exists(a.known_bad) else set()
    run_fc = {r: fc_cache(a.fc_root, a.condition, a.parcellation, r) for r in RUNS[a.condition]}
    rng = random.Random(a.seed)
    picks = sorted(rng.sample(range(len(ids)), min(a.n, len(ids))))
    bad, n_run_cmp, worst = 0, 0, 0.0
    for i in picks:
        sid = int(ids[i])
        x = np.asarray(data[offsets[i]:offsets[i + 1]])
        if (sid, "combined") not in known_bad:
            _, comb = _read_ts(os.path.join(a.xcpd_dir, f"sub-{sid}", "func",
                                            ts_name(f"sub-{sid}", a.condition, a.parcellation) + ".tsv"))
            if not np.array_equal(comb, x):
                print(f"[spotcheck] MISMATCH sub-{sid}: rows != combined TSV")
                bad += 1
        for _, row in runs[runs.subject == sid].iterrows():
            ref = run_fc[row.run]
            if ref is None or sid not in set(ref[0].tolist()):
                continue
            j = int(np.searchsorted(ref[0], sid))
            seg = np.asarray(data[row.start:row.start + row.n_tr])
            dmax = float(np.abs(corr(seg) - ref[1][j]).max())
            worst, n_run_cmp = max(worst, dmax), n_run_cmp + 1
            if not dmax <= a.corr_tol:
                print(f"[spotcheck] MISMATCH sub-{sid} run {row.run}: max|corr - per-run fc| = {dmax:.3g}")
                bad += 1
    print(f"[spotcheck] {d}: {len(picks)} subjects, {n_run_cmp} run comparisons (max|diff| {worst:.3g}), "
          f"{bad} mismatches")
    sys.exit(1 if bad else 0)


def cmd_catalog(a):
    ts_root = os.path.join(a.out_root, "timeseries")
    rows = []
    for cond in CONDITIONS:
        for parc in PARCELLATIONS:
            name = cache_dir_name(cond, parc)
            mpath = os.path.join(ts_root, name, "manifest.json")
            if not os.path.exists(mpath):
                rows.append(dict(cache_dir=name, condition=cond, parcellation=parc, note="missing"))
                continue
            with open(mpath) as fh:
                m = json.load(fh)
            rows.append(dict(cache_dir=name, condition=cond, parcellation=parc, n_subjects=m["n_subjects"],
                             n_nodes=len(m["node_labels"]), n_tr_total=m["n_tr_total"],
                             gb=round(os.path.getsize(os.path.join(ts_root, name, "timeseries.npy")) / 1e9, 2),
                             max_abs_corr_minus_fc=m["validation"]["max_abs_corr_minus_fc"],
                             git_commit=m["build"]["git_commit"], built=m["build"]["finished"],
                             sha256_timeseries=m["sha256"]["timeseries.npy"], note=""))
    _, file_acl = acl_refs(os.path.join(a.out_root, "fc"))
    pd.DataFrame(rows).to_csv(os.path.join(ts_root, "catalog.tsv"), sep="\t", index=False)
    _set_acl(os.path.join(ts_root, "catalog.tsv"), file_acl)
    print(f"[catalog] wrote {ts_root}/catalog.tsv ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("build")
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--parcellation", required=True, choices=PARCELLATIONS)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--xcpd-dir", default=XCPD_DIR)
    p.add_argument("--fc-root", default=DEFAULT_OUT_ROOT, help="root holding the fc/ caches used for validation")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--subjects", nargs="*", help="restrict to these sub-* folders (testing)")
    p.add_argument("--git-commit", default=None)
    p.add_argument("--exclusions", default=EXCLUSIONS_TSV, help="TSV of subjects whose per-run sources are bad")
    p.add_argument("--corr-tol", type=float, default=1e-6)
    p.set_defaults(func=cmd_build)

    p = sp.add_parser("spotcheck")
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--parcellation", required=True, choices=PARCELLATIONS)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--xcpd-dir", default=XCPD_DIR)
    p.add_argument("--fc-root", default=DEFAULT_OUT_ROOT, help="root holding the per-run fc/ caches")
    p.add_argument("--known-bad", default=EXCLUSIONS_TSV, help="rows with file=combined are skipped in the TSV check")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corr-tol", type=float, default=1e-6)
    p.set_defaults(func=cmd_spotcheck)

    p = sp.add_parser("catalog")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.set_defaults(func=cmd_catalog)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
