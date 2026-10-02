"""Validation for the fc-conditions branch (task-FC loaders, HCP_Base fc_conditions, shared subject ordering,
condition viz). Run from the branch checkout; OLD_TREE must hold `git archive <main commit> data models`.

    OLD_TREE=/path/to/old python scripts/sbatch/checks/verify_fc_conditions.py [Glasser|4S456Parcels]

Checks:
  1. Default HCP_Base (no conditions) is identical to main's: subjects, partitions, FC/SC arrays.
  2. HCP_Base(fc_conditions=...) rows equal the raw cache rows for every subject; subject set = intersection.
  3. HCP_Base.subject_order == main's Evaluator._compute_subject_order logic for every ordering.
  4. pairwise_affine_invariant_distance_within == pairwise_affine_invariant_distance (upper triangle).
  5. Condition viz: invariants (pearson diag 1, distances diag 0, rest vs rest = identical) and brute-force
     within/between-subject means; figures saved to results/figures/fc_conditions_checks/.
Writes results/logs/verify_fc_conditions_<parc>.json and exits nonzero on any failure.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

PARC = sys.argv[1] if len(sys.argv) > 1 else "Glasser"
OLD_TREE = os.environ["OLD_TREE"]
TASKS = ["emotion", "gambling", "language", "motor", "relational", "social", "wm"]
CACHE_ROOT = "/scratch/asr655/neuroinformatics/Conn2Conn_data"
FIG_DIR = REPO_ROOT / "results/figures/fc_conditions_checks"
results, failures = {}, []


def check(name, ok, detail=""):
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


FINGERPRINT = r"""
import sys, json, hashlib, numpy as np
sys.path.insert(0, sys.argv[1])
from data.hcp_dataset import HCP_Base
b = HCP_Base(parcellation=sys.argv[2], source='SC', target='FC', data_load_mode='precomputed', shuffle_seed=0)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
idx = np.asarray(b.trainvaltest_partition_indices['val'])
def old_order(order_by):
    if order_by == 'family':
        md = b.metadata_df
        return np.argsort(md.loc[md.index[idx], 'Family_ID'].values).tolist()
    if order_by == 'demographic':
        c = np.concatenate([np.asarray(b.sex_oh)[idx], np.asarray(b.race_eth_oh)[idx]], axis=1)
        return np.argsort(np.unique(c, axis=0, return_inverse=True)[1]).tolist()
    return np.argsort(np.asarray(b.age_z)[idx].ravel()).tolist()
new_order = (lambda o: b.subject_order(idx, o).tolist()) if hasattr(b, 'subject_order') else None
print(json.dumps({
    'subjects': b.metadata_df['subject'].astype(int).tolist(),
    'partitions': {k: list(map(int, v)) for k, v in b.trainvaltest_partition_indices.items()},
    'fc_tri': h(b.fc_upper_triangles), 'fc_mat': h(b.fc_matrices), 'sc_tri': h(b.sc_upper_triangles),
    'old_order': {o: old_order(o) for o in ('family', 'demographic', 'age')},
    'new_order': None if new_order is None else {o: new_order(o) for o in ('family', 'demographic', 'age')},
}))
"""


def fingerprint(tree):
    out = subprocess.run([sys.executable, "-c", FINGERPRINT, tree, PARC], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# 1 + 3 ----------------------------------------------------------------------------------------------------
t0 = time.time()
old_fp, new_fp = fingerprint(OLD_TREE), fingerprint(str(REPO_ROOT))
for key in ("subjects", "partitions", "fc_tri", "fc_mat", "sc_tri"):
    check(f"1.default_base_identical.{key}", old_fp[key] == new_fp[key])
for order_by in ("family", "demographic", "age"):
    check(f"3.subject_order.{order_by}", new_fp["new_order"][order_by] == old_fp["old_order"][order_by])
print(f"  fingerprints {time.time() - t0:.0f}s", flush=True)

# 2 -------------------------------------------------------------------------------------------------------
from data.hcp_dataset import HCP_Base
from data import data_viz as dv

t0 = time.time()
base = HCP_Base(parcellation=PARC, source="SC", target="FC", data_load_mode="precomputed", shuffle_seed=0,
                expose_fc_sessions=True, fc_conditions=["rest", *TASKS])
canon = base.metadata_df["subject"].astype(int).tolist()
expected = set(old_fp["subjects"])
for t in TASKS:
    expected &= set(np.load(f"{CACHE_ROOT}/fc/parc-{PARC}_hemi-both_task-{t}/subject_ids.npy").tolist())
for s in ("session1", "session2"):
    expected &= set(np.load(f"{CACHE_ROOT}/fc/parc-{PARC}_hemi-both_{s}/subject_ids.npy").tolist())
check("2.canonical_is_intersection", canon == sorted(expected), f"n={len(canon)}")
for t in TASKS:
    d = f"{CACHE_ROOT}/fc/parc-{PARC}_hemi-both_task-{t}"
    ids = np.load(f"{d}/subject_ids.npy")
    tri = np.load(f"{d}/upper_triangles.npy", mmap_mode="r")
    pos = np.searchsorted(ids, np.asarray(canon))
    ok = np.array_equal(ids[pos], canon) and np.array_equal(np.asarray(tri[pos]), base.fc_condition_upper_triangles[t])
    check(f"2.condition_rows_match_cache.{t}", ok)
md = base.metadata_df
check("2.partitions_point_at_their_subjects", all(
    (md["train_val_test"].values[np.asarray(base.trainvaltest_partition_indices[p])] == p).all()
    and sorted(base.trainvaltest_partition_indices[p]) == list(np.where(md["train_val_test"].values == p)[0])
    for p in ("train", "val", "test")), f"sizes={[len(base.trainvaltest_partition_indices[p]) for p in ('train', 'val', 'test')]}")
check("2.rest_alias_is_main_fc", base.fc_condition_upper_triangles["rest"] is base.fc_upper_triangles)
check("2.matrices_not_loaded_by_default", all(base.fc_condition_matrices_by_condition[t] is None for t in TASKS))
print(f"  base with conditions {time.time() - t0:.0f}s, n={len(canon)}", flush=True)

# 4 -------------------------------------------------------------------------------------------------------
from models.eval.fc_distance import pairwise_affine_invariant_distance, pairwise_affine_invariant_distance_within

rng = np.random.default_rng(0)
mats = np.stack([dv._shrink(dv._edges_to_square(base.fc_condition_upper_triangles[t][0], 1.0), 0.1) for t in TASKS[:4]])
ref, fast = pairwise_affine_invariant_distance(mats, mats), pairwise_affine_invariant_distance_within(mats)
iu = np.triu_indices(4, 1)
check("4.geodesic_within_matches_upper", np.allclose(ref[iu], fast[iu], rtol=0, atol=1e-10),
      f"max|diff|={np.abs(ref[iu] - fast[iu]).max():.2e}")
check("4.geodesic_symmetric_close", np.allclose(ref, fast, rtol=1e-6, atol=1e-6))

# 5 -------------------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR.mkdir(parents=True, exist_ok=True)
conds = ["rest", "rest_S1", "rest_S2", *TASKS]
timings = {}

t0 = time.time()
fig, _, d1 = dv.plot_condition_connectomes(base, conds, partition="val", ncols=5, show=False)
fig.savefig(FIG_DIR / f"fig1_population_mean_{PARC}.png", bbox_inches="tight"); plt.close(fig)
fig, _, d1s = dv.plot_condition_connectomes(base, conds, partition="val", position=0, demean_partition="train",
                                           ncols=5, header_metadata="age_sex", show=False)
fig.savefig(FIG_DIR / f"fig1_subject0_demeaned_{PARC}.png", bbox_inches="tight"); plt.close(fig)
timings["fig1"] = time.time() - t0
val = np.asarray(base.trainvaltest_partition_indices["val"])
manual = dv._edges_to_square(base.fc_condition_upper_triangles["wm"][val].mean(axis=0), 1.0)
check("5.fig1_mean_matches_manual", np.allclose(d1["matrices"]["wm"], manual))
check("5.fig1_diag", np.allclose(np.diag(d1["matrices"]["rest"]), 1) and np.allclose(np.diag(d1s["matrices"]["rest"]), 0))

t0 = time.time()
pop = dv.compute_condition_similarity(base, conds + ["rest"], partition="val", mode="population")
check("5.fig2_pearson_diag_1", np.allclose(np.diag(pop["metrics"]["pearson"]["mean"]), 1))
check("5.fig2_distance_diag_0", all(np.allclose(np.diag(pop["metrics"][m]["mean"]), 0) for m in ("euclidean", "geodesic")))
k = len(conds)
check("5.fig2_rest_vs_rest_identical", np.isclose(pop["metrics"]["pearson"]["mean"][0, k], 1)
      and np.isclose(pop["metrics"]["euclidean"]["mean"][0, k], 0, atol=1e-8)
      and np.isclose(pop["metrics"]["geodesic"]["mean"][0, k], 0, atol=1e-6))
pop = dv.compute_condition_similarity(base, conds, partition="val", mode="population")
fig, _ = dv.plot_condition_similarity(base, pop, show=False)
fig.savefig(FIG_DIR / f"fig2_population_{PARC}.png", bbox_inches="tight"); plt.close(fig)
timings["fig2_population"] = time.time() - t0
t0 = time.time()
one = dv.compute_condition_similarity(base, conds, partition="val", mode="subject", position=0)
check("5.fig2_subject_shrinkage_default", one["shrinkage"] == 0.1)
timings["fig2_subject"] = time.time() - t0
t0 = time.time()
subs = dv.compute_condition_similarity(base, conds, partition="val", mode="subjects", max_subjects=10)
check("5.fig2_subjects_mean_of_per_subject",
      np.allclose(subs["metrics"]["pearson"]["mean"], subs["metrics"]["pearson"]["per_subject"].mean(axis=0)))
fig, _ = dv.plot_condition_similarity(base, subs, show_std=True, show=False)
fig.savefig(FIG_DIR / f"fig2_subjects10_{PARC}.png", bbox_inches="tight"); plt.close(fig)
timings["fig2_subjects_per_subject"] = (time.time() - t0) / 10

t0 = time.time()
small = dv.compute_subject_condition_correlations(base, TASKS[:3], partition="val", max_subjects=4)
e = dv.condition_upper_triangles(base, TASKS[:3])
rows = {(s, c): e[c][val[s]] for s in range(4) for c in range(3)}
r = lambda a, b: np.corrcoef(rows[a], rows[b])[0, 1]
w = np.array([[np.mean([r((s, a), (s, b)) for s in range(4)]) for b in range(3)] for a in range(3)])
btw = np.array([[np.mean([r((s, a), (t, b)) for s in range(4) for t in range(4) if s != t]) for b in range(3)] for a in range(3)])
check("5.fig3_within_bruteforce", np.allclose(small["within"], w, atol=1e-6))
check("5.fig3_between_bruteforce", np.allclose(small["between"], btw, atol=1e-6))
check("5.fig3_block_layout", np.isclose(small["corr"][0, 1], r((0, 0), (0, 1)), atol=1e-6))
smallc = dv.compute_subject_condition_correlations(base, TASKS[:3], partition="val", max_subjects=4, block_by="condition")
check("5.fig3_condition_layout", np.isclose(smallc["corr"][0, 1], r((0, 0), (1, 0)), atol=1e-6))
for order_by in ("original", "family"):
    fig, pfig, d3 = dv.plot_subject_condition_corrmap(base, conds, partition="val", order_by=order_by, show=False)
    fig.savefig(FIG_DIR / f"fig3_val_{order_by}_{PARC}.png", bbox_inches="tight"); plt.close(fig)
    pfig.savefig(FIG_DIR / f"fig3_val_pairs_{order_by}_{PARC}.png", bbox_inches="tight"); plt.close(pfig)
timings["fig3_val"] = (time.time() - t0) / 2
results["fig3_val_summary"] = d3["summary"].round(4).to_dict()
print(d3["summary"].round(4), flush=True)
results["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
results["n_subjects"] = {"canonical": len(canon), "val": int(len(val))}
print("timings (s):", results["timings_s"], flush=True)

out = REPO_ROOT / f"results/logs/verify_fc_conditions_{PARC}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"failures": failures, "results": results}, indent=2, default=str))
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'} -> {out}")
sys.exit(1 if failures else 0)
