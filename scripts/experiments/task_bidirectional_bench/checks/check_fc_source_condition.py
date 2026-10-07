"""Spec v3 E0 check: HCP_Base(fc_source_condition=...) on the all-conditions cohort.

1. rest on the all-conditions cohort == the plain HCP_Base (no conditions) restricted to that cohort, for SC, FC,
   and the per-seed train/val/test labels.
2. Every condition: identical cohort and splits across conditions; FC rebound to that condition's cache rows
   (tasks from fc_condition_upper_triangles, rest_S1 from the session-1 cache); PCA basis fit on that FC.

3. --direction sc2fc: the dataset target y (HCP_Partition) is that condition's FC (the condition is the target).
4. The cohort size matches config.yml cohort[<parcellation>].

    python scripts/experiments/task_bidirectional_bench/checks/check_fc_source_condition.py [--seeds 0 1] \
        [--parcellation Glasser|4S456Parcels] [--direction fc2sc|sc2fc]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.dataset_utils import FC_TASK_CONDITIONS  # noqa: E402
from data.hcp_dataset import HCP_Base, HCP_Partition  # noqa: E402
import yaml  # noqa: E402

CONDITIONS = ("rest", "rest_S1", *FC_TASK_CONDITIONS)
COMMON = dict(parcellation="Glasser", hemi="both", source="FC", target="SC", data_load_mode="precomputed")
CFG = yaml.safe_load((REPO_ROOT / "scripts" / "experiments" / "task_bidirectional_bench" / "config.yml").read_text())


def base(seed, cond=None, all_conditions=True):
    kw = dict(COMMON, shuffle_seed=seed)
    if all_conditions:
        kw.update(fc_conditions=list(FC_TASK_CONDITIONS), expose_fc_sessions=True, fc_source_condition=cond)
    return HCP_Base(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--parcellation", default="Glasser")
    ap.add_argument("--direction", choices=["fc2sc", "sc2fc"], default="fc2sc")
    args = ap.parse_args()
    d = CFG["directions"][args.direction]
    COMMON.update(parcellation=args.parcellation, source=d["source"], target=d["target"])
    ok = True
    for seed in args.seeds:
        plain = base(seed, all_conditions=False)
        pid = [str(s) for s in plain.metadata_df["subject"]]
        ref = None
        for cond in CONDITIONS:
            b = base(seed, cond)
            ids = [str(s) for s in b.metadata_df["subject"]]
            parts = {k: [str(x) for x in v] for k, v in b.trainvaltest_partition_ids.items()}
            if ref is None:
                ref = (ids, parts)
                pos = [pid.index(s) for s in ids]
                plain_parts = {k: [str(x) for x in v if str(x) in set(ids)] for k, v in plain.trainvaltest_partition_ids.items()}
                checks = {
                    "cohort_size": len(ids),
                    "cohort_matches_config": len(ids) == CFG["cohort"][args.parcellation],
                    "sc_equal": np.array_equal(b.sc_upper_triangles, plain.sc_upper_triangles[pos]),
                    "rest_fc_equal": np.array_equal(b.fc_upper_triangles, plain.fc_upper_triangles[pos]),
                    "splits_are_plain_restricted": parts == plain_parts,
                }
            else:
                if cond == "rest_S1":
                    want = b.fc_session1_upper_triangles
                else:
                    want = b.fc_condition_upper_triangles[cond]
                train = b.trainvaltest_partition_indices["train"]
                checks = {
                    "same_cohort": ids == ref[0],
                    "same_splits": parts == ref[1],
                    "fc_is_condition": np.array_equal(b.fc_upper_triangles, want),
                    "fc_differs_from_rest": not np.array_equal(b.fc_upper_triangles, b.fc_condition_upper_triangles["rest"]
                                                               if "rest" in b.fc_condition_upper_triangles else plain.fc_upper_triangles[[pid.index(s) for s in ids]]),
                    "pca_mean_from_condition": np.allclose(b.fc_train_avg, want[train].mean(0), atol=1e-5),
                    "dense_matches_upper": np.array_equal(b.fc_matrices[0][np.triu_indices(b.fc_matrices.shape[1], k=1)], b.fc_upper_triangles[0]),
                }
            if args.direction == "sc2fc":  # the condition FC is the dataset target
                cond_fc = b.fc_upper_triangles
                part = HCP_Partition(b, "val")
                checks["target_y_is_condition"] = all(
                    np.allclose(part[i]["y"].cpu().numpy(), cond_fc[part.indices[i]]) for i in (0, len(part) - 1))
                checks["source_x_is_sc"] = np.allclose(part[0]["x"].cpu().numpy(), b.sc_upper_triangles[part.indices[0]])
            bad = [k for k, v in checks.items() if v is False]
            ok &= not bad
            print(f"seed {seed} {cond:10s} {'OK ' if not bad else 'FAIL'} {checks}", flush=True)
    print("ALL_OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
