"""Which subjects have which FC conditions, within the modeling cohort.

Cohort = the `HCP_Base` canonical subjects before any FC condition is required: metadata subjects with SC,
FreeSurfer, parcel node features and rest FC (default SC metric, log1p). Conditions = rest, rest_S1, rest_S2 and
the seven HCP tasks, from the `Conn2Conn_data/fc/` caches (see data/data_caching/build_fc_cache.py).

Writes one row per cohort subject (partition, "has all conditions" flag and missing conditions per parcellation)
and prints the summary as markdown tables (the table at the top of FC_matrix_analysis.ipynb).

    python data/data_caching/fc_condition_availability.py [--out results/tables/fc_condition_subject_availability.tsv]
"""
import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from data.dataset_utils import (  # noqa: E402
    DEFAULT_CONN2CONN_CACHE_ROOT,
    FC_TASK_CONDITIONS,
    load_freesurfer_data,
    load_metadata,
)

PARCELLATIONS = ("Glasser", "4S456Parcels")
CONDITIONS = ("rest", "rest_S1", "rest_S2", *FC_TASK_CONDITIONS)
SC_METRIC_DIR = "metric-sift_invnodevol_radius2_count_connectivity_log1p-1"
NODEFEAT_DIR = "vol-volume_mm3_cent-centroid_mm"


def _ids(cache_root, rel):
    return set(np.load(os.path.join(cache_root, rel, "subject_ids.npy")).astype(int).tolist())


def condition_subjects(cache_root, parc):
    fc = f"fc/parc-{parc}_hemi-both"
    out = {"rest": _ids(cache_root, fc), "rest_S1": _ids(cache_root, f"{fc}_session1"),
           "rest_S2": _ids(cache_root, f"{fc}_session2")}
    out.update({t: _ids(cache_root, f"{fc}_task-{t}") for t in FC_TASK_CONDITIONS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "results/tables/fc_condition_subject_availability.tsv"))
    ap.add_argument("--cache-root", default=DEFAULT_CONN2CONN_CACHE_ROOT)
    a = ap.parse_args()

    md, _ = load_metadata(shuffle_seed=0)
    partition = dict(zip(md["subject"].astype(int), md["train_val_test"]))
    freesurfer = set(load_freesurfer_data()["subject"].astype(int))

    per_parc = {}
    for parc in PARCELLATIONS:
        conds = condition_subjects(a.cache_root, parc)
        cohort = (set(partition) & freesurfer & conds["rest"]
                  & _ids(a.cache_root, f"sc/parc-{parc}_hemi-both_{SC_METRIC_DIR}")
                  & _ids(a.cache_root, f"parcel_node_features/parc-{parc}_hemi-both_{NODEFEAT_DIR}"))
        per_parc[parc] = (conds, cohort)
    cohort = sorted(set.union(*(c for _, c in per_parc.values())))

    rows = []
    for s in cohort:
        row = {"subject": s, "partition": partition[s]}
        for parc, (conds, parc_cohort) in per_parc.items():
            missing = [c for c in CONDITIONS if s not in conds[c]]
            row[f"all_conditions_{parc}"] = int(s in parc_cohort and not missing)
            row[f"missing_{parc}"] = ",".join(missing)
        rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    table.to_csv(a.out, sep="\t", index=False)

    # --- summary (markdown) ---
    print(f"Wrote {a.out}: {len(table)} cohort subjects\n")
    lines = ["| Condition | " + " | ".join(f"{p} (in cache / in cohort / cohort missing)" for p in PARCELLATIONS) + " |",
             "|---|" + "---|" * len(PARCELLATIONS)]
    for c in CONDITIONS:
        cells = []
        for parc in PARCELLATIONS:
            conds, parc_cohort = per_parc[parc]
            cells.append(f"{len(conds[c])} / {len(conds[c] & parc_cohort)} / {len(parc_cohort - conds[c])}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    print("\n".join(lines) + "\n")

    lines = ["| Parcellation | Cohort | All conditions | train | val | test |", "|---|---|---|---|---|---|"]
    for parc in PARCELLATIONS:
        full = table[table[f"all_conditions_{parc}"] == 1]
        split = full["partition"].value_counts()
        lines.append(f"| {parc} | {len(per_parc[parc][1])} | {len(full)} | "
                     + " | ".join(str(int(split.get(p, 0))) for p in ("train", "val", "test")) + " |")
    print("\n".join(lines) + "\n")

    for parc in PARCELLATIONS:
        miss = table.loc[table[f"all_conditions_{parc}"] == 0, f"missing_{parc}"].str.split(",")
        patterns = collections.Counter(",".join(m) for m in miss)
        print(f"{parc}: {len(miss)} cohort subjects missing ≥1 condition; patterns: "
              + "; ".join(f"{k} ×{v}" for k, v in patterns.most_common()))


if __name__ == "__main__":
    main()
