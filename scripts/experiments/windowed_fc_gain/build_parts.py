"""windowed_fc_gain G7 data: every E2 cohort subject's part FC samples (halves, quarters) for one condition, encoded
into each seed's E2 block FC PCA code space (fit on full train targets; E2's `SeedData.cs`), plus a re-encoding of the
full FC rebuilt from the timeseries as a check against the cached codes (G7.0).

Rows of `codes` (per seed): 0 full (rebuilt), 1-2 halves, 3-6 quarters; (7, n_cohort, K_CODES). Subjects not usable for
part samples (gate 6 inclusion) are NaN in rows 1-6 and `included` is False.

    python scripts/experiments/windowed_fc_gain/build_parts.py --condition wm
        -> results/cache/experiments/windowed_fc_gain/parts/wm.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching")]
from common import SeedData  # noqa: E402  (E2 per-seed code space and cohort)
from data.timeseries import TimeseriesCache, part_bounds, segment_fc  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yml").read_text())
PARTS_DIR = REPO_ROOT / "results" / "cache" / "experiments" / "windowed_fc_gain" / "parts"
SEEDS = (0, 1, 2, 3, 4)
K_CODES = 256
ROWS = ["full"] + [f"half{k}" for k in range(2)] + [f"quarter{k}" for k in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CFG["conditions"])
    c = ap.parse_args().condition
    sds = {s: SeedData(s, "main") for s in SEEDS}
    ids = sds[0].subject_ids
    for s in SEEDS:
        assert np.array_equal(sds[s].subject_ids, ids), "cohort differs across seeds"
    inc = pd.read_csv(HERE / "results" / f"{c}_inclusion.csv").set_index("subject")["included"]
    ts = TimeseriesCache(c, CFG["parcellation"])
    P = len(ts.labels)
    iu = np.triu_indices(P, k=1)
    n = len(ids)
    edges = np.full((len(ROWS), n, len(iu[0])), np.nan, dtype=np.float32)
    included = np.zeros(n, dtype=bool)
    for i, sid in enumerate(map(int, ids)):
        x = ts.subject(sid)
        b = ts.run_bounds(sid)
        edges[0, i] = segment_fc(x, b, CFG["estimator"])[iu]
        if bool(inc.get(sid, False)):
            included[i] = True
            for k in range(2):
                edges[1 + k, i] = segment_fc(x, part_bounds(b, k, 2), CFG["estimator"])[iu]
            for k in range(4):
                edges[3 + k, i] = segment_fc(x, part_bounds(b, k, 4), CFG["estimator"])[iu]
        if i % 200 == 0:
            print(f"{c}: {i}/{n}", flush=True)
    out = dict(subject_ids=ids, included=included, rows=np.array(ROWS))
    for s, sd in sds.items():
        codes = np.stack([sd.cs.encode(c, e, K_CODES) for e in edges])            # NaN rows stay NaN
        ref = sd.codes[c][:, :K_CODES]
        rel = float(np.abs(codes[0] - ref).max() / np.abs(ref).max())
        out[f"codes_seed{s}"] = codes.astype(np.float32)
        out[f"full_check_seed{s}"] = rel
        print(f"{c} seed {s}: rebuilt full codes vs cache max |diff| / max |code| = {rel:.2e}", flush=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PARTS_DIR / f"{c}.npz", **out)
    print(f"wrote {PARTS_DIR / f'{c}.npz'} ({included.sum()} / {n} subjects with part samples)")


if __name__ == "__main__":
    main()
