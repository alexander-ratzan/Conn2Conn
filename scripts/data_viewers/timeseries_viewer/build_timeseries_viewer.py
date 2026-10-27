"""Self-contained HTML viewer of regional timeseries and connectivity for a few subjects, all 8 conditions.

Per subject x condition it embeds the parcel-mean timeseries (int16 with a per-region scale), the run bounds, the
full-scan FC (fc/ cache = xcp-d combined-run relmat) and the full-scan intrinsic timescale of every region
(data/timeseries_metrics.py, averaged over runs). The page computes, in the browser and with the same definitions:
window FC (windows never span a run boundary), window timescales, and autocorrelation functions (full scan and window)
of the mean timeseries and the two selected regions. A self-check block holds Python values for one window; the page
recomputes them and shows the largest difference. (Multiscale entropy, data/timeseries_metrics.py, was dropped from the
page 2026-10-27: on band-passed data it mostly re-expresses the ACF.)

Data: Conn2Conn_data/timeseries/ (data/timeseries.py) and Conn2Conn_data/fc/. Page: timeseries_viewer.template on the
shared viewer_kit (one data block per subject x condition). Output, next to this file (generated, not tracked):
timeseries_viewer_{P}.html and its data cache timeseries_viewer_{P}_data.json (--from-cache rebuilds the page only).

    python scripts/data_viewers/timeseries_viewer/build_timeseries_viewer.py [--subjects 100206 ...] [--n-subjects 4]
        [--parcellation Glasser] [--from-cache] [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.timeseries import CONDITIONS, TimeseriesCache, corr, window_starts  # noqa: E402
from data.timeseries_metrics import acf, timescale, timescale_runs  # noqa: E402
from scripts.data_viewers.viewer_kit import Viewer, int16_columns, int16_unit, packed, round_list  # noqa: E402

FC_ROOT = Path("/scratch/asr655/neuroinformatics/Conn2Conn_data/fc")
WINDOW, STEP = 144, 24
ACF_CHECK_LAGS = 40
BAND = (0.01, 0.08)   # xcp-d band-pass (Hz), for the page's filter-only reference ACF
NETWORK_COLUMN = {"Glasser": "community_yeo", "4S456Parcels": "network_label"}


def fc_upper(condition, parcellation, sid):
    suffix = "" if condition == "rest" else f"_task-{condition}"
    d = FC_ROOT / f"parc-{parcellation}_hemi-both{suffix}"
    ids = np.load(d / "subject_ids.npy")
    i = int(np.searchsorted(ids, sid))
    if i >= len(ids) or ids[i] != sid:
        raise KeyError(f"sub-{sid} not in {d}")
    return np.load(d / "upper_triangles.npy", mmap_mode="r")[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--parcellation", default="Glasser", choices=sorted(NETWORK_COLUMN))
    Viewer.add_args(ap)
    a = ap.parse_args()
    viewer = Viewer(__file__)
    if a.from_cache:
        viewer.write(*viewer.load_cache(a.parcellation), variant=a.parcellation, cache=False, smoke=a.smoke)
        return

    caches = {c: TimeseriesCache(c, a.parcellation) for c in CONDITIONS}
    common = sorted(set.intersection(*(set(map(int, t.subject_ids)) for t in caches.values())))
    subjects = a.subjects or [common[i] for i in np.linspace(0, len(common) - 1, a.n_subjects).round().astype(int)]
    print(f"{len(common)} subjects have all 8 conditions; viewer subjects {subjects}", flush=True)

    labels = caches["rest"].labels
    atlas = pd.read_csv(REPO_ROOT / "data" / "atlas_info" / f"{a.parcellation}_dseg_reformatted.csv")
    if [str(x) for x in atlas["label"]] != labels:
        raise ValueError("atlas_info labels differ from the cache node labels")
    network = atlas[NETWORK_COLUMN[a.parcellation]].fillna("other").astype(str).tolist()

    sets = {}
    check = None
    for sid in subjects:
        for c in CONDITIONS:
            t0 = time.time()
            x = caches[c].subject(sid)
            bounds = caches[c].run_bounds(sid)
            q, scale = int16_columns(x)
            sets[f"{sid}/{c}"] = {
                "subject": str(sid), "condition": c,
                "n_tr": int(len(x)),
                "runs": [[r, int(s), int(n)] for r, s, n in caches[c].run_table(sid).itertuples(index=False)],
                "q": packed(q), "scale": scale.tolist(),
                "fc": packed(int16_unit(fc_upper(c, a.parcellation, sid))),
                "tau": round_list(timescale_runs(x, bounds, caches[c].tr), 4),
            }
            if check is None:  # Python reference for the page's own computations, on the dequantized values
                xq = q.astype(np.float64) * scale
                s0 = int(window_starts(bounds, WINDOW, STEP)[1])
                w = xq[s0:s0 + WINDOW]
                C = corr(w)
                pairs = [(0, 1), (5, 200), (len(labels) - 2, len(labels) - 1)]
                check = {"subject": str(sid), "condition": c, "start": s0, "length": WINDOW,
                         "tau": round_list(timescale(w, caches[c].tr)),
                         "regions": [0, 1],
                         "acf": [round_list(row) for row in acf(w[:, :2], ACF_CHECK_LAGS).T],
                         "fc_pairs": [[i, j, float(C[i, j])] for i, j in pairs]}
            print(f"  sub-{sid} {c}: {len(x)} TRs, {time.time() - t0:.1f}s", flush=True)

    meta = {"parcellation": a.parcellation, "tr": caches["rest"].tr, "labels": labels, "network": network,
            "conditions": list(CONDITIONS), "subjects": [str(s) for s in subjects],
            "window": WINDOW, "step": STEP, "band": list(BAND), "check": check}
    viewer.write(meta, sets, variant=a.parcellation, smoke=a.smoke)


if __name__ == "__main__":
    main()
