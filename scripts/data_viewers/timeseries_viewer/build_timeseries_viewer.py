"""Self-contained HTML viewer of regional timeseries and connectivity for a few subjects, all 8 conditions.

Per subject x condition it embeds the parcel-mean timeseries (int16 with a per-region scale), the run bounds, the
full-scan FC (fc/ cache = xcp-d combined-run relmat) and full-scan regional metrics from data/timeseries_metrics.py
(intrinsic timescale averaged over runs; multiscale sample entropy with counts pooled over runs, and its area over
scales; defaults m, r, scales from data/timeseries_metrics.py, after Wijesinghe et al. 2026). The page computes,
in the browser and with the same definitions: window FC (windows never span a run boundary), window timescales for all
regions and window multiscale entropy for the two selected regions. A self-check block holds Python values for one
window; the page recomputes them and shows the largest difference.

Data: Conn2Conn_data/timeseries/ (data/timeseries.py) and Conn2Conn_data/fc/. Template: timeseries_viewer.template (not
a viewer itself; __META__ and __BLOCKS__ are replaced). As for the E1 latent viewer (cca_latent_bench/viewer), the data
go in one <script type="application/json"> block per subject x condition, with base64 strings split into lines of at
most CHUNK characters: editor previews (Cursor) fail on very long lines. Output (generated, not tracked):
results/data_viewers/timeseries_viewer_{P}.html.

    python scripts/data_viewers/timeseries_viewer/build_timeseries_viewer.py [--subjects 100206 ...] [--n-subjects 4]
        [--parcellation Glasser] [--out PATH]
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.timeseries import CONDITIONS, TimeseriesCache, corr, window_starts  # noqa: E402
from data.timeseries_metrics import (M, R_FRAC, SCALES, mse_auc, multiscale_entropy,  # noqa: E402
                                     multiscale_entropy_runs, timescale, timescale_runs)

FC_ROOT = Path("/scratch/asr655/neuroinformatics/Conn2Conn_data/fc")
TEMPLATE = HERE / "timeseries_viewer.template"
CHUNK = 200_000
WINDOW, STEP = 144, 24
NETWORK_COLUMN = {"Glasser": "community_yeo", "4S456Parcels": "network_label"}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def chunks(s):
    return [s[i:i + CHUNK] for i in range(0, len(s), CHUNK)]


def block(rec):
    """One <script type="application/json"> block; each base64 chunk on its own line (no line > ~CHUNK chars)."""
    head = {k: v for k, v in rec.items() if k not in ("q", "fc")}
    body = json.dumps(head, separators=(",", ":"))[:-1]
    for k in ("q", "fc"):
        body += f',\n"{k}":[\n' + ",\n".join(json.dumps(c) for c in chunks(rec[k])) + "\n]"
    return (f'<script class="rec" type="application/json" data-key="{rec["subject"]}/{rec["condition"]}">\n'
            f"{body}}}\n</script>")


def quantize(x):
    """int16 codes and float64 per-column scales with x ~= q * scale."""
    scale = np.abs(x).max(0).astype(np.float64) / 32767.0
    scale[scale == 0] = 1.0
    q = np.round(x / scale).astype("<i2")
    return q, scale


def fc_upper(condition, parcellation, sid):
    suffix = "" if condition == "rest" else f"_task-{condition}"
    d = FC_ROOT / f"parc-{parcellation}_hemi-both{suffix}"
    ids = np.load(d / "subject_ids.npy")
    i = int(np.searchsorted(ids, sid))
    if i >= len(ids) or ids[i] != sid:
        raise KeyError(f"sub-{sid} not in {d}")
    return np.load(d / "upper_triangles.npy", mmap_mode="r")[i]


def round_list(a, nd=None):
    """Flat list for JSON: NaN -> None, optionally rounded."""
    return [None if not np.isfinite(v) else (float(v) if nd is None else round(float(v), nd))
            for v in np.asarray(a, dtype=float).ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--parcellation", default="Glasser", choices=sorted(NETWORK_COLUMN))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "data_viewers" / f"timeseries_viewer_{a.parcellation}.html"

    caches = {c: TimeseriesCache(c, a.parcellation) for c in CONDITIONS}
    common = sorted(set.intersection(*(set(map(int, t.subject_ids)) for t in caches.values())))
    subjects = a.subjects or [common[i] for i in np.linspace(0, len(common) - 1, a.n_subjects).round().astype(int)]
    print(f"{len(common)} subjects have all 8 conditions; viewer subjects {subjects}", flush=True)

    labels = caches["rest"].labels
    atlas = pd.read_csv(REPO_ROOT / "data" / "atlas_info" / f"{a.parcellation}_dseg_reformatted.csv")
    if [str(x) for x in atlas["label"]] != labels:
        raise ValueError("atlas_info labels differ from the cache node labels")
    network = atlas[NETWORK_COLUMN[a.parcellation]].fillna("other").astype(str).tolist()

    data = {}
    check = None
    for sid in subjects:
        data[str(sid)] = {}
        for c in CONDITIONS:
            t0 = time.time()
            x = caches[c].subject(sid)
            bounds = caches[c].run_bounds(sid)
            q, scale = quantize(x)
            mse = multiscale_entropy_runs(x, bounds, SCALES, M, R_FRAC)
            data[str(sid)][c] = {
                "subject": str(sid), "condition": c,
                "n_tr": int(len(x)),
                "runs": [[r, int(s), int(n)] for r, s, n in caches[c].run_table(sid).itertuples(index=False)],
                "q": b64(q), "scale": scale.tolist(),
                "fc": b64(np.round(np.clip(fc_upper(c, a.parcellation, sid), -1, 1) * 32767).astype("<i2")),
                "tau": round_list(timescale_runs(x, bounds, caches[c].tr), 4),
                "mse": [round_list(row, 4) for row in mse],
                "mse_auc": round_list(mse_auc(mse), 4),
            }
            if check is None:  # Python reference for the page's own computations, on the dequantized values
                xq = q.astype(np.float64) * scale
                s0 = int(window_starts(bounds, WINDOW, STEP)[1])
                w = xq[s0:s0 + WINDOW]
                C = corr(w)
                pairs = [(0, 1), (5, 200), (len(labels) - 2, len(labels) - 1)]
                check = {"subject": str(sid), "condition": c, "start": s0, "length": WINDOW,
                         "tau": round_list(timescale(w, caches[c].tr)),
                         "mse_regions": [0, 1],
                         "mse": [round_list(row) for row in multiscale_entropy(w[:, :2], SCALES, M, R_FRAC).T],
                         "fc_pairs": [[i, j, float(C[i, j])] for i, j in pairs]}
            print(f"  sub-{sid} {c}: {len(x)} TRs, {time.time() - t0:.1f}s", flush=True)

    meta = {"parcellation": a.parcellation, "tr": caches["rest"].tr, "labels": labels, "network": network,
            "conditions": list(CONDITIONS), "subjects": [str(s) for s in subjects],
            "scales": list(SCALES), "m": M, "r_frac": R_FRAC, "window": WINDOW, "step": STEP, "check": check}
    blocks = "\n".join(block(rec) for recs in data.values() for rec in recs.values())
    html = TEMPLATE.read_text().replace("__META__", json.dumps(meta, separators=(",", ":"))).replace("__BLOCKS__", blocks)
    longest = max(len(line) for line in html.splitlines())
    if longest > 2 * CHUNK:
        raise ValueError(f"a line of {longest} characters; keep lines short for editor previews")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, longest line {longest} characters)")


if __name__ == "__main__":
    main()
