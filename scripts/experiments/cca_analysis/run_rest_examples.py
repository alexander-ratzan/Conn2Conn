"""Spec v4 E5 H3 example figure inputs (GPU): three composite-loss decoders into rest, each shown on a representative
test subject.

Picks (seed 0, test; candidates = every decoder x combination into rest except SC -> rest, from run_composite.py's rows):
  high demeaned r, lower rank  = highest demeaned r among points with average rank below the median
  high rank, lower demeaned r  = highest average rank among points with demeaned r below the median
  best balance                 = highest sum of the z-scored demeaned r and average rank
Each picked decoder is retrained exactly as in run_composite.py (same head index -> same generator seed) and must
reproduce its stored test metrics; its predictions for every test subject are then scored per subject (demeaned r;
identification rank 1 - position / n of the subject's own prediction, rows = true subjects, as average rank). The
representative subject minimises |dr_i - mean dr| / SD + |rank_i - mean rank| / SD, so the drawn subject shows the
decoder's typical accuracy and identification. -> OUT/render/rest_examples_seed0.npz

    python scripts/experiments/cca_analysis/run_rest_examples.py
"""
import json

import numpy as np
import pandas as pd
import torch

from e5_common import OUT, SPLITS, VIEWS, load_views
from models.eval.metrics import compute_corr_matrix, corr_avg_rank
from run_composite import COMBOS, build_heads, head_inputs, train_head

SEED = 0


def picks(rows):
    pts = rows[(rows.target == "rest") & (rows.combo != "ridge") & (rows.source != "SC")].copy()
    z = lambda s: (s - s.mean()) / s.std()
    pts["zr"], pts["zk"] = z(pts.demeaned_pearson), z(pts.avg_rank)
    lo_rank = pts[pts.avg_rank < pts.avg_rank.median()]
    lo_dr = pts[pts.demeaned_pearson < pts.demeaned_pearson.median()]
    return pts, {"high demeaned r, lower rank": lo_rank.demeaned_pearson.idxmax(),
                 "high rank, lower demeaned r": lo_dr.avg_rank.idxmax(),
                 "best balance": (pts.zr + pts.zk).idxmax()}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pd.DataFrame(json.loads((OUT / "h3" / f"composite_seed{SEED}.json").read_text())["rows"])
    pts, chosen = picks(rows)
    z = np.load(OUT / "h3" / f"seed{SEED}.npz")
    meta = json.loads(str(z["alpha_json"]))
    enc = {s: {v: torch.as_tensor(z[f"enc__{s}__{v}"], device=dev) for v in VIEWS} for s in SPLITS}
    data, splits, ids = load_views(SEED)
    Y = {s: torch.as_tensor(data["rest"][splits[s]], device=dev) for s in SPLITS}
    mu = Y["train"].mean(0)
    heads = build_heads(meta)
    out = {"mu_rest": mu.cpu().numpy()}
    for j, (name, idx) in enumerate(chosen.items()):
        p = pts.loc[idx]
        i = next(k for k, h in enumerate(heads) if (h[1], h[2], h[3], h[4]) == (p.model, p.setting, p.source, "rest"))
        res, _, _, _, state, _ = train_head(head_inputs(heads[i], enc), Y, mu, heads[i][6], SEED * 1000 + i, dev)
        r = res[p.combo]
        diff = max(abs(r["demeaned_pearson"] - p.demeaned_pearson), abs(r["avg_rank"] - p.avg_rank))
        with torch.no_grad():
            P = mu + head_inputs(heads[i], enc)["test"].float() @ state[COMBOS.index(p.combo)]
        T = Y["test"]
        ccd = compute_corr_matrix(T - mu, P - mu)
        dr = torch.diagonal(ccd).cpu().numpy()
        _, rank = corr_avg_rank(cc=compute_corr_matrix(T, P), return_ranklist=True)
        rank = rank.cpu().numpy()
        score_ = np.abs(dr - dr.mean()) / dr.std() + np.abs(rank - rank.mean()) / rank.std()
        s = int(np.argmin(score_))
        print(f"{name}: {p.model} {p.setting} {p.source} {p.combo}; reproduced |delta| {diff:.1e}; "
              f"mean dr {dr.mean():.3f} rank {rank.mean():.3f}; subject {s}: dr {dr[s]:.3f} rank {rank[s]:.3f}", flush=True)
        assert diff < 2e-3, f"pick {name} does not reproduce its stored metrics"
        out.update({f"p{j}__name": name, f"p{j}__model": p.model, f"p{j}__setting": p.setting, f"p{j}__source": p.source,
                    f"p{j}__combo": p.combo, f"p{j}__test_dr": dr.mean(), f"p{j}__test_rank": rank.mean(),
                    f"p{j}__stored_dr": p.demeaned_pearson, f"p{j}__stored_rank": p.avg_rank,
                    f"p{j}__subject_dr": dr, f"p{j}__subject_rank": rank, f"p{j}__subject": s,
                    f"p{j}__subject_id": int(ids[splits["test"][s]]),
                    f"p{j}__pred": P[s].cpu().numpy(), f"p{j}__true": T[s].cpu().numpy()})
    (OUT / "render").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "render" / f"rest_examples_seed{SEED}.npz", **out)
    print("wrote render/rest_examples_seed0.npz")


if __name__ == "__main__":
    main()
