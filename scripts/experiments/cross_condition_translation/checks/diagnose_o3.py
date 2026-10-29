"""Diagnose the O3 failure (2026-10-28; run 1 -> diagnose_o3_run1.json, before FM-lin's per-target state term): FM-lin's mean matches ridge on linear-Gaussian synthetic data but its energy score
is worse than the closed-form Gaussian (+0.55 +- 0.06). Same data, training and cells as check_fm.py O3; FM-nl added so
its calibration is tested too (no check covered it).

Per model, cell and temperature T in {0.25, 0.5, 0.75, 1}:
  energy  FM and closed-form Gaussian (ridge mean + train residual covariance) test energy scores, absolute and difference
  scale   sqrt(tr sample covariance / tr residual covariance), sample covariance pooled over subjects around each
          subject's sample mean (1 = right overall width)
  shape   sd of log eigenvalues of R^-1/2 C R^-1/2 (C sample, R residual covariance; 0 = C a multiple of R)
  split   gap from the mean = energy(FM mean of 256 samples + the Gaussian's noise) - Gaussian; gap from the spread = rest
Reading: if one T brings scale to ~1 in every cell and the energy gap closes, the check was too strict (T is chosen in
the pipeline); if the scale needed differs by cell, or shape stays large, the shared state term is the limit.

    python scripts/experiments/cross_condition_translation/checks/diagnose_o3.py  -> checks/diagnose_o3.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from fm_common import energy_metrics, eval_cells, n_params, sample, train  # noqa: E402
from check_fm import HP, SynthData, ridge_cell  # noqa: E402

TEMPS = (0.5, 0.75, 1.0, 1.25, 1.5)


def spread(X, R):
    """X (K, n, d) samples; R (d, d) residual covariance -> (scale, shape)."""
    D = (X - X.mean(0, keepdims=True)).reshape(-1, X.shape[2])
    C = D.T @ D / (len(D) - X.shape[1])
    w, V = np.linalg.eigh(R)
    Ri = V @ np.diag(np.clip(w, 1e-8, None) ** -0.5) @ V.T
    ev = np.clip(np.linalg.eigvalsh(Ri @ C @ Ri), 1e-12, None)
    return float(np.sqrt(np.trace(C) / np.trace(R))), float(np.log(ev).std())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lin = SynthData(64, dev, nonlinear=False, seed=1)
    cells = [c for c in eval_cells()][::6]
    out = {}
    for model, hp in (("fm_lin", HP["fm_lin"]), ("fm_nl", dict(HP["fm_nl"], depth=2))):
        net, info = train(model, hp, lin, 2, max_epochs=1500)
        rows, floor = [], []
        for src, t, on in cells:
            P_r, R = ridge_cell(lin, src, t, on)
            Y = lin.fc["test"][t].cpu().numpy()
            G = P_r[None] + np.random.default_rng(0).multivariate_normal(np.zeros(64), R, size=(64, len(P_r)))
            es_g = energy_metrics(G, Y)["energy_score"]
            floor.append(spread(G, R)[1])
            Xm = sample(net, lin, "test", src, t, on, 256, 1.0).cpu().numpy().mean(0)
            es_mix = energy_metrics(Xm[None] + (G - P_r[None]), Y)["energy_score"]    # FM mean + the Gaussian's spread
            for T in TEMPS:
                X = sample(net, lin, "test", src, t, on, 64, T).cpu().numpy()
                es = energy_metrics(X, Y)["energy_score"]
                scale, shape = spread(X, R)
                rows.append(dict(src=int(src), tgt=t, sc=bool(on), T=T, es_fm=es, es_gauss=es_g, d_es=es - es_g,
                                 d_mean=es_mix - es_g, d_spread=es - es_mix, scale=scale, shape=shape))
        g_shape = float(np.mean(floor))
        out[model] = dict(hp=hp, n_params=n_params(net), best_epoch=info["best_epoch"], epochs_run=info["epochs_run"], rows=rows,
                          gauss_shape_floor=g_shape)
        print(f"{model} ({n_params(net):,} params): epochs {info['epochs_run']} (best {info['best_epoch']}); shape floor of the "
              f"Gaussian itself {g_shape:.3f}")
        for T in TEMPS:
            r = [x for x in rows if x["T"] == T]
            d, sc = np.array([x["d_es"] for x in r]), np.array([x["scale"] for x in r])
            print(f"  T {T:<4}: energy FM {np.mean([x['es_fm'] for x in r]):.3f} vs Gaussian {np.mean([x['es_gauss'] for x in r]):.3f}"
                  f" (diff {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}); scale {sc.mean():.3f} "
                  f"[{sc.min():.3f}, {sc.max():.3f}]; shape {np.mean([x['shape'] for x in r]):.3f}; gap from the mean "
                  f"{np.mean([x['d_mean'] for x in r]):+.3f}, from the spread {np.mean([x['d_spread'] for x in r]):+.3f}", flush=True)
            if T == 1.0:
                ok = d.mean() < 2 * d.std(ddof=1) / np.sqrt(len(d))
                print(f"  O3 energy criterion at T 1 (diff < 2 SE): {'pass' if ok else 'FAIL'}", flush=True)
    (HERE / "diagnose_o3.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
