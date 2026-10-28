"""Spec v4 E4.2 checks before the flow search (D10): optimisability of FM-nl and FM-lin, then G1.

  O1  real data (seed 0, code size k): training loss falls (last 5 epochs < 0.9 x first 5) with finite gradients
  O2  capacity: each model, unregularised, memorises 32 training subjects (mean per-subject r of the generated mean
      with the true target codes > 0.8 over every single-source cell)
  O3  FM-lin on synthetic linear-Gaussian data recovers the closed-form linear-Gaussian: its mean agrees with ridge
      (r > 0.9) and its test energy score is not worse than the ridge + residual-covariance Gaussian beyond 2 SE
  G1  synthetic data with a planted nonlinear condition -> condition map and a linear SC path: FM-nl beats ridge on
      >= 80 % of the FC -> FC cells (test, per-subject r of codes) and is within 2 SE of ridge on the SC-only cells
Writes checks/check_fm.json and prints ALL_OK / FAIL lines.

    python scripts/experiments/cross_condition_translation/checks/check_fm.py --k 256
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fm_common import NONE, FlowData, energy_metrics, eval_cells, ridge_gcv_np, sample, train  # noqa: E402
from e4_common import CFG  # noqa: E402

HP = {"fm_nl": dict(lr=1e-3, wd=1e-3, width=256, depth=1, dropout=0.1, emb_dim=32, k_sc=64),
      "fm_lin": dict(lr=1e-3, wd=1e-4, k_sc=64)}
results, ok_all = {}, True


def report(name, ok, **info):
    global ok_all
    ok_all &= bool(ok)
    results[name] = dict(ok=bool(ok), **info)
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {info}", flush=True)


def row_r(P, Y, mu):
    a, b = P - mu, Y - mu
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


class SynthData(FlowData):
    """Same interface as FlowData on simulated codes: a subject latent u (8-d); condition m's codes f_m(u) + noise
    (nonlinear: two-layer tanh map; linear: a linear map); SC = L u + noise (linear path)."""

    def __init__(self, k, device, nonlinear, seed=0, n=(659, 74, 184), noise=0.5):
        rng = np.random.default_rng(seed)
        self.k, self.device, self.seed, self.sd = k, device, seed, None
        W1 = rng.normal(size=(8, 8, 32)) / np.sqrt(8)
        W2 = rng.normal(size=(8, 32, k)) / np.sqrt(32)
        Wl = rng.normal(size=(8, 8, k)) / np.sqrt(8)
        L = rng.normal(size=(8, 64)) / np.sqrt(8)
        fc, sc = {}, {}
        for split, m in zip(("train", "val", "test"), n):
            u = rng.normal(size=(m, 8))
            if nonlinear:
                Y = np.stack([np.tanh(2.0 * u @ W1[c]) @ W2[c] for c in range(8)])
            else:
                Y = np.stack([u @ Wl[c] for c in range(8)])
            fc[split] = Y + noise * rng.normal(size=Y.shape)
            sc[split] = u @ L + noise * rng.normal(size=(m, 64))
        self.s_fc, self.s_sc = float(fc["train"].std()), float(sc["train"].std())
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.fc = {s: t(fc[s] / self.s_fc) for s in fc}
        self.sc = {s: t(sc[s] / self.s_sc) for s in sc}
        self.raw = fc
        self.n = {s: fc[s].shape[1] for s in fc}


def ridge_cell(data, a, t, on):
    Xs = {}
    for s in ("train", "test"):
        parts = [] if a == NONE else [data.fc[s][a].cpu().numpy()]
        if on:
            parts.append(data.sc[s].cpu().numpy()[:, :64])
        Xs[s] = np.concatenate(parts, axis=1)
    Y = data.fc["train"][t].cpu().numpy()
    B, a_rel = ridge_gcv_np(Xs["train"], Y)
    resid = Y - Xs["train"] @ B
    return Xs["test"] @ B, np.cov(resid, rowvar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="fewer epochs (smoke test of the checks themselves)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = (lambda x: max(20, x // 10)) if a.quick else (lambda x: x)

    # O1 / O2 on real data
    data = FlowData(0, a.k, dev)
    for model, hp in HP.items():
        net, info = train(model, hp, data, 0, max_epochs=E(150), log_every=50)
        h = np.array([x[1] for x in info["history"]])
        report(f"O1 {model} loss falls", h[-5:].mean() < 0.9 * h[:5].mean() and np.isfinite(h).all(),
               first=float(h[:5].mean()), last=float(h[-5:].mean()))
        sub = torch.arange(32, device=dev)
        hp0 = dict(hp, dropout=0.0) if model == "fm_nl" else hp
        net, _ = train(model, hp0, data, 1, max_epochs=E(1500), subset=sub, regularise=False)
        rs = []
        for src, t, on in [c for c in eval_cells() if c[0] != NONE][::7]:
            P = sample(net, data, "train", src, t, on, 32, 1.0).mean(0)[:32].cpu().numpy()
            Y = data.fc["train"][t][:32].cpu().numpy()
            rs.append(row_r(P, Y, data.fc["train"][t].cpu().numpy().mean(0)).mean())
        report(f"O2 {model} memorises 32 subjects", np.mean(rs) > 0.8, mean_r=float(np.mean(rs)))

    # O3: FM-lin recovers the linear-Gaussian on linear-Gaussian data
    lin = SynthData(64, dev, nonlinear=False, seed=1)
    net, _ = train("fm_lin", HP["fm_lin"], lin, 2, max_epochs=E(1500))
    agree, d_es = [], []
    for src, t, on in [c for c in eval_cells()][::6]:
        X = sample(net, lin, "test", src, t, on, 64, 1.0).cpu().numpy()
        P_r, cov = ridge_cell(lin, src, t, on)
        Y = lin.fc["test"][t].cpu().numpy()
        agree.append(np.mean([np.corrcoef(p, q)[0, 1] for p, q in zip(X.mean(0), P_r)]))
        G = P_r[None] + np.random.default_rng(0).multivariate_normal(np.zeros(64), cov, size=(64, len(P_r)))
        d_es.append(energy_metrics(X, Y)["energy_score"] - energy_metrics(G, Y)["energy_score"])
    d = np.array(d_es)
    report("O3 FM-lin = closed-form linear-Gaussian", np.mean(agree) > 0.9 and d.mean() < 2 * d.std(ddof=1) / np.sqrt(len(d)),
           mean_r_with_ridge=float(np.mean(agree)), energy_diff=float(d.mean()), energy_diff_se=float(d.std(ddof=1) / np.sqrt(len(d))))

    # G1: planted nonlinear FC -> FC map, linear SC path
    nl = SynthData(64, dev, nonlinear=True, seed=3)
    net, _ = train("fm_nl", dict(HP["fm_nl"], width=256, depth=2), nl, 4, max_epochs=E(3000))
    wins, n_ff, sc_d = 0, 0, []
    for src, t, on in eval_cells():
        P = sample(net, nl, "test", src, t, on, 64, 1.0).mean(0).cpu().numpy()
        P_r, _ = ridge_cell(nl, src, t, on)
        Y = nl.fc["test"][t].cpu().numpy()
        mu = nl.fc["train"][t].cpu().numpy().mean(0)
        r_fm, r_rd = row_r(P, Y, mu), row_r(P_r, Y, mu)
        if src == NONE:
            sc_d.append(r_fm.mean() - r_rd.mean())
        elif not on:
            n_ff += 1
            wins += int(r_fm.mean() > r_rd.mean())
    sc_d = np.array(sc_d)
    report("G1 FM-nl beats ridge on a planted nonlinear map", wins / n_ff >= 0.8, win_frac=wins / n_ff)
    report("G1 FM-nl matches ridge on the linear SC path", abs(sc_d.mean()) <= 2 * sc_d.std(ddof=1) / np.sqrt(len(sc_d)) or sc_d.mean() > 0,
           mean_diff=float(sc_d.mean()), se=float(sc_d.std(ddof=1) / np.sqrt(len(sc_d))))

    print("ALL_OK" if ok_all else "FAIL", flush=True)
    (HERE / "check_fm.json").write_text(json.dumps(dict(k=a.k, quick=a.quick, ok=ok_all, results=results), indent=1, default=float))


if __name__ == "__main__":
    main()
