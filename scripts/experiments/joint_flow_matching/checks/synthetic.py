"""Spec v4 E2 gate G1: A3 and A4 on simulated views with known conditional means, through the real training loop.

Nine views z^m = u M_m + sigma * eps (u ~ N(0, I_q), q = 16; k = 64; M_m random), so E[z^b | z^a] = z^a S_aa^-1 S_ab is
known. Train / val / test sizes as seed 0 (659 / 74 / 184). Fixed configuration, no search.
  G1a: A3 and A4 one-step test predictions reach R^2 >= g1_min_r2 against the true conditional mean on every pair;
  G1b: A4 trained with shuffled pairing has code-space demeaned r <= g1_max_shuffled_dr with the true target.
Also reports the oracle-fit ridge R^2 and A4 integrated (RK4 16) for reference.

    python scripts/experiments/joint_flow_matching/checks/synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CFG, pairs  # noqa: E402
import tune  # noqa: E402

N = {"train": 659, "val": 74, "test": 184}
Q, K, SIGMA = 16, 64, 0.5
HP = dict(lr=1e-3, weight_decay=1e-4, width=512, depth=3, dropout=0.0, token_dim=32, k=K)


class SynthData(tune.Data):
    def __init__(self, views, device, shuffle_pairs=False, seed=0):
        rng = np.random.default_rng(seed)
        self.views, self.device, self.code_scale = views, device, 1.0
        self.pairs = pairs(views)
        self.vid = {v: i for i, v in enumerate(views)}
        self.P = torch.tensor([[self.vid[a], self.vid[b]] for a, b in self.pairs], device=device)
        self.M = {v: rng.normal(size=(Q, K)) / np.sqrt(Q) for v in views}
        self.raw = {}
        for split, n in N.items():
            u = rng.normal(size=(n, Q))
            self.raw[split] = {v: u @ self.M[v] + SIGMA * rng.normal(size=(n, K)) for v in views}
        self.codes = {s: torch.tensor(np.stack([self.raw[s][v] for v in views]), dtype=torch.float32, device=device)
                      for s in N}
        n = N["train"]
        g = torch.Generator().manual_seed(99)
        self.target_perm = torch.randperm(n, generator=g).to(device) if shuffle_pairs else torch.arange(n, device=device)

    def cond_mean(self, a, b, za):
        Saa = self.M[a].T @ self.M[a] + SIGMA ** 2 * np.eye(K)
        Sab = self.M[a].T @ self.M[b]
        return za @ np.linalg.solve(Saa, Sab)


def r2(pred, truth):
    return 1 - np.square(pred - truth).sum() / np.square(truth - truth.mean(0)).sum()


def code_dr(pred, true):
    a, b = pred - pred.mean(0), true - true.mean(0)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views = [f"v{i}" for i in range(9)]
    g = CFG["gates"]
    data = SynthData(views, device)
    models = {arm: tune.train(arm, HP, data, seed=0)[0] for arm in ("A3", "A4")}
    rows, ok = [], True
    for p, (a, b) in enumerate(data.pairs):
        za = data.raw["test"][a]
        cm = data.cond_mean(a, b, za)
        Ztr_a, Ztr_b = data.raw["train"][a], data.raw["train"][b]
        ridge = np.linalg.solve(Ztr_a.T @ Ztr_a + 1e-3 * np.eye(K), Ztr_a.T @ Ztr_b)
        row = dict(pair=f"{a}->{b}", ridge=r2(za @ ridge, cm))
        row["A3"] = r2(tune.predict("A3", models["A3"], data, "test", K, p).cpu().numpy(), cm)
        row["A4_onestep"] = r2(tune.predict("A4", models["A4"], data, "test", K, p, n_steps=1).cpu().numpy(), cm)
        row["A4_rk4_16"] = r2(tune.predict("A4", models["A4"], data, "test", K, p, n_steps=16).cpu().numpy(), cm)
        rows.append(row)
    r = {k: np.array([row[k] for row in rows]) for k in ("ridge", "A3", "A4_onestep", "A4_rk4_16")}
    for k, v in r.items():
        print(f"R^2 vs true conditional mean, {k}: min {v.min():.3f} median {np.median(v):.3f} over {len(v)} pairs")
    g1a = r["A3"].min() >= g["g1_min_r2"] and r["A4_onestep"].min() >= g["g1_min_r2"]
    print(f"{'OK  ' if g1a else 'FAIL'} G1a: A3 and A4 one-step min R^2 >= {g['g1_min_r2']}")

    sdata = SynthData(views, device, shuffle_pairs=True)
    m = tune.train("A4", HP, sdata, seed=0)[0]
    drs = [code_dr(tune.predict("A4", m, sdata, "test", K, p, n_steps=16).cpu().numpy(), sdata.raw["test"][b])
           for p, (_, b) in enumerate(sdata.pairs)]
    g1b = max(np.mean(drs), 0) <= g["g1_max_shuffled_dr"]
    print(f"{'OK  ' if g1b else 'FAIL'} G1b: shuffled-pairing A4 mean code demeaned r {np.mean(drs):.4f} "
          f"(max over pairs {np.max(drs):.4f}) <= {g['g1_max_shuffled_dr']}")
    print("ALL_OK" if (g1a and g1b) else "FAILED")


if __name__ == "__main__":
    main()
