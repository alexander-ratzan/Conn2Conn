"""Spec v4 E2.1 gate G1 (E2.1-shaped part): the real E2.1 training code on simulated subjects with known answers.

Each subject has a 16-d trait u; the input (SC-like, 256-d) is x = u B + small noise; each of 8 conditions has an
encoding g^m = u M_m + s(u) * eps (32-d) with a subject-specific spread s(u) = 0.3 exp(0.6 u_1). So E[g^m | x] is
(almost) linear and the spread of a subject's 8 encodings depends on the input. Splits as seed 0 (659 / 74 / 184).
Fixed configurations, no search.
  G1a: corrector and bridge (one step) reach R^2 >= 0.9 against E[g^m | x] for every condition;
  G1b: the conditional flow trained with x shuffled across subjects carries no subject signal (code demeaned r of its
       sample means with the true means <= 0.02);
  G1c: without a condition token the flow's per-subject spread correlates >= 0.5 with the true expected spread, and
       its energy distance to each subject's 8 encodings is below the linear-Gaussian reference's.

    python scripts/experiments/joint_flow_matching/checks/synthetic_e21.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e21_run as R  # noqa: E402
from e21_common import crossfit, ridge_fit  # noqa: E402

N = {"train": 659, "val": 74, "test": 184}
Q, P_IN, D = 16, 256, 32
HP = dict(lr=1e-3, wd=1e-4, width=256, depth=2, dropout=0.0, token_dim=16, k_sc=P_IN, p0=0.25)


class Synth(R.Data):
    def __init__(self, device, shuffle_sc=False, seed=0):
        rng = np.random.default_rng(seed)
        self.device, self.d, self.seed = device, D, seed
        B = rng.normal(size=(Q, P_IN)) / np.sqrt(Q)
        M = rng.normal(size=(8, Q, D)) / np.sqrt(Q)
        self.raw, self.x, self.truth, self.spread = {}, {}, {}, {}
        for split, n in N.items():
            u = rng.normal(size=(n, Q))
            s = 0.3 * np.exp(0.6 * u[:, 0])
            self.x[split] = u @ B + 0.05 * rng.normal(size=(n, P_IN))
            mu = np.einsum("nq,mqd->mnd", u, M)
            self.raw[split] = mu + s[None, :, None] * rng.normal(size=(8, n, D))
            # E[u | x] for the Gaussian model, then the conditional mean of each condition
            Sxx = B.T @ B + 0.05 ** 2 * np.eye(P_IN)
            Eu = self.x[split] @ np.linalg.solve(Sxx, B.T)
            self.truth[split] = np.einsum("nq,mqd->mnd", Eu, M)
            self.spread[split] = mu.var(axis=0, ddof=1).sum(1) + D * s ** 2          # expected trace of the cloud
        self.rc_raw = {}
        for split in N:
            self.rc_raw[split] = np.stack([ridge_fit(self.x["train"], self.raw["train"][m])(self.x[split]) for m in range(8)])
        self.rc_raw["train"] = np.stack([crossfit(self.x["train"], self.raw["train"][m]) for m in range(8)])
        resid = (self.raw["train"] - self.rc_raw["train"]).reshape(-1, D)
        self.cov = np.cov(resid, rowvar=False)
        self.s_lat, self.s_sc = float(self.raw["train"].std()), float(self.x["train"].std())
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.enc = {s: t(self.raw[s] / self.s_lat) for s in N}
        self.rc = {s: t(self.rc_raw[s] / self.s_lat) for s in N}
        self.sc = {s: t(self.x[s] / self.s_sc) for s in N}
        n = N["train"]
        self.perm = (torch.randperm(n, generator=torch.Generator().manual_seed(99)).to(device) if shuffle_sc
                     else torch.arange(n, device=device))


def r2(pred, truth):
    return 1 - np.square(pred - truth).sum() / np.square(truth - truth.mean(0)).sum()


def code_dr(pred, true):
    a, b = pred - pred.mean(0), true - true.mean(0)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = Synth(device)
    ok = True
    ridge_r2 = min(r2(S.rc_raw["test"][m], S.truth["test"][m]) for m in range(8))
    print(f"ridge (reference): min R^2 over conditions {ridge_r2:.3f}")
    for model, steps in (("corrector", None), ("bridge", 1)):
        net, info = R.train(model, HP, S, seed=0)
        P = R.predict(model, net, S, "test", P_IN, steps=steps or 16).cpu().numpy()
        worst = min(r2(P[m], S.truth["test"][m]) for m in range(8))
        good = worst >= 0.9
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} G1a {model}{' one step' if steps else ''}: min R^2 {worst:.3f} "
              f"(epochs {info['epochs_run']})", flush=True)
    net, info = R.train("flow", HP, S, seed=0)
    X = R.predict("flow", net, S, "test", P_IN, token=R.NULL, n_samples=R.K_SAMPLES).cpu().numpy()
    Pm = R.predict("flow", net, S, "test", P_IN, n_samples=16).cpu().numpy()
    print(f"flow: point R^2 (mean of 16 samples, token) min {min(r2(Pm[m], S.truth['test'][m]) for m in range(8)):.3f} "
          f"(epochs {info['epochs_run']})")
    dist = R.distribution_metrics(X, S.raw["test"])
    gd = R.distribution_metrics(R.gauss_samples(S, "test"), S.raw["test"])
    pv = X.var(axis=0, ddof=1).sum(1)
    sc = float(np.corrcoef(pv, S.spread["test"])[0, 1])
    good = sc >= 0.5 and dist["energy"] < gd["energy"]
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} G1c flow: spread corr with truth {sc:.3f} (>= 0.5); energy {dist['energy']:.4f} "
          f"vs Gaussian {gd['energy']:.4f}; set top-1 {dist['set_top1']:.3f} vs {gd['set_top1']:.3f}", flush=True)
    Ssh = Synth(device, shuffle_sc=True)
    snet, _ = R.train("flow", HP, Ssh, seed=0)
    Ps = R.predict("flow", snet, Ssh, "test", P_IN, n_samples=16).cpu().numpy()
    dr = float(np.mean([code_dr(Ps[m], Ssh.truth["test"][m]) for m in range(8)]))
    good = dr <= 0.02
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} G1b shuffled flow: demeaned r with the true means {dr:.4f} (<= 0.02)", flush=True)
    print("ALL_OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
