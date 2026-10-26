"""Spec v4 E2.1 addendum: can un-anchored flows (from noise straight to the encoding; linear or nonlinear velocity)
learn on the E2.1-shaped simulation (checks/synthetic_e21.py)? Reported, not a gate. For each flow: point R^2 of the
sample means against the true conditional means (worst condition), energy distance vs the linear-Gaussian, spread
correlation with the true spread, and a shuffled-input null.

    python scripts/experiments/joint_flow_matching/checks/synthetic_e21_addendum.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e21_run as R  # noqa: E402
from synthetic_e21 import HP, P_IN, Synth, r2  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, Ssh = Synth(device), Synth(device, shuffle_sc=True)
    gd = R.distribution_metrics(R.gauss_samples(S, "test"), S.raw["test"])
    print(f"ridge point R^2 (worst condition) {min(r2(S.rc_raw['test'][m], S.truth['test'][m]) for m in range(8)):.3f}; "
          f"Gaussian energy {gd['energy']:.4f}, set top-1 {gd['set_top1']:.3f}")
    for model in R.FREE:
        net, info = R.train(model, HP, S, seed=0)
        P = R.predict(model, net, S, "test", P_IN, n_samples=R.FLOW_POINT_SAMPLES).cpu().numpy()
        X = R.predict(model, net, S, "test", P_IN, token=R.NULL, n_samples=R.K_SAMPLES).cpu().numpy()
        d = R.distribution_metrics(X, S.raw["test"])
        sc = float(np.corrcoef(X.var(axis=0, ddof=1).sum(1), S.spread["test"])[0, 1])
        snet, _ = R.train(model, HP, Ssh, seed=0)
        Xs = R.predict(model, snet, Ssh, "test", P_IN, token=R.NULL, n_samples=R.K_SAMPLES).cpu().numpy()
        ds = R.distribution_metrics(Xs, Ssh.raw["test"])
        print(f"{model}: point R^2 worst {min(r2(P[m], S.truth['test'][m]) for m in range(8)):.3f}; energy {d['energy']:.4f} "
              f"(Gaussian {gd['energy']:.4f}); spread r {sc:.3f}; set top-1 {d['set_top1']:.3f}; shuffled set rank "
              f"{ds['set_avg_rank']:.3f}; epochs {info['epochs_run']}", flush=True)


if __name__ == "__main__":
    main()
