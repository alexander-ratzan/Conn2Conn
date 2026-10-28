"""windowed_fc_gain G7b: E2.1's flows retrained on part-sample arms (one model x route x seed, all arms in one job).

Models (E2.1): `flow` (residual flow anchored on the ridge mean; cca route only, as E2.1), `flow_linear` and
`flow_free` (un-anchored linear / nonlinear flows; cca and pca routes). Each arm retrains the model at E2.1's selected
hyperparameters (its search record), with E2.1's TRAIN settings, on the arm's units (g7_common), and is evaluated by
E2.1's `evaluate` on the full val / test targets (point rows: mean of 256 samples; distribution: 64 samples). The `flow`
model is anchored on the arm's own ridge means (run_g7a.py cache).
- Equal steps: one epoch = n_train / 64 optimizer steps in every arm (each step sees proportionally more units).
- Token arms: the condition token becomes m + 9 * type (full 0, half 1, quarter 2; null 8 + 9 * type), 27 tokens;
  evaluation uses type full, i.e. E2.1's tokens.
- No nulls (E2.1 has them for the `none` setting).

    python scripts/experiments/windowed_fc_gain/run_g7b.py --model flow_linear --route pca --seed 0
        [--arms none aug_halves ...] [--out-tag halves]   -> results/g7/flow_linear_pca_seed0[_<tag>].json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g7_common import ARMS, G7_CACHE, ROUTES  # noqa: E402
import e21_run as E  # noqa: E402

OUT = HERE / "results" / "g7"
N_TYPES = 3


class ArmData(E.Data):
    """E2.1's Data with the arm's training units (and, for the residual flow, the arm's ridge anchors)."""

    def set_arm(self, seed, route, arm, model):
        z = np.load(G7_CACHE / f"closed_seed{seed}_{route}_{arm}.npz")
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        if not hasattr(self, "_sc_train0"):
            self._sc_train0 = self.sc["train"]
        subj = torch.as_tensor(z["subj"], device=self.device)
        self.enc["train"] = t(z["tgt"] / self.s_lat)
        self.sc["train"] = self._sc_train0[subj]
        self.unit_type = torch.as_tensor(z["typ"], device=self.device)
        self.perm = torch.arange(len(subj), device=self.device)
        self.token_on = arm.startswith("token")
        self.rc["train"] = t(z["rc_train"] / self.s_lat)      # per unit (batches index units); used by `flow` only
        if model == "flow":                     # anchor = the arm's ridge means
            for s in ("val", "test"):
                self.rc[s] = t(z[f"rc_{s}"] / self.s_lat)
            self.rc_raw = {s: z[f"rc_{s}"] for s in ("val", "test")}
        return len(subj)

    def batch(self, split, subj, k_sc):
        z0, y, c, m = super().batch(split, subj, k_sc)
        if split == "train" and self.token_on:
            m = m + 9 * self.unit_type[subj.repeat(8)]
        return z0, y, c, m


def make_tokens(n_tokens):
    base = E.make

    def make(model, hp, d):
        if model == "flow_linear":
            return E.LinearFlowVelocity(d, hp["k_sc"], n_tokens=n_tokens)
        return E.TokenBackbone(k=d, n_views=n_tokens, token_dim=hp["token_dim"], width=hp["width"], depth=hp["depth"],
                               dropout=hp["dropout"], use_tokens=True, use_time=True, cond_dim=hp["k_sc"])
    return make if n_tokens != 9 else base


def flow_loss(model, net, hp, z0, y, c, m, data, gen):
    """E2.1's flow loss; the 20 % condition dropout keeps the sample type (null token 8 + 9 * type)."""
    tau = torch.rand(len(y), device=y.device, generator=gen)
    if model == "flow":
        y = y - z0
    start = torch.randn(y.shape, device=y.device, generator=gen)
    null = E.NULL + 9 * torch.div(m, 9, rounding_mode="floor")
    m = torch.where(torch.rand(len(y), device=y.device, generator=gen) < 0.2, null, m)
    zt = (1 - tau[:, None]) * start + tau[:, None] * y
    return F.mse_loss(net(zt, m, m, tau, cond=c), y - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["flow", "flow_linear", "flow_free"])
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out-tag", default="", help="results file suffix (one file per arm group)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = ROUTES[a.route]
    hp = json.loads((E.OUT / f"{a.model}_seed{a.seed}{suffix}.json").read_text())["hp"]
    E.loss_fn = flow_loss
    batch0 = E.TRAIN["batch_subjects"]
    rec = dict(model=a.model, route=a.route, seed=a.seed, hp=hp, train=dict(E.TRAIN),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", arms={})
    out = OUT / f"{a.model}_{a.route}_seed{a.seed}{'_' + a.out_tag if a.out_tag else ''}.json"
    if out.exists():                                   # resume: keep finished arms
        rec["arms"] = json.loads(out.read_text())["arms"]
    for arm in a.arms:
        if arm in rec["arms"]:
            continue
        ts = time.time()
        data = ArmData(a.seed, device, target=a.route)
        n_train = data.sc["train"].shape[0]
        n_units = data.set_arm(a.seed, a.route, arm, a.model)
        E.make = make_tokens(9 * N_TYPES if data.token_on else 9)
        E.TRAIN["batch_subjects"] = int(round(batch0 * n_units / n_train))       # equal optimizer steps per epoch
        net, info = E.train(a.model, hp, data, a.seed * 1000)
        val = E.val_objective(a.model, net, data, hp)
        rows, dist = E.evaluate(a.model, net, data, hp)
        E.TRAIN["batch_subjects"] = batch0
        rec["arms"][arm] = dict(n_units=n_units, batch=int(round(batch0 * n_units / n_train)), val_objective=val,
                                seconds=time.time() - ts, rows=rows,
                                dist={s: {k: v for k, v in d.items() if k not in ("per_subject", "example")} for s, d in dist.items()},
                                **info)
        print(f"{a.model} {a.route} seed {a.seed} {arm}: units {n_units}, epochs {info['epochs_run']} (best {info['best_epoch']}), "
              f"val {val:.4f}, test energy {dist['test']['energy']:.4f}, {time.time() - ts:.0f} s", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
