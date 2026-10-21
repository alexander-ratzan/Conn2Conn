"""Spec v4 E1 arms A2 (per-pair MLP), A3 (joint regression with view tokens) and A4 (joint paired bridge).

One job = one arm x seed (x pair for A2): an Optuna TPE search (`search.n_trials`), each trial trained with early
stopping on the val code loss; the trial with the best mean val demeaned r (decoded edges, all pairs) is kept and its
weights are scored once on val and test. A4 is selected at RK4 `flow.tune_steps` and scored at every `flow.eval_steps`
(1 = one Euler step). With `--fixed` the search is skipped and the given configuration is trained once (E1.5 arms,
E1.7, N3).

Writes `records_dir/<arm>/seed<S>_<tag>.json` (config, all trials, per-pair val / test metrics) and the predicted
val / test codes to `cache_dir/preds/<arm>_seed<S>_<tag>.npz`.

    python scripts/experiments/joint_flow_matching/tune.py --arm A3 --seed 0
    python scripts/experiments/joint_flow_matching/tune.py --arm A2 --seed 0 --pair rest SC
    python scripts/experiments/joint_flow_matching/tune.py --arm A4 --seed 0 --fixed <record.json> --shuffle-pairs --tag n3
"""
import argparse
import copy
import json
import socket
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F

from common import CFG, REPO_ROOT, SeedData, demeaned_r_torch, pairs, score
from models.architectures.crossmodal_flow_matching import TokenBackbone

TR, SR, FL = CFG["train"], CFG["search"], CFG["flow"]
SR = dict(SR, lr=[float(x) for x in SR["lr"]], weight_decay=[float(x) for x in SR["weight_decay"]],
          dropout=[float(x) for x in SR["dropout"]])


class Data:
    """Codes of the job's views on the device, the pair list and the val edges for selection."""

    def __init__(self, sd, views, pair_list, device, shuffle_pairs=False):
        self.sd, self.views, self.pairs, self.device = sd, views, pair_list, device
        self.vid = {v: i for i, v in enumerate(views)}
        self.P = torch.tensor([[self.vid[a], self.vid[b]] for a, b in pair_list], device=device)
        # one shared constant (sqrt(n_train)) puts codes at unit mean squared norm per subject without changing the
        # equal-variance balance between views; predictions are divided by it before decoding
        n = len(sd.splits["train"])
        self.code_scale = float(np.sqrt(n))
        self.codes = {split: self.code_scale * torch.stack([torch.as_tensor(sd.code(v, split, sd.cs.k_max))
                                                            for v in views]).to(device)
                      for split in ("train", "val", "test")}                     # (n_views, n_split, k_max)
        g = torch.Generator().manual_seed(1234 + sd.seed)
        self.target_perm = torch.randperm(n, generator=g).to(device) if shuffle_pairs else torch.arange(n, device=device)
        self.val_edges = {v: torch.as_tensor(sd.edges[("val", v)], device=device) for v in views}
        self.val_mu = {v: torch.as_tensor(sd.train_mean[v], device=device) for v in views}

    def batch(self, split, subj, pidx, k):
        a, b = self.P[pidx, 0], self.P[pidx, 1]
        tgt_subj = self.target_perm[subj] if split == "train" else subj
        return self.codes[split][a, subj, :k], self.codes[split][b, tgt_subj, :k], a, b


def make_model(arm, hp, n_views):
    return TokenBackbone(k=hp["k"], n_views=n_views, token_dim=hp["token_dim"], width=hp["width"], depth=hp["depth"],
                         dropout=hp["dropout"], use_tokens=(arm != "A2"), use_time=(arm == "A4"))


def loss_fn(arm, model, za, zb, a, b, gen=None):
    if arm == "A4":
        tau = torch.rand(za.shape[0], device=za.device, generator=gen)
        zt = (1 - tau[:, None]) * za + tau[:, None] * zb
        return F.mse_loss(model(zt, a, b, tau), zb - za)
    return F.mse_loss(model.predict_regression(za, a, b), zb)


@torch.no_grad()
def predict(arm, model, data, split, k, pidx, n_steps=None):
    """Predicted codes (cache units) for every subject of `split` for pair index `pidx`."""
    model.eval()
    n = data.codes[split].shape[1]
    subj = torch.arange(n, device=data.device)
    za, _, a, b = data.batch(split, subj, torch.full((n,), pidx, device=data.device), k)
    if arm == "A4":
        steps = n_steps or FL["tune_steps"]
        z = model.predict_flow(za, a, b, n_steps=steps, method="euler" if steps == 1 else "rk4")
    else:
        z = model.predict_regression(za, a, b)
    return z / data.code_scale


@torch.no_grad()
def val_code_loss(arm, model, data, k):
    model.eval()
    n, npair = data.codes["val"].shape[1], len(data.pairs)
    subj = torch.arange(n, device=data.device).repeat(npair)
    pidx = torch.arange(npair, device=data.device).repeat_interleave(n)
    za, zb, a, b = data.batch("val", subj, pidx, k)
    return loss_fn(arm, model, za, zb, a, b, gen=torch.Generator(device=data.device).manual_seed(0)).item()


@torch.no_grad()
def val_mean_dr(arm, model, data, k, n_steps=None):
    out = []
    for p, (_, tgt) in enumerate(data.pairs):
        z = predict(arm, model, data, "val", k, p, n_steps)
        out.append(demeaned_r_torch(data.sd.cs.decode_torch(tgt, z, data.device), data.val_edges[tgt], data.val_mu[tgt]))
    return float(np.mean(out))


def train(arm, hp, data, seed, trial=None, train_cfg=TR):
    torch.manual_seed(seed)
    k = hp["k"]
    model = make_model(arm, hp, len(data.views)).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg["max_epochs"])
    n, npair = data.codes["train"].shape[1], len(data.pairs)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    gen = torch.Generator(device=data.device).manual_seed(seed)
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        order = torch.randperm(n * npair, device=data.device, generator=gen)
        for s in range(0, len(order), train_cfg["batch_size"]):
            idx = order[s:s + train_cfg["batch_size"]]
            za, zb, a, b = data.batch("train", idx % n, idx // n, k)
            loss = loss_fn(arm, model, za, zb, a, b, gen=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step()
        sched.step()
        vl = val_code_loss(arm, model, data, k)
        if vl < best:
            best, best_epoch, bad = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if trial is not None:
            trial.report(-vl, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= train_cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, dict(best_val_loss=best, best_epoch=best_epoch, epochs_run=epoch + 1)


def sample_hp(trial):
    return dict(lr=trial.suggest_float("lr", *SR["lr"], log=True),
                weight_decay=trial.suggest_float("weight_decay", *SR["weight_decay"], log=True),
                width=trial.suggest_categorical("width", SR["width"]),
                depth=trial.suggest_categorical("depth", SR["depth"]),
                dropout=trial.suggest_float("dropout", *SR["dropout"]),
                token_dim=trial.suggest_categorical("token_dim", SR["token_dim"]),
                k=trial.suggest_categorical("k", SR["k"]))


def evaluate(arm, model, data, k, sd):
    """Full v3:E0 metrics per pair on val and test; predicted codes for saving."""
    steps_list = FL["eval_steps"] if arm == "A4" else [None]
    per_pair, codes = [], {}
    for steps in steps_list:
        for p, (src, tgt) in enumerate(data.pairs):
            for split in ("val", "test"):
                z = predict(arm, model, data, split, k, p, steps).cpu().numpy()
                codes[f"{split}__{src}__{tgt}__{steps or 0}"] = z
                m = score(sd.decode(tgt, z), sd.edges[(split, tgt)], sd.train_mean[tgt])
                per_pair.append(dict(source=src, target=tgt, split=split, steps=steps or 0, **m))
    return per_pair, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A2", "A3", "A4"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="main", help="code-space cache variant (main | per_view)")
    ap.add_argument("--views", default="views", help="config key of the view list (views | session_views)")
    ap.add_argument("--pair", nargs=2, help="A2 only: source target")
    ap.add_argument("--n-trials", type=int, default=SR["n_trials"])
    ap.add_argument("--fixed", help="record JSON whose selected hyperparameters are trained once (no search)")
    ap.add_argument("--shuffle-pairs", action="store_true", help="N3: train targets from permuted subjects")
    ap.add_argument("--tag", default="", help="suffix for output names")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = SeedData(args.seed, args.variant)
    views = CFG[args.views]
    if args.arm == "A2":
        assert args.pair, "--pair is required for A2"
        pair_list, views = [tuple(args.pair)], list(args.pair)
    else:
        pair_list = pairs(views, CFG["nested"])
    data = Data(sd, views, pair_list, device, shuffle_pairs=args.shuffle_pairs)
    tag = "_".join(x for x in [args.variant, args.views if args.views != "views" else "",
                                "-".join(args.pair) if args.pair else "", args.tag] if x)
    train_cfg = dict(TR, **CFG.get("a2_train", {})) if args.arm == "A2" else TR
    t0 = time.time()

    trials, best = [], {"value": -np.inf}
    def objective(trial, hp=None):
        hp = hp or sample_hp(trial)
        ts = time.time()
        model, info = train(args.arm, hp, data, seed=args.seed * 1000 + (trial.number if trial else 0),
                            trial=trial, train_cfg=train_cfg)
        val = val_mean_dr(args.arm, model, data, hp["k"])
        trials.append(dict(number=trial.number if trial else 0, hp=hp, val_mean_dr=val, seconds=time.time() - ts, **info))
        print(f"trial {trials[-1]['number']}: val dr {val:.4f} ({trials[-1]['seconds']:.0f} s, "
              f"epochs {info['epochs_run']}) {hp}", flush=True)
        if val > best["value"]:
            best.update(value=val, hp=hp, state={k: v.cpu() for k, v in model.state_dict().items()})
        return val

    if args.fixed:
        objective(None, hp=json.loads(Path(args.fixed).read_text())["selected"]["hp"])
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=SR["prune_from_epoch"])
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=args.n_trials)

    hp = best["hp"]
    model = make_model(args.arm, hp, len(views)).to(device)
    model.load_state_dict(best["state"])
    per_pair, codes = evaluate(args.arm, model, data, hp["k"], sd)

    rec = dict(arm=args.arm, seed=args.seed, variant=args.variant, views=views, pairs=pair_list, tag=tag,
               shuffle_pairs=args.shuffle_pairs, fixed=args.fixed,
               selected=dict(hp=hp, val_mean_dr=best["value"]), trials=trials, per_pair=per_pair,
               seconds_total=time.time() - t0, host=socket.gethostname(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    rec_path = REPO_ROOT / CFG["records_dir"] / args.arm / f"seed{args.seed}_{tag}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    pred_path = Path(CFG["cache_dir"]) / "preds" / f"{args.arm}_seed{args.seed}_{tag}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pred_path, **codes)
    print(f"wrote {rec_path} and {pred_path}; total {rec['seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
