"""Spec v4 E2.1: retrain the seed-0 un-anchored flows at their selected settings and log learning, for both routes.

Routes: `cca` (SC PCA -> CCA task latent -> block PCA -> FC edges) and `pca` (SC PCA -> block PCA codes -> FC edges).
Models: `flow_free` (nonlinear velocity) and `flow_linear` (linear velocity), both from noise with no ridge anchor.
Hyperparameters come from the E2.1 records (`records/e21/<model>_seed<S>[_pca].json`). Logged per epoch: train loss
(mean over batches), val loss (raw and smoothed, as used for early stopping); every `--every` epochs: val decoded
demeaned r (mean of 16 samples, mean over the 8 conditions) and val energy distance to each subject's 8 encodings.
Writes `cache/e21/learning/<model>_<target>_seed<S>.npz` (curves) and `.pt` (final weights, best smoothed val loss).

    python scripts/experiments/joint_flow_matching/flow_learning.py --seed 0 [--every 10]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e21_run as R  # noqa: E402


def run(model, target, seed, every, device):
    sfx = "" if target == "cca" else f"_{target}"
    hp = json.loads((R.OUT / f"{model}_seed{seed}{sfx}.json").read_text())["hp"]
    data = R.Data(seed, device, target=target)
    torch.manual_seed(seed)
    net = R.make(model, hp, data.d).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=R.TRAIN["max_epochs"])
    n = data.sc["train"].shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    log = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_loss_smooth")}
    evals = {k: [] for k in ("epoch", "val_dr", "val_energy")}
    best, state, bad, ema = np.inf, None, 0, None
    t0 = time.time()
    for ep in range(R.TRAIN["max_epochs"]):
        net.train()
        order = torch.randperm(n, device=device, generator=gen)
        losses = []
        for s in range(0, n, R.TRAIN["batch_subjects"]):
            z0, y, c, m = data.batch("train", order[s:s + R.TRAIN["batch_subjects"]], hp["k_sc"])
            loss = R.loss_fn(model, net, hp, z0, y, c, m, data, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), R.TRAIN["grad_clip"])
            opt.step()
            losses.append(loss.item())
        sched.step()
        net.eval()
        with torch.no_grad():
            nv = data.sc["val"].shape[0]
            z0, y, c, m = data.batch("val", torch.arange(nv, device=device), hp["k_sc"])
            vl = R.loss_fn(model, net, hp, z0, y, c, m, data, torch.Generator(device=device).manual_seed(999)).item()
        ema = vl if ema is None else 0.9 * ema + 0.1 * vl
        for k, v in zip(log, (ep, float(np.mean(losses)), vl, ema)):
            log[k].append(v)
        if ep % every == 0:
            P = R.predict(model, net, data, "val", hp["k_sc"], n_samples=16)
            X = R.predict(model, net, data, "val", hp["k_sc"], token=R.NULL, n_samples=32).cpu().numpy()
            evals["epoch"].append(ep)
            evals["val_dr"].append(float(np.mean([data.val_dr(i, P[i]) for i in range(8)])))
            evals["val_energy"].append(float(np.mean(R.energy_per_subject(X, data.raw["val"]))))
        if ema < best:
            best, bad, state = ema, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= R.TRAIN["patience_flow"]:
            break
    out = R.CACHE / "learning"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{model}_{target}_seed{seed}.npz", **{f"log_{k}": np.array(v) for k, v in log.items()},
             **{f"eval_{k}": np.array(v) for k, v in evals.items()}, best_epoch=np.argmin(log["val_loss_smooth"]))
    torch.save(dict(state=state, hp=hp, model=model, target=target), out / f"{model}_{target}_seed{seed}.pt")
    print(f"{model} {target}: {ep + 1} epochs, best smoothed val loss at epoch {int(np.argmin(log['val_loss_smooth']))}, "
          f"last val dr {evals['val_dr'][-1]:.4f}, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for target in ("cca", "pca"):
        for model in ("flow_free", "flow_linear"):
            run(model, target, args.seed, args.every, device)


if __name__ == "__main__":
    main()
