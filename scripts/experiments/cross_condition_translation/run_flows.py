"""Spec v4 E4.3-E4.4: flow search (seed 0) and final runs (seeds 0-4) for FM-nl and FM-lin at the fixed code size k.

search  one worker of a shared Optuna study (journal file; several GPU jobs run workers in parallel): each trial trains on
        seed 0 (pruned on the smoothed val flow loss), then picks the starting-noise temperature T in {0.25, 0.5, 1} and
        scores the trial by mean val average rank over every single-source and SC-only cell (D9)
final   retrain the study's best settings on one seed (train seed = seed x 1000), re-pick T on that seed's val split,
        evaluate val and test (point: mean of 256 samples, decoded edges; distribution: 64 samples, code space), save
        the record (results/flows/k<k>/<model>_seed<S>.json) and the weights (results/cache/experiments/...)

    python scripts/experiments/cross_condition_translation/run_flows.py search --model fm_nl --k 256 --trials 8 --worker 0
    python scripts/experiments/cross_condition_translation/run_flows.py final --model fm_nl --k 256 --seed 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from fm_common import FlowData, evaluate, n_params, train, val_avg_rank
from e4_common import HERE, REPO_ROOT

TEMPS = (0.25, 0.5, 1.0)
CACHE = REPO_ROOT / "results" / "cache" / "experiments" / "cross_condition_translation"
SPACE = dict(lr=(1e-4, 3e-3), wd=(1e-6, 1e-1), width=[128, 256, 512], depth=[1, 2, 3], dropout=(0.0, 0.3), emb_dim=[16, 32],
             k_sc=[64, 128, 256, 512])


def sample_hp(trial, model):
    hp = dict(lr=trial.suggest_float("lr", *SPACE["lr"], log=True), wd=trial.suggest_float("wd", *SPACE["wd"], log=True),
              k_sc=trial.suggest_categorical("k_sc", SPACE["k_sc"]))
    if model == "fm_nl":
        hp.update(width=trial.suggest_categorical("width", SPACE["width"]), depth=trial.suggest_categorical("depth", SPACE["depth"]),
                  dropout=trial.suggest_float("dropout", *SPACE["dropout"]), emb_dim=trial.suggest_categorical("emb_dim", SPACE["emb_dim"]))
    return hp


def pick_temperature(net, data):
    scores = {T: val_avg_rank(net, data, T) for T in TEMPS}
    T = max(scores, key=scores.get)
    return T, scores


def study(model, k):
    path = CACHE / "optuna" / f"{model}_k{k}.journal"
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(path)))
    return optuna.create_study(study_name=f"{model}_k{k}", storage=storage, direction="maximize", load_if_exists=True,
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50))


def search(model, k, trials, worker, device):
    data = FlowData(0, k, device)
    st = study(model, k)
    st.sampler = optuna.samplers.TPESampler(seed=1000 + worker)

    def objective(trial):
        hp = sample_hp(trial, model)
        t0 = time.time()
        net, info = train(model, hp, data, 0 * 1000 + trial.number, trial=trial)
        T, scores = pick_temperature(net, data)
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("temperature", T)
        trial.set_user_attr("val_avg_rank_by_T", {str(t): v for t, v in scores.items()})
        trial.set_user_attr("epochs_run", info["epochs_run"])
        trial.set_user_attr("best_epoch", info["best_epoch"])
        trial.set_user_attr("n_params", n_params(net))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: val avg rank {scores[T]:.4f} at T {T} ({time.time() - t0:.0f} s, epochs "
              f"{info['epochs_run']}, best {info['best_epoch']}, {n_params(net):,} params)", flush=True)
        return scores[T]

    st.optimize(objective, n_trials=trials)


def final(model, k, seed, device):
    st = study(model, k)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(done, key=lambda t: t.value)
    hp = best.user_attrs["hp"]
    data = FlowData(seed, k, device)
    t0 = time.time()
    net, info = train(model, hp, data, seed * 1000)
    T, scores = pick_temperature(net, data)
    rows, dist = evaluate(net, data, T, model)
    hist = info.pop("history")
    rec = dict(model=model, k=k, seed=seed, hp=hp, search_best_trial=best.number, search_best_value=best.value,
               n_trials_complete=len(done), temperature=T, val_avg_rank_by_T={str(t): v for t, v in scores.items()},
               n_params=n_params(net), train=info, history=hist[::10], rows=rows, dist=dist, seconds=time.time() - t0,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    out = HERE / "results" / "flows" / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model}_seed{seed}.json").write_text(json.dumps(rec, default=float))
    wdir = CACHE / "weights" / f"k{k}"
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), hp=hp, model=model, k=k, temperature=T), wdir / f"{model}_seed{seed}.pt")
    print(f"{model} k {k} seed {seed}: T {T}, val avg rank {scores[T]:.4f}, epochs {info['epochs_run']} (best "
          f"{info['best_epoch']}), {n_params(net):,} params, {(time.time() - t0) / 60:.1f} min -> {out.name}/{model}_seed{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "final"])
    ap.add_argument("--model", required=True, choices=["fm_nl", "fm_lin"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mode == "search":
        search(a.model, a.k, a.trials, a.worker, device)
    else:
        final(a.model, a.k, a.seed, device)


if __name__ == "__main__":
    main()
