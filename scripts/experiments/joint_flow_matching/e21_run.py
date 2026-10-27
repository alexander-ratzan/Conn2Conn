"""Spec v4 E2.1: SC -> each condition's latent encoding (point and conditional-generative models), one seed x model.

Models (all share `TokenBackbone`: state z = 32-d latent, cond = SC codes, token = condition 0-7, 8 = no condition):
  corrector     z_hat = ridge(SC) + f(ridge(SC), SC, m)            squared error
  corrector_id  the same + w * demeaned corr-eye per condition       selected on val average rank
  bridge        deterministic flow from ridge(SC) to the encoding; flow time 0 with prob. p0, else uniform
  flow          residual conditional flow: from noise to g - ridge(SC) (ridge mean added back), conditioned on SC and
                the condition token (dropped -> 8 for 20 % of examples); the subject's cloud mixes the 8 conditions as
                the Gaussian reference does; early stopping on a smoothed val loss (patience 200)
                (also trains `flow_shuffled`: the selected configuration with SC permuted across train subjects for
                the whole model, ridge means included)
  latent2sc     (H7) latent -> SC codes: ridge(latent) + f(.), token = input (0 consensus, 1-7 tasks)
References evaluated in every job: ridge_cond (per-condition ridge, from the cache) and gauss (ridge_cond means + the
shared residual covariance). Each job: Optuna search (n_trials), early stopping on val loss, selection on val, then val
and test evaluation; writes records/e21/<model>_seed<S>.json.

    python scripts/experiments/joint_flow_matching/e21_run.py --model corrector --seed 0 [--n-trials 32]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e21_common import CACHE, FC, Latent, crossfit, ridge_fit  # noqa: E402
from common import REPO_ROOT, score  # noqa: E402
from models.architectures.crossmodal_flow_matching import TokenBackbone  # noqa: E402
from models.eval.metrics import distance_avg_rank, distance_top1_accuracy  # noqa: E402
from models.train.loss import compute_correye_dm_loss  # noqa: E402

OUT = REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching" / "results" / "Glasser" / "records" / "e21"
NULL = 8
PRED = CACHE / "preds"
K_SAMPLES = 64
FLOW_STEPS = 32
TEMPERATURE = {"cca": 1.0, "pca": 0.25}   # scale of the flows' starting noise per route (user 2026-10-27: block PCA 0.25)
FLOW_POINT_SAMPLES = 256      # flow point estimate = ridge mean + mean of 256 residual samples (16 was too noisy; G3 2026-10-26)
TRAIN = dict(batch_subjects=64, max_epochs=1000, max_epochs_flow=3000, min_epochs=50, patience=100, patience_flow=200,
             grad_clip=1.0, prune_from=50)   # flows (user 2026-10-27): cap 3000; no stopping or best-weight pick before epoch 50
SPACE = dict(lr=(1e-4, 3e-3), wd=(1e-6, 1e-1), width=[128, 256, 512], depth=[1, 2, 3], dropout=(0.0, 0.3),
             token_dim=[8, 16, 32], k_sc=[64, 128, 256, 512], p0=[0.0, 0.25, 0.5], id_w=[0.1, 0.3, 1.0])


# ----------------------------------------------------------------------------------------------- data
class Data:
    """Tensors for one seed. Latents and SC codes are divided by one scalar each (unit train SD) for training;
    predictions are returned in latent units."""

    def __init__(self, seed, device, shuffle_sc=False, refit_ridge=True, target="cca"):
        self.suffix = "" if target == "cca" else f"_{target}"
        self.target = target
        self.temperature = TEMPERATURE[target]
        z = np.load(CACHE / f"seed{seed}{self.suffix}.npz")
        self.seed, self.device = seed, device
        self.L = Latent.__new__(Latent)          # light handle for decoding / scoring (no refit)
        from common import SeedData
        self.L.sd = SeedData(seed, "main")
        self.L.D = {v: z["D"][i] for i, v in enumerate(FC)}
        self.d = int(z["latent"][1])
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.s_lat = float(z["enc_train"].std())
        self.s_sc = float(z["sc_train"].std())
        self.enc = {s: t(z[f"enc_{s}"] / self.s_lat) for s in ("train", "val", "test")}     # (8, n, d)
        self.rc = {s: t(z[f"rc_{s}"] / self.s_lat) for s in ("train", "val", "test")}
        self.sc = {s: t(z[f"sc_{s}"] / self.s_sc) for s in ("train", "val", "test")}
        self.raw = {s: z[f"enc_{s}"] for s in ("train", "val", "test")}
        self.rc_raw = {s: z[f"rc_{s}"] for s in ("train", "val", "test")}
        self.cov = z["gauss_cov"]
        n = self.sc["train"].shape[0]
        self.perm = (torch.randperm(n, generator=torch.Generator().manual_seed(4321 + seed)).to(device)
                     if shuffle_sc else torch.arange(n, device=device))
        if shuffle_sc and refit_ridge:   # full null: refit the ridge means on permuted pairs too (spread-only null: keep them)
            pnp = self.perm.cpu().numpy()
            for m, k in enumerate(z["rc_k"]):
                xtr = z["sc_train"][pnp, :k]
                f = ridge_fit(xtr, z["enc_train"][m])
                for split in ("val", "test"):
                    self.rc_raw[split][m] = f(z[f"sc_{split}"][:, :k])
                self.rc_raw["train"][m] = crossfit(xtr, z["enc_train"][m], seed=seed)
            self.rc = {s: t(self.rc_raw[s] / self.s_lat) for s in ("train", "val", "test")}
        # fast decoding for val selection: latent (raw units) -> demeaned FC edges, per condition
        cs = self.L.sd.cs
        self.dec, self.mu, self.val_edges = [], [], []
        for v in FC:
            W = cs.scale[v] * (torch.as_tensor(z["D"][FC.index(v)], device=device, dtype=torch.float32)
                               @ torch.as_tensor(cs.basis(v)[:, :z["D"].shape[2]].T.copy(), device=device))
            self.dec.append(W)
            self.mu.append(torch.as_tensor(cs.mean[v], device=device))
            self.val_edges.append(torch.as_tensor(self.L.sd.edges[("val", v)], device=device))
        self.train_mean = [torch.as_tensor(self.L.sd.train_mean[v], device=device) for v in FC]

    def batch(self, split, subj, k_sc):
        """All 8 conditions for `subj`: z0 (ridge), target, cond, token; flattened to (8 * len(subj), .)."""
        m = torch.arange(8, device=self.device).repeat_interleave(len(subj))
        s = subj.repeat(8)
        cond_subj = self.perm[s] if split == "train" else s
        return (self.rc[split][m, s], self.enc[split][m, s], self.sc[split][cond_subj, :k_sc], m)

    # val metrics on decoded FC edges (torch)
    def edges(self, m, g_raw):
        return self.mu[m] + g_raw @ self.dec[m]

    def val_dr(self, m, g_raw):
        a = self.edges(m, g_raw) - self.train_mean[m]
        b = self.val_edges[m] - self.train_mean[m]
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-10)).mean().item()

    def val_rank(self, m, g_raw):
        a, b = self.edges(m, g_raw), self.val_edges[m]           # raw correlation matrix, rows = targets (as v3:E0)
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        cc = (b / b.norm(dim=1, keepdim=True)) @ (a / a.norm(dim=1, keepdim=True)).T
        own = cc.diag()[:, None]
        return (1 - (cc > own).float().sum(1) / cc.shape[0]).mean().item()


# ----------------------------------------------------------------------------------------------- models
FREE = ("flow_free", "flow_linear")          # E2.1 addendum: un-anchored flows from noise straight to the encoding
FLOWLIKE = ("flow",) + FREE


def base_model(model):
    """Model whose sampling rule applies (nulls share their parent's)."""
    return model.replace("_shuffled", "").replace("_spread_null", "")


class LinearFlowVelocity(torch.nn.Module):
    """v(z, tau, SC, m) = sum_j phi_j(tau) (z A_j + SC B_j + C_j[m]): linear in the state, the SC codes and the condition,
    with coefficients that vary over flow time through a small basis phi (polynomial + rational terms, the form of the
    exact velocity for linear-Gaussian data). Starts at zero."""

    def __init__(self, d, cond_dim, n_tokens=9):
        super().__init__()
        nb = 6
        self.A = torch.nn.Parameter(torch.zeros(nb, d, d))
        self.B = torch.nn.Parameter(torch.zeros(nb, cond_dim, d))
        self.C = torch.nn.Parameter(torch.zeros(nb, n_tokens, d))

    @staticmethod
    def basis(tau):
        r = 1.0 / (1.05 - tau)
        return torch.stack([torch.ones_like(tau), tau, tau ** 2, tau ** 3, 0.05 * r, 0.05 * tau * r], dim=1)

    def forward(self, z, src=None, tgt=None, tau=None, cond=None):
        phi = self.basis(tau)                                                    # (n, nb)
        terms = torch.einsum("nd,jde->nje", z, self.A) + torch.einsum("nc,jce->nje", cond, self.B) + self.C[:, tgt].transpose(0, 1)
        return (phi[:, :, None] * terms).sum(1)

    def predict_flow(self, z, src=None, tgt=None, n_steps=16, method="rk4", cond=None):
        from models.architectures.crossmodal_flow_matching import integrate
        return integrate(lambda x, t: self(x, src, tgt, t, cond=cond), z, n_steps=n_steps, method=method)


def make(model, hp, d):
    if model == "flow_linear":
        return LinearFlowVelocity(d, hp["k_sc"])
    return TokenBackbone(k=d, n_views=9, token_dim=hp["token_dim"], width=hp["width"], depth=hp["depth"],
                         dropout=hp["dropout"], use_tokens=True, use_time=model in ("bridge",) + FLOWLIKE,
                         cond_dim=hp["k_sc"])


def loss_fn(model, net, hp, z0, y, c, m, data, gen):
    if model in ("corrector", "corrector_id"):
        pred = net.predict_regression(z0, m, m, cond=c)
        loss = F.mse_loss(pred, y)
        if model == "corrector_id":
            P, Y = pred.view(8, -1, pred.shape[1]), y.view(8, -1, y.shape[1])
            loss = loss + hp["id_w"] * torch.stack([compute_correye_dm_loss(P[i], Y[i], data.enc["train"][i].mean(0))
                                                    for i in range(8)]).mean()
        return loss
    tau = torch.rand(len(y), device=y.device, generator=gen)
    if model == "bridge":
        tau = torch.where(torch.rand(len(y), device=y.device, generator=gen) < hp["p0"], torch.zeros_like(tau), tau)
        start = z0
    else:                                  # flows from noise; the residual flow generates g - ridge(SC), the free flows g
        if model == "flow":
            y = y - z0
        start = torch.randn(y.shape, device=y.device, generator=gen)
        m = torch.where(torch.rand(len(y), device=y.device, generator=gen) < 0.2, torch.full_like(m, NULL), m)
    zt = (1 - tau[:, None]) * start + tau[:, None] * y
    return F.mse_loss(net(zt, m, m, tau, cond=c), y - start)


@torch.no_grad()
def predict(model, net, data, split, k_sc, steps=16, token=None, n_samples=1, seed=0):
    """Point predictions (8, n, d) in latent units; flows: mean of n_samples. token=NULL -> unconditional samples
    (returned as (n_samples, n, d) per call with token NULL)."""
    net.eval()
    n = data.sc[split].shape[0]
    subj = torch.arange(n, device=data.device)
    z0, _, c, m = data.batch(split, subj, k_sc)
    if model in ("corrector", "corrector_id"):
        out = net.predict_regression(z0, m, m, cond=c)
    elif model == "bridge":
        out = net.predict_flow(z0, m, m, n_steps=steps, method="euler" if steps == 1 else "rk4", cond=c)
    else:
        g = torch.Generator(device=data.device).manual_seed(seed)
        if token == NULL:
            # the subject's cloud, built as the Gaussian reference's: each sample picks a condition, draws that
            # condition's residual from the flow and adds that condition's ridge mean
            subj_k = subj.repeat(n_samples)
            tok = torch.randint(0, 8, (len(subj_k),), device=data.device, generator=g)
            cc = data.sc[split][subj_k, :k_sc]
            eps = getattr(data, "temperature", 1.0) * torch.randn((len(cc), data.d), device=data.device, generator=g)
            res = net.predict_flow(eps, tok, tok, n_steps=FLOW_STEPS, cond=cc)
            anchor = data.rc[split][tok, subj_k] if model == "flow" else 0
            return (anchor + res).view(n_samples, n, data.d) * data.s_lat
        acc = 0
        for _ in range(n_samples):
            eps = getattr(data, "temperature", 1.0) * torch.randn(z0.shape, device=data.device, generator=g)
            acc = acc + net.predict_flow(eps, m, m, n_steps=FLOW_STEPS, cond=c)
        out = (z0 if model == "flow" else 0) + acc / n_samples
    return out.view(8, n, data.d) * data.s_lat


def val_objective(model, net, data, hp):
    if model in FLOWLIKE:
        X = predict(model, net, data, "val", hp["k_sc"], token=NULL, n_samples=32)
        return -float(np.mean(energy_per_subject(X.cpu().numpy(), data.raw["val"])))
    P = predict(model, net, data, "val", hp["k_sc"], steps=16)
    f = data.val_rank if model == "corrector_id" else data.val_dr
    return float(np.mean([f(i, P[i]) for i in range(8)]))


def train(model, hp, data, seed, trial=None):
    torch.manual_seed(seed)
    net = make(model, hp, data.d).to(data.device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    max_ep = TRAIN["max_epochs_flow"] if model in FLOWLIKE else TRAIN["max_epochs"]
    min_ep = TRAIN["min_epochs"] if model in FLOWLIKE else 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    n = data.sc["train"].shape[0]
    gen = torch.Generator(device=data.device).manual_seed(seed)
    vgen_seed = 999
    best, state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(max_ep):
        net.train()
        order = torch.randperm(n, device=data.device, generator=gen)
        for s in range(0, n, TRAIN["batch_subjects"]):
            z0, y, c, m = data.batch("train", order[s:s + TRAIN["batch_subjects"]], hp["k_sc"])
            loss = loss_fn(model, net, hp, z0, y, c, m, data, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), TRAIN["grad_clip"])
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            nv = data.sc["val"].shape[0]
            z0, y, c, m = data.batch("val", torch.arange(nv, device=data.device), hp["k_sc"])
            vl = loss_fn(model, net, hp, z0, y, c, m, data,
                         torch.Generator(device=data.device).manual_seed(vgen_seed)).item()
        if model in FLOWLIKE:               # a flow's val loss is noise-dominated: early-stop on a smoothed value
            ema = vl if ep == 0 else 0.9 * ema + 0.1 * vl
            vl = ema
        if ep < min_ep - 1:                 # flows: no best-weight pick or stopping before min_epochs
            pass
        elif vl < best:
            best, best_ep, bad, state = vl, ep, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if trial is not None and ep >= TRAIN["prune_from"]:
            trial.report(-vl, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= (TRAIN["patience_flow"] if model in FLOWLIKE else TRAIN["patience"]):
            break
    net.load_state_dict(state)
    return net, dict(best_val_loss=best, best_epoch=best_ep, epochs_run=ep + 1)


def sample_hp(trial, model):
    if model == "flow_linear":
        return dict(lr=trial.suggest_float("lr", *SPACE["lr"], log=True), wd=trial.suggest_float("wd", *SPACE["wd"], log=True),
                    k_sc=trial.suggest_categorical("k_sc", SPACE["k_sc"]), width=0, depth=0, dropout=0.0, token_dim=0)
    hp = dict(lr=trial.suggest_float("lr", *SPACE["lr"], log=True), wd=trial.suggest_float("wd", *SPACE["wd"], log=True),
              width=trial.suggest_categorical("width", SPACE["width"]),
              depth=trial.suggest_categorical("depth", SPACE["depth"]),
              dropout=trial.suggest_float("dropout", *SPACE["dropout"]),
              token_dim=trial.suggest_categorical("token_dim", SPACE["token_dim"]),
              k_sc=trial.suggest_categorical("k_sc", SPACE["k_sc"]))
    if model == "bridge":
        hp["p0"] = trial.suggest_categorical("p0", SPACE["p0"])
    if model == "corrector_id":
        hp["id_w"] = trial.suggest_categorical("id_w", SPACE["id_w"])
    return hp


# ----------------------------------------------------------------------------------------------- distribution metrics
def _pd(A, B):
    return torch.cdist(torch.as_tensor(A, dtype=torch.float64), torch.as_tensor(B, dtype=torch.float64)).numpy()


def energy_per_subject(X, Y):
    """X: (K, n, d) samples; Y: (8, n, d) observed. Energy distance per subject."""
    out = []
    for i in range(X.shape[1]):
        x, y = X[:, i], Y[:, i]
        out.append(2 * _pd(x, y).mean() - _pd(x, x).sum() / (len(x) * (len(x) - 1)) - _pd(y, y).sum() / (len(y) * (len(y) - 1)))
    return np.array(out)


def distribution_metrics(X, Y):
    """Energy distance, set-to-set identification, spread calibration and coverage."""
    K, n, d = X.shape
    ed = energy_per_subject(X, Y)
    xin = np.array([_pd(X[:, i], X[:, i]).sum() / (K * (K - 1)) for i in range(n)])
    yin = np.array([_pd(Y[:, i], Y[:, i]).sum() / (Y.shape[0] * (Y.shape[0] - 1)) for i in range(n)])
    Xf = X.transpose(1, 0, 2).reshape(n * K, d)
    Yf = Y.transpose(1, 0, 2).reshape(n * Y.shape[0], d)
    cross = _pd(Yf, Xf).reshape(n, Y.shape[0], n, K).mean(axis=(1, 3))      # [observed j, generated i]
    M = 2 * cross - yin[:, None] - xin[None, :]
    pv = X.var(axis=0, ddof=1).sum(1)
    ov = Y.var(axis=0, ddof=1).sum(1)
    mean = X.mean(0)
    dx = np.linalg.norm(X - mean[None], axis=2)
    q90 = np.quantile(dx, 0.9, axis=0)
    cover = (np.linalg.norm(Y - mean[None], axis=2) <= q90[None]).mean()
    return dict(energy=float(ed.mean()), set_top1=float(distance_top1_accuracy(M)), set_avg_rank=float(distance_avg_rank(M)),
                spread_corr=float(np.corrcoef(pv, ov)[0, 1]), spread_ratio=float(pv.mean() / ov.mean()),
                coverage90=float(cover),
                per_subject=dict(pred_spread=pv.tolist(), obs_spread=ov.tolist(), energy=ed.tolist()),
                example=dict(generated=X[:16, :6, :2].tolist(), observed=Y[:, :6, :2].tolist()))   # 6 subjects, dims 1-2


def gauss_samples(data, split, K=K_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    means = data.rc_raw[split]                                               # (8, n, d)
    n, d = means.shape[1], means.shape[2]
    pick = rng.integers(0, 8, size=(K, n))
    noise = rng.multivariate_normal(np.zeros(d), data.cov, size=(K, n))
    return means[pick, np.arange(n)[None, :]] + noise


# ----------------------------------------------------------------------------------------------- evaluation
def point_rows(name, data, P, split, extra=None):
    rows = []
    for i, v in enumerate(FC):
        g = P[i] if isinstance(P, np.ndarray) else P[i].cpu().numpy()
        m = score(data.L.decode(i, g), data.L.sd.edges[(split, v)], data.L.sd.train_mean[v])
        rows.append(dict(model=name, condition=v, split=split, **(extra or {}), **m))
    return rows


def evaluate(model, net, data, hp):
    rows, dist = [], {}
    for split in ("val", "test"):
        if model in ("corrector", "corrector_id"):
            rows += point_rows(model, data, predict(model, net, data, split, hp["k_sc"]), split)
        elif model == "bridge":
            for st in (1, 4, 16, 64):
                rows += point_rows(model, data, predict(model, net, data, split, hp["k_sc"], steps=st), split, dict(steps=st))
        elif model.startswith("flow"):          # flow, flow_shuffled, flow_spread_null
            rows += point_rows(model, data, predict(base_model(model), net, data, split, hp["k_sc"], n_samples=FLOW_POINT_SAMPLES), split,
                               dict(note=f"mean of {FLOW_POINT_SAMPLES} samples with condition token"))
            X = predict(base_model(model), net, data, split, hp["k_sc"], token=NULL, n_samples=K_SAMPLES).cpu().numpy()
            dist[split] = distribution_metrics(X, data.raw[split])
    return rows, dist


@torch.no_grad()
def samples_for_condition(net, data, split, k_sc, m, K=K_SAMPLES, seed=0):
    """K residual-flow samples of condition m for every subject of `split`, in latent units: (K, n, d)."""
    net.eval()
    n = data.sc[split].shape[0]
    g = torch.Generator(device=data.device).manual_seed(seed)
    subj = torch.arange(n, device=data.device).repeat(K)
    tok = torch.full((len(subj),), m, device=data.device)
    eps = torch.randn((len(subj), data.d), device=data.device, generator=g)
    res = net.predict_flow(eps, tok, tok, n_steps=FLOW_STEPS, cond=data.sc[split][subj, :k_sc])
    return ((data.rc[split][m][subj] + res) * data.s_lat).view(K, n, data.d).cpu().numpy()


def retest_metrics(net, data, hp):
    """Rest test-retest calibration: the flow's predicted spread for rest vs each subject's observed difference
    between rest session 1 and 2 in the latent (variance estimate ||s1 - s2||^2 / 2, summed over dimensions)."""
    path = CACHE / f"seed{data.seed}_sessions.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    out = {}
    for split in ("val", "test"):
        obs = np.square(z[f"rest_S1_{split}"] - z[f"rest_S2_{split}"]).sum(1) / 2
        X = samples_for_condition(net, data, split, hp["k_sc"], FC.index("rest"))
        pred = X.var(axis=0, ddof=1).sum(1)
        out[split] = dict(retest_corr=float(np.corrcoef(pred, obs)[0, 1]), ratio_flow=float(pred.mean() / obs.mean()),
                          ratio_gauss=float(np.trace(data.cov) / obs.mean()))
    return out


def reference_rows(data):
    rows, dist = [], {}
    for split in ("val", "test"):
        rows += point_rows("ridge_cond", data, data.rc_raw[split], split)
        dist[split] = distribution_metrics(gauss_samples(data, split), data.raw[split])
    return rows, dist


# ----------------------------------------------------------------------------------------------- latent -> SC (H7)
def run_latent2sc(seed, data, n_trials, device):
    """Inputs: consensus (token 0) and each task's encoding (tokens 1-7); target SC codes (64). Ridge reference and
    a corrector (ridge + MLP) shared over inputs."""
    sd, ksc = data.L.sd, 64
    enc = {s: data.raw[s] for s in data.raw}
    inp = {s: np.concatenate([enc[s].mean(0)[None], enc[s][1:]]) for s in enc}          # (8 inputs, n, d)
    Y = {s: sd.code("SC", s, ksc).astype(np.float64) for s in enc}
    rows, rc = [], {}
    names = ["consensus"] + [f"{t} encoding" for t in FC[1:]]
    for j in range(8):
        f = ridge_fit(inp["train"][j], Y["train"])
        rc[j] = {"train": crossfit(inp["train"][j], Y["train"], seed=seed), "val": f(inp["val"][j]), "test": f(inp["test"][j])}
        for split in ("val", "test"):
            rows.append(dict(model="ridge_latent2sc", input=names[j], split=split,
                             **score(sd.decode("SC", rc[j][split]), sd.edges[(split, "SC")], sd.train_mean["SC"])))
    sy = float(Y["train"].std()); sx = float(inp["train"].std())
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Z0 = {s: T(np.stack([rc[j][s] for j in range(8)]) / sy) for s in enc}
    X = {s: T(inp[s] / sx) for s in enc}
    Yt = {s: T(Y[s] / sy) for s in enc}
    mu_sc = torch.as_tensor(sd.train_mean["SC"], device=device)
    W = torch.as_tensor((sd.cs.scale["SC"] * sd.cs.basis("SC")[:, :ksc]).T.copy(), device=device)
    vedges = torch.as_tensor(sd.edges[("val", "SC")], device=device)

    def batch(split, subj):
        mm = torch.arange(8, device=device).repeat_interleave(len(subj)); ss = subj.repeat(8)
        return Z0[split][mm, ss], Yt[split][ss], X[split][mm, ss], mm      # one SC target per subject

    def fit(hp, tseed, trial=None):
        torch.manual_seed(tseed)
        net = TokenBackbone(k=ksc, n_views=8, token_dim=hp["token_dim"], width=hp["width"], depth=hp["depth"],
                            dropout=hp["dropout"], cond_dim=data.d).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        n = Yt["train"].shape[0]; g = torch.Generator(device=device).manual_seed(tseed)
        best, state, bad = np.inf, None, 0
        for ep in range(TRAIN["max_epochs"]):
            net.train()
            order = torch.randperm(n, device=device, generator=g)
            for s in range(0, n, TRAIN["batch_subjects"]):
                z0, y, c, mm = batch("train", order[s:s + TRAIN["batch_subjects"]])
                loss = F.mse_loss(net.predict_regression(z0, mm, mm, cond=c), y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                z0, y, c, mm = batch("val", torch.arange(Yt["val"].shape[0], device=device))
                vl = F.mse_loss(net.predict_regression(z0, mm, mm, cond=c), y).item()
            if vl < best:
                best, bad, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if bad >= TRAIN["patience"]:
                break
        net.load_state_dict(state)
        return net

    @torch.no_grad()
    def pred(net, split):
        n = Yt[split].shape[0]
        z0, _, c, mm = batch(split, torch.arange(n, device=device))
        return (net.predict_regression(z0, mm, mm, cond=c) * sy).view(8, n, ksc)

    def objective(trial):
        hp = dict(lr=trial.suggest_float("lr", *SPACE["lr"], log=True), wd=trial.suggest_float("wd", *SPACE["wd"], log=True),
                  width=trial.suggest_categorical("width", SPACE["width"]), depth=trial.suggest_categorical("depth", SPACE["depth"]),
                  dropout=trial.suggest_float("dropout", *SPACE["dropout"]),
                  token_dim=trial.suggest_categorical("token_dim", SPACE["token_dim"]))
        net = fit(hp, seed * 1000 + trial.number)
        P = pred(net, "val")
        vals = []
        for j in range(8):
            a = mu_sc + P[j] @ W - torch.as_tensor(sd.train_mean["SC"], device=device)
            b = vedges - torch.as_tensor(sd.train_mean["SC"], device=device)
            a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
            vals.append(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean().item())
        trial.set_user_attr("hp", hp)
        return float(np.mean(vals))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    hp = study.best_trial.user_attrs["hp"]
    net = fit(hp, seed * 1000 + study.best_trial.number)
    for split in ("val", "test"):
        P = pred(net, split).cpu().numpy()
        for j in range(8):
            rows.append(dict(model="latent2sc", input=names[j], split=split,
                             **score(sd.decode("SC", P[j]), sd.edges[(split, "SC")], sd.train_mean["SC"])))
    return rows, hp, [dict(number=t.number, value=t.value) for t in study.trials]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["corrector", "corrector_id", "bridge", "flow", "flow_free", "flow_linear", "latent2sc", "refs"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=32)
    ap.add_argument("--target", choices=["cca", "pca"], default="cca", help="pca: block FC PCA targets (second addendum)")
    ap.add_argument("--retrain-selected", action="store_true",
                    help="no search: retrain the configuration selected in this model/seed/target's record (current TRAIN "
                         "settings) and write records / weights with the --tag suffix")
    ap.add_argument("--tag", default="_long")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training of the main model: load weights saved with --weights-tag, evaluate with the current "
                         "sampling settings (temperature), retrain the null, write records with --tag")
    ap.add_argument("--weights-tag", default="_long")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data = Data(args.seed, device, target=args.target)
    rec = dict(model=args.model, seed=args.seed, n_trials=args.n_trials, target=args.target,
               temperature=data.temperature,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    tag = ""
    if args.model == "refs":
        rec["rows"], rec["dist"] = reference_rows(data)
    elif args.model == "latent2sc":
        rec["rows"], rec["hp"], rec["trials"] = run_latent2sc(args.seed, data, args.n_trials, device)
    else:
        trials, best = [], {"value": -np.inf}
        tag = args.tag if (args.retrain_selected or args.eval_only) else ""

        def objective(trial):
            hp = sample_hp(trial, args.model)
            ts = time.time()
            net, info = train(args.model, hp, data, args.seed * 1000 + trial.number, trial)
            val = val_objective(args.model, net, data, hp)
            trials.append(dict(number=trial.number, hp=hp, value=val, seconds=time.time() - ts, **info))
            print(f"trial {trial.number}: val {val:.4f} ({time.time() - ts:.0f} s, epochs {info['epochs_run']})", flush=True)
            if val > best["value"]:
                best.update(value=val, hp=hp, state={k: v.cpu() for k, v in net.state_dict().items()})
            return val

        if args.eval_only:
            ck = torch.load(CACHE / "models" / f"{args.model}_seed{args.seed}{data.suffix}{args.weights_tag}.pt",
                            map_location="cpu")
            net = make(args.model, ck["hp"], data.d).to(device)
            net.load_state_dict(ck["state"])
            val = val_objective(args.model, net, data, ck["hp"])
            prev = json.loads((OUT / f"{args.model}_seed{args.seed}{data.suffix}{args.weights_tag}.json").read_text())
            trials.extend(prev.get("trials", []))
            best.update(value=val, hp=ck["hp"], state=ck["state"])
            print(f"evaluated saved weights ({args.weights_tag}) at temperature {data.temperature}: val {val:.4f}", flush=True)
        elif args.retrain_selected:
            hp = json.loads((OUT / f"{args.model}_seed{args.seed}{data.suffix}.json").read_text())["hp"]
            ts = time.time()
            net, info = train(args.model, hp, data, args.seed * 1000)
            val = val_objective(args.model, net, data, hp)
            trials.append(dict(number=0, hp=hp, value=val, seconds=time.time() - ts, **info))
            best.update(value=val, hp=hp, state={k: v.cpu() for k, v in net.state_dict().items()})
            print(f"retrained selected: val {val:.4f} ({time.time() - ts:.0f} s, epochs {info['epochs_run']}, best epoch "
                  f"{info['best_epoch']})", flush=True)
        else:
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            study.optimize(objective, n_trials=args.n_trials)
        net = make(args.model, best["hp"], data.d).to(device)
        net.load_state_dict(best["state"])
        rec.update(hp=best["hp"], val_objective=best["value"], trials=trials)
        rec["rows"], rec["dist"] = evaluate(args.model, net, data, best["hp"])
        preds = {}                                        # point predictions (latent units) for gate G3 / analysis
        for split in ("val", "test"):
            kw = dict(n_samples=FLOW_POINT_SAMPLES) if args.model in FLOWLIKE else dict(steps=16)
            preds[split] = predict(args.model, net, data, split, best["hp"]["k_sc"], **kw).cpu().numpy()
        PRED.mkdir(parents=True, exist_ok=True)
        np.savez(PRED / f"{args.model}_seed{args.seed}{data.suffix}{tag}.npz", **preds)
        if args.model in FLOWLIKE:                       # weights, for the trajectory notebook
            (CACHE / "models").mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=best["state"], hp=best["hp"], model=args.model, target=args.target),
                       CACHE / "models" / f"{args.model}_seed{args.seed}{data.suffix}{tag}.pt")
        if args.model in FREE:                         # full null for the un-anchored flows
            sdata = Data(args.seed, device, shuffle_sc=True, target=args.target)
            snet, _ = train(args.model, best["hp"], sdata, args.seed * 1000 + 777)
            r, dst = evaluate(f"{args.model}_shuffled", snet, sdata, best["hp"])
            rec["rows"] += r
            rec["dist_shuffled"] = dst
        if args.model == "flow":
            rec["retest"] = retest_metrics(net, data, best["hp"])
            # full null: SC shuffled for the whole model (ridge means refit on permuted pairs)
            sdata = Data(args.seed, device, shuffle_sc=True)
            snet, _ = train("flow", best["hp"], sdata, args.seed * 1000 + 777)
            r, dst = evaluate("flow_shuffled", snet, sdata, best["hp"])
            rec["rows"] += r
            rec["dist_shuffled"] = dst
            # spread-only null: real ridge means, SC shuffled only for the residual flow
            pdata = Data(args.seed, device, shuffle_sc=True, refit_ridge=False)
            pnet, _ = train("flow", best["hp"], pdata, args.seed * 1000 + 778)
            _, dst = evaluate("flow_spread_null", pnet, pdata, best["hp"])
            rec["dist_spread_null"] = dst
            rec["retest_spread_null"] = retest_metrics(pnet, pdata, best["hp"])
    rec["seconds_total"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_seed{args.seed}{data.suffix}{tag if args.model not in ('refs', 'latent2sc') else ''}.json"
    (OUT / name).write_text(json.dumps(rec, indent=1, default=float))
    print(f"wrote {OUT / name} in {rec['seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
