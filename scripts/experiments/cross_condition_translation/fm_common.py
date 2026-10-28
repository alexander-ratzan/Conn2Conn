"""Spec v4 E4 flows (E4.2-E4.4): conditional flows from noise for every condition -> condition cell, with SC.

State: the target condition's k pooled block FC PCA codes (D2), divided by one train SD. Conditioning (DeepSets form,
D8): each available source is encoded by its condition's own linear encoder into a shared latent and mean-pooled (first
pass: at most one source; none -> a learned vector); SC codes (projected; zeros + flag when dropped); the target
condition through target-specific weights (E4.2: shared weights cannot express 56 different condition -> condition maps).
  FM-nl  velocity = per-target head(FiLM(target)-modulated residual MLP([z, tau-embedding, source latent, SC latent, flag]))
  FM-lin velocity = sum_j b_j(tau) (A_j z + B_j [z_src, SC, flag] + C_j[m_src] + D_j[m_tgt]) with 8 learned time
         curves b(tau); linear in state and conditioning at every tau, so it is exactly a linear-Gaussian model
Training examples (pairwise, D8): every train subject with all 8 targets per epoch; per example the source is none
with probability 0.2, else a random other condition; SC dropped with probability 0.2 when a source is present. Flow
matching from N(0, I): z_tau = (1 - tau) eps + tau y, target velocity y - eps. Point prediction = mean of samples at
starting-noise temperature T (selected on val average rank, D9).
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e4_common import CFG, FC, SC_ONLY, SeedData, energy_metrics, ridge_gcv, score
from models.architectures.crossmodal_flow_matching import ResidualBlock, SinusoidalEmbedding, integrate
from models.eval.metrics import compute_corr_matrix, corr_avg_rank

NONE = len(FC)                       # source index for "no FC input"
P_NONE, P_SC_DROP = 0.2, 0.2
FLOW_STEPS = 32
TRAIN = dict(batch_subjects=64, max_epochs=3000, min_epochs=300, patience=500, grad_clip=1.0)   # E4.2: 50 / 200 stopped FM-lin too early


# ------------------------------------------------------------------------------------------------ data
class FlowData:
    """Codes of one seed at code size k (torch, scaled): fc[split] (8, n, k), sc[split] (n, 512)."""

    def __init__(self, seed, k, device):
        sd = SeedData(seed, "main")
        self.sd, self.k, self.device, self.seed = sd, k, device, seed
        fc = {s: np.stack([sd.code(v, s, k) for v in FC]).astype(np.float64) for s in ("train", "val", "test")}
        sc = {s: sd.code("SC", s, max(CFG["k_sc"])).astype(np.float64) for s in fc}
        self.s_fc, self.s_sc = float(fc["train"].std()), float(sc["train"].std())
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.fc = {s: t(fc[s] / self.s_fc) for s in fc}
        self.sc = {s: t(sc[s] / self.s_sc) for s in sc}
        self.raw = fc
        self.n = {s: fc[s].shape[1] for s in fc}

    def sample_batch(self, subj, gen):
        """All 8 targets for `subj`, a random source per target (none with P_NONE), SC dropped with P_SC_DROP."""
        B = len(subj)
        tgt = torch.arange(8, device=self.device).repeat_interleave(B)
        s = subj.repeat(8)
        r = torch.randint(0, 7, (len(s),), device=self.device, generator=gen)
        src = r + (r >= tgt).long()                                     # a random condition other than the target
        none = torch.rand(len(s), device=self.device, generator=gen) < P_NONE
        src = torch.where(none, torch.full_like(src, NONE), src)
        sc_on = ~((torch.rand(len(s), device=self.device, generator=gen) < P_SC_DROP) & ~none)
        return self.inputs("train", s, src, tgt, sc_on) + (self.fc["train"][tgt, s],)

    def inputs(self, split, s, src, tgt, sc_on):
        z_src = torch.where((src == NONE)[:, None], torch.zeros(len(s), self.k, device=self.device),
                            self.fc[split][src.clamp(max=7), s])
        return z_src, src, tgt, self.sc[split][s], sc_on


# ------------------------------------------------------------------------------------------------ models
def by_index(x, idx, W, b=None):
    """Row-wise x @ W[idx] (+ b[idx]) without materialising W[idx] per row (loops over the distinct indices)."""
    out = x.new_zeros(x.shape[0], W.shape[-1])
    for m in torch.unique(idx).tolist():
        sel = idx == m
        out[sel] = x[sel] @ W[m] + (b[m] if b is not None else 0)
    return out


class SourceEncoder(nn.Module):
    """DeepSets source side: h = E_{m_src} z_src + e_{m_src}, a linear encoder per source condition into a shared
    latent, mean-pooled over the available sources (first pass: <= 1); no source -> a learned vector."""

    def __init__(self, k, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(len(FC), k, dim) / math.sqrt(k))
        self.b = nn.Parameter(torch.zeros(len(FC), dim))
        self.none = nn.Parameter(torch.zeros(dim))

    def forward(self, z_src, src):
        h = by_index(z_src, src.clamp(max=len(FC) - 1), self.W, self.b)
        return torch.where((src == NONE)[:, None], self.none.expand_as(h), h)


class FiLMBlock(nn.Module):
    """Residual block whose hidden units are scaled and shifted per target condition (starts as the plain block)."""

    def __init__(self, width, dropout):
        super().__init__()
        self.block = ResidualBlock(width, dropout)
        self.gamma = nn.Parameter(torch.zeros(len(FC), width))
        self.beta = nn.Parameter(torch.zeros(len(FC), width))

    def forward(self, h, tgt):
        h = self.block(h)
        return h * (1 + self.gamma[tgt]) + self.beta[tgt]


class FMnl(nn.Module):
    """Velocity = per-target head(FiLM-modulated residual MLP([z, tau-embedding, pooled source latent, SC latent, flag]))."""

    def __init__(self, k, k_sc, width=256, depth=1, dropout=0.1, time_dim=64, **_):
        super().__init__()
        self.k_sc = k_sc
        self.source = SourceEncoder(k, width)
        self.sc_proj = nn.Linear(k_sc, width)
        self.time = SinusoidalEmbedding(time_dim)
        self.inp = nn.Linear(k + time_dim + 2 * width + 1, width)
        self.blocks = nn.ModuleList([FiLMBlock(width, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head_W = nn.Parameter(torch.zeros(len(FC), width, k))
        self.head_b = nn.Parameter(torch.zeros(len(FC), k))

    def forward(self, z, tau, z_src, src, tgt, sc, sc_on):
        f = sc_on.float()[:, None]
        x = torch.cat([z, self.time(tau), self.source(z_src, src), self.sc_proj(sc[:, :self.k_sc]) * f, f], dim=1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, tgt)
        return by_index(self.norm(h), tgt, self.head_W, self.head_b)


class FMlin(nn.Module):
    """Velocity linear in z and the conditioning for every (source, target) pair and flow time:
        v = D_tgt (g(tau) * [E_src z_src, S z_sc * flag, flag]) + sum_j b_j(tau) (A_j z + u_j[tgt] * z + c_j[tgt, src])
    E_src: per-source encoders into a rank-r latent (CCA-like), D_tgt: per-target decoders, g and b: learned curves of
    tau only. Exactly a linear-Gaussian model per cell (E4.2: a shared map across cells could not express 56 maps)."""

    def __init__(self, k, k_sc, rank=128, nb=8, time_dim=64, **_):
        super().__init__()
        self.k_sc, c = k_sc, 2 * rank + 1
        self.source = SourceEncoder(k, rank)
        self.sc_proj = nn.Linear(k_sc, rank, bias=False)
        self.curves = nn.Sequential(SinusoidalEmbedding(time_dim), nn.Linear(time_dim, 64), nn.SiLU(), nn.Linear(64, nb + c))
        self.nb = nb
        self.D = nn.Parameter(torch.zeros(len(FC), c, k))
        self.A = nn.Parameter(torch.zeros(nb, k, k))
        self.u = nn.Parameter(torch.zeros(nb, len(FC), k))
        self.cb = nn.Parameter(torch.zeros(nb, len(FC), len(FC) + 1, k))

    def forward(self, z, tau, z_src, src, tgt, sc, sc_on):
        f = sc_on.float()[:, None]
        hc = torch.cat([self.source(z_src, src), self.sc_proj(sc[:, :self.k_sc]) * f, f], dim=1)
        cur = self.curves(tau)
        b, g = cur[:, :self.nb], cur[:, self.nb:]
        cond = by_index(g * hc, tgt, self.D)
        state = (b[:, :, None] * (torch.einsum("nd,jde->nje", z, self.A) + self.u[:, tgt].transpose(0, 1) * z[:, None]
                                  + self.cb[:, tgt, src].transpose(0, 1))).sum(1)
        return cond + state


def make(model, hp, k):
    if model == "fm_lin":
        return FMlin(k, hp["k_sc"], rank=hp.get("rank", 128))
    return FMnl(k, hp["k_sc"], width=hp["width"], depth=hp["depth"], dropout=hp["dropout"])


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def ridge_gcv_np(X, Y):
    """E2.0's ridge (GCV over the config grid) for the synthetic checks."""
    return ridge_gcv(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), [float(a) for a in CFG["ridge_alphas"]])


# ------------------------------------------------------------------------------------------------ training
def fm_loss(net, batch, gen):
    z_src, src, tgt, sc, sc_on, y = batch
    tau = torch.rand(len(y), device=y.device, generator=gen)
    eps = torch.randn(y.shape, device=y.device, generator=gen)
    zt = (1 - tau[:, None]) * eps + tau[:, None] * y
    return F.mse_loss(net(zt, tau, z_src, src, tgt, sc, sc_on), y - eps)


def val_batch(data, split="val"):
    """Fixed evaluation set: every subject x target x (no source, each other source) x (SC on / off where a source)."""
    out = {"z_src": [], "src": [], "tgt": [], "sc": [], "sc_on": [], "y": []}
    n = data.n[split]
    s = torch.arange(n, device=data.device)
    for t in range(8):
        for a in list(range(8)) + [NONE]:
            if a == t:
                continue
            for on in ([True] if a == NONE else [False, True]):
                src = torch.full((n,), a, device=data.device)
                tgt = torch.full((n,), t, device=data.device)
                z_src, src, tgt, sc, sc_on = data.inputs(split, s, src, tgt, torch.full((n,), on, device=data.device))
                for k_, v in zip(("z_src", "src", "tgt", "sc", "sc_on"), (z_src, src, tgt, sc, sc_on)):
                    out[k_].append(v)
                out["y"].append(data.fc[split][t])
    return tuple(torch.cat(out[k_]) for k_ in ("z_src", "src", "tgt", "sc", "sc_on", "y"))


def train(model, hp, data, seed, trial=None, max_epochs=None, log_every=0, subset=None, regularise=True, early_stop=True,
          restore_best=True):
    torch.manual_seed(seed)
    net = make(model, hp, data.k).to(data.device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"] if regularise else 0.0)
    max_ep = max_epochs or TRAIN["max_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    gen = torch.Generator(device=data.device).manual_seed(seed)
    subj_all = torch.arange(data.n["train"], device=data.device) if subset is None else subset
    vb = val_batch(data)
    best, state, best_ep, bad, hist = np.inf, None, 0, 0, []
    for ep in range(max_ep):
        net.train()
        order = subj_all[torch.randperm(len(subj_all), device=data.device, generator=gen)]
        tl = []
        for i in range(0, len(order), TRAIN["batch_subjects"]):
            loss = fm_loss(net, data.sample_batch(order[i:i + TRAIN["batch_subjects"]], gen), gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), TRAIN["grad_clip"])
            if not torch.isfinite(gn):
                raise FloatingPointError(f"non-finite gradient norm at epoch {ep}")
            opt.step()
            tl.append(loss.item())
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = fm_loss(net, vb, torch.Generator(device=data.device).manual_seed(999)).item()
        ema = vl if ep == 0 else 0.9 * ema + 0.1 * vl
        hist.append((ep, float(np.mean(tl)), vl))
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep}: train {np.mean(tl):.4f} val {vl:.4f}", flush=True)
        if ep >= TRAIN["min_epochs"] - 1:
            if ema < best:
                best, best_ep, bad, state = ema, ep, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
        if trial is not None and ep >= TRAIN["min_epochs"]:
            trial.report(-ema, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if early_stop and subset is None and bad >= TRAIN["patience"]:
            break
    if restore_best and state is not None:
        net.load_state_dict(state)
    return net, dict(best_val_loss=best, best_epoch=best_ep, epochs_run=ep + 1, history=hist)


# ------------------------------------------------------------------------------------------------ prediction / scoring
@torch.no_grad()
def sample(net, data, split, src, tgt, sc_on, n_samples, temperature, seed=0, chunk=4096):
    """(n_samples, n, k) generated target codes (scaled units) for every subject of `split`."""
    net.eval()
    n = data.n[split]
    s = torch.arange(n, device=data.device)
    z_src, src_t, tgt_t, sc, on = data.inputs(split, s, torch.full((n,), src, device=data.device),
                                              torch.full((n,), tgt, device=data.device), torch.full((n,), sc_on, device=data.device))
    g = torch.Generator(device=data.device).manual_seed(seed)
    rep = lambda x: x.repeat(n_samples, *([1] * (x.dim() - 1)))
    Zs, Ss, Ts, SCs, Os = rep(z_src), rep(src_t), rep(tgt_t), rep(sc), rep(on)
    out = []
    for i in range(0, len(Zs), chunk):
        sl = slice(i, i + chunk)
        eps = temperature * torch.randn((len(Zs[sl]), data.k), device=data.device, generator=g)
        out.append(integrate(lambda x, t: net(x, t, Zs[sl], Ss[sl], Ts[sl], SCs[sl], Os[sl]), eps, n_steps=FLOW_STEPS))
    return torch.cat(out).view(n_samples, n, data.k)


def eval_cells():
    """(source index or NONE, target index, sc_on) for every evaluated single-source / SC-only cell."""
    out = []
    for t in range(8):
        out.append((NONE, t, True))
        for a in range(8):
            if a != t:
                out += [(a, t, False), (a, t, True)]
    return out


def decode_scores(data, split, t, codes_scaled):
    v = FC[t]
    return score(data.sd.decode(v, codes_scaled * data.s_fc), data.sd.edges[(split, v)], data.sd.train_mean[v])


def val_avg_rank(net, data, temperature, n_samples=32, cells=None):
    """Mean val average rank (decoded edges) over the evaluated cells: the selection objective (D9)."""
    vals = []
    for a, t, on in (cells or eval_cells()):
        P = sample(net, data, "val", a, t, on, n_samples, temperature).mean(0).cpu().numpy() * data.s_fc
        v = FC[t]
        cc = compute_corr_matrix(np.asarray(data.sd.edges[("val", v)], dtype=np.float64), data.sd.decode(v, P).astype(np.float64))
        vals.append(float(corr_avg_rank(cc=cc)))
    return float(np.mean(vals))


def evaluate(net, data, temperature, model, n_point=256, n_dist=64):
    rows, dist = [], []
    for split in ("val", "test"):
        for a, t, on in eval_cells():
            src = SC_ONLY if a == NONE else FC[a]
            X = sample(net, data, split, a, t, on, max(n_point, n_dist), temperature, seed=1).cpu().numpy()
            m = decode_scores(data, split, t, X[:n_point].mean(0))
            rows.append(dict(model=model, source=src, target=FC[t], with_sc=bool(on), split=split, **m))
            dist.append(dict(model=model, source=src, target=FC[t], with_sc=bool(on), split=split,
                             **energy_metrics(X[:n_dist] * data.s_fc, data.raw[split][t])))
    return rows, dist
