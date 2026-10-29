"""Spec v4 E5.2 (H3): composite loss on the CCA ridge decoders, one seed (GPU).

Every E5.1 CCA decoder (72 pairs; H2's CCA-agg and CCA-cat per held-out view and setting) is a linear map W from the
source encoding to the target's edges, pred = mean_b + g W. Its ridge solution minimises
    MSE + a / (n E) ||W||^2        (a = the E5.1 penalty; n train subjects, E edges),
and H3 adds the composite terms of grid v3's factorial (Var-match, Demeaned corr-eye, Neighbor dist at 0.5):
    L = MSE + a / (n E) ||W||^2 + sum_t w_t L_t / c_t,
with c_t = mean |L_t| / mean MSE measured on the ridge decoder over one epoch of train batches (the composite-loss
protocol's scaling). All 8 combinations are trained together as independent copies of W (Adam, batch 64, cosine
schedule, lr = lr_rel x RMS of the ridge weights), initialised at the ridge solution. Every `eval_every` epochs (and at
epoch 0, the ridge solution) the val average rank is computed; each combination keeps its best-val weights (D9).
G1: the MSE-only decoders reproduce the ridge decoders on test (mean |delta| < g1_tol in demeaned r and average rank),
and the ridge decoders reproduce E5.1's rows.

Writes OUT/h3/composite_seed<S>.json and, for the example seed, OUT/h3/examples_seed<S>.npz (test predictions into rest).
With --nparts P the 106 decoders are split round-robin into P jobs (--part p) writing composite_seed<S>_part<p>of<P>.json
(each with its own G1 over its decoders); `analyze.load_h3` merges the parts.

    python scripts/experiments/cca_analysis/run_composite.py --seed 0 [--part 0 --nparts 2]
"""
import argparse
import json
import time

import numpy as np
import torch

from e5_common import CFG, FC, OUT, PAIRS, SPLITS, VIEWS, load_views
from models.eval.metrics import compute_corr_matrix, corr_avg_rank, corr_topn_accuracy

T0 = time.time()
CC = CFG["composite"]
COMBOS = list(CC["combos"])
TERMS = ("varmatch", "correye_dm", "neidist")


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


def terms(P, y, mu):
    """Composite terms for C stacked predictions P (C, B, E) against y (B, E); each (C,). Same definitions as
    models/train/loss.py (compute_var_match_loss, compute_correye_dm_loss, compute_neidist_loss), batched over C."""
    tv = torch.mean((y - y.mean(0, keepdim=True)) ** 2)
    pv = torch.mean((P - P.mean(1, keepdim=True)) ** 2, dim=(1, 2))
    vm = ((tv - pv) / (tv + 1e-10)) ** 2

    def unit(x):
        x = x - x.mean(-1, keepdim=True)
        return x / torch.sqrt((x ** 2).sum(-1, keepdim=True) + 1e-10)

    cc = unit(y - mu)[None] @ unit(P - mu).transpose(1, 2)                       # (C, B, B), rows = true
    eye = torch.eye(y.shape[0], device=y.device)
    ce = torch.linalg.matrix_norm(cc - eye)
    d = torch.cdist(y[None].expand(P.shape[0], -1, -1).contiguous(), P)            # (C, B, B)
    dself = torch.diagonal(d, dim1=1, dim2=2).mean(1)
    dnei = d + eye * d.amax(dim=(1, 2), keepdim=True)
    dother = ((dnei.min(dim=1).values + dnei.min(dim=2).values) / 2).mean(1)
    return dict(varmatch=vm, correye_dm=ce, neidist=dself - dother)


def metrics(pred, true, mu):
    """E2's `score` quantities (Pearson r, demeaned r, average rank, top-1, MSE), torch on one device."""
    cc = compute_corr_matrix(true, pred)
    ccd = compute_corr_matrix(true - mu, pred - mu)
    return dict(pearson=float(torch.diagonal(cc).mean()), demeaned_pearson=float(torch.diagonal(ccd).mean()),
                avg_rank=float(corr_avg_rank(cc=cc)), top1_acc=float(corr_topn_accuracy(cc=cc, topn=1)),
                mse=float(torch.mean((pred - true) ** 2)))


def ridge_edges(X, Yc, a_rel):
    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    a = a_rel * torch.mean(s ** 2)
    return Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Yc)), float(a)


def train_head(X, Y, mu, a_rel, seed, dev):
    """X: {split: (n, d_in)} float64 tensors; Y: {split: (n, E)} float32. Returns per-combo results and state."""
    n, E = Y["train"].shape
    B0, a = ridge_edges(X["train"], (Y["train"] - mu).double(), a_rel)
    B0 = B0.float()
    Xf = {s: X[s].float() for s in X}
    W = torch.nn.Parameter(B0[None].repeat(len(COMBOS), 1, 1).contiguous())
    w = torch.tensor([[CC["combos"][c][t] for t in TERMS] for c in COMBOS], device=dev)          # (C, 3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs = CC["batch_size"]
    # c_t on the ridge decoder over one epoch of train batches
    acc = {t: [] for t in TERMS + ("mse",)}
    with torch.no_grad():
        for idx in torch.randperm(n, generator=gen).split(bs):
            if len(idx) < 2:
                continue
            P = mu + Xf["train"][idx] @ B0
            y = Y["train"][idx]
            acc["mse"].append(float(torch.mean((P - y) ** 2)))
            for t, v in terms(P[None], y, mu).items():
                acc[t].append(float(v.abs()[0]))
    c_t = torch.tensor([np.mean(acc[t]) / np.mean(acc["mse"]) for t in TERMS], device=dev)
    opt = torch.optim.Adam([W], lr=CC["lr_rel"] * float(B0.pow(2).mean().sqrt()))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CC["max_epochs"])
    pen = a / (n * E)

    def val_scores():
        with torch.no_grad():
            P = mu + Xf["val"][None] @ W                                                          # (C, n_val, E)
            return [float(corr_avg_rank(cc=compute_corr_matrix(Y["val"], P[c]))) for c in range(len(COMBOS))]

    best = val_scores()
    best_ep = [0] * len(COMBOS)
    state = W.detach().clone()
    curve = [[0] + best]
    for ep in range(1, CC["max_epochs"] + 1):
        for idx in torch.randperm(n, generator=gen).split(bs):
            if len(idx) < 2:
                continue
            y = Y["train"][idx]
            P = mu + Xf["train"][idx][None] @ W
            mse = torch.mean((P - y) ** 2, dim=(1, 2))
            tt = terms(P, y, mu)
            comp = sum(w[:, i] * tt[t] / c_t[i] for i, t in enumerate(TERMS))
            loss = (mse + pen * W.pow(2).sum(dim=(1, 2)) + comp).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if ep % CC["eval_every"] == 0:
            v = val_scores()
            curve.append([ep] + v)
            for c in range(len(COMBOS)):
                if v[c] > best[c]:
                    best[c], best_ep[c] = v[c], ep
                    state[c] = W.detach()[c]
    out = {}
    with torch.no_grad():
        for c, name in enumerate(COMBOS):
            P = mu + Xf["test"] @ state[c]
            out[name] = dict(best_epoch=best_ep[c], val_avg_rank=best[c], **metrics(P, Y["test"], mu))
        ridge = metrics(mu + Xf["test"] @ B0, Y["test"], mu)
    return out, ridge, dict(zip(TERMS, c_t.tolist())), curve, state, B0


def build_heads(meta):
    """Every CCA decoder, in a fixed order (the index seeds each decoder's training): (hyp, model, setting, source label,
    target, source views, ridge penalty)."""
    heads = [("H1", "cca_ridge", "pair", a, b, [a], meta["pairs"][f"{a}|{b}"]) for a, b in PAIRS]
    for b in VIEWS:
        for setting in ("other8", "other7"):
            if setting == "other7" and b == "SC":
                continue
            srcs = [v for v in (VIEWS if setting == "other8" else FC) if v != b]
            for model in ("cca_agg", "cca_cat"):
                heads.append(("H2", model, setting, "+".join(srcs), b, srcs, meta["h2"][f"{model}|{setting}|{b}"]))
    return heads


def head_inputs(head, enc):
    """Decoder input per split: the source encoding, or the sources' encodings averaged (cca_agg) / concatenated (cca_cat)."""
    model, srcs = head[1], head[5]
    return {s: (torch.cat([enc[s][a] for a in srcs], 1) if model == "cca_cat" else
                torch.stack([enc[s][a] for a in srcs]).mean(0)) for s in SPLITS}


def run(seed, part=0, nparts=1):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(OUT / "h3" / f"seed{seed}.npz")
    meta = json.loads(str(z["alpha_json"]))
    enc = {s: {v: torch.as_tensor(z[f"enc__{s}__{v}"], device=dev) for v in VIEWS} for s in SPLITS}
    data, splits, _ = load_views(seed)
    assert all((splits[s] == z[f"split__{s}"]).all() for s in SPLITS)
    Y = {v: {s: torch.as_tensor(data[v][splits[s]], device=dev) for s in SPLITS} for v in VIEWS}
    mu = {v: Y[v]["train"].mean(0) for v in VIEWS}
    del data
    log(f"seed {seed}: data on {dev}; CCA d {meta['d']} lambda {meta['lam']}")

    heads = [h for i, h in enumerate(build_heads(meta)) if i % nparts == part]
    rows, scales, curves, examples = [], {}, {}, {}
    ex = CC["examples"]
    for i, (hyp, model, setting, src, b, srcs, a_rel) in enumerate(heads):
        X = head_inputs(heads[i], enc)
        res, ridge, c_t, curve, state, B0 = train_head(X, Y[b], mu[b], a_rel, seed * 1000 + i, dev)
        key = f"{model}|{setting}|{src}|{b}"
        scales[key], curves[key] = c_t, curve
        rows.append(dict(seed=seed, hyp=hyp, model=model, setting=setting, source=src, target=b, combo="ridge",
                         best_epoch=0, **ridge))
        for combo, r in res.items():
            rows.append(dict(seed=seed, hyp=hyp, model=model, setting=setting, source=src, target=b, combo=combo, **r))
        if seed == ex["seed"] and b == "rest":
            sub = slice(0, ex["n_subjects"])
            with torch.no_grad():
                for c, combo in enumerate(COMBOS):
                    examples[f"{key}|{combo}"] = (mu[b] + X["test"][sub].float() @ state[c]).cpu().numpy().astype(np.float16)
        if i % 10 == 0:
            log(f"head {i + 1} / {len(heads)} {key}: ridge avg rank {ridge['avg_rank']:.4f}, "
                + ", ".join(f"{c} {r['avg_rank']:.4f}" for c, r in res.items()))

    # G1
    import pandas as pd
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["model", "setting", "source", "target"], columns="combo",
                         values=["demeaned_pearson", "avg_rank"])
    g1_dr = float((piv["demeaned_pearson"]["mse_only"] - piv["demeaned_pearson"]["ridge"]).abs().mean())
    g1_rank = float((piv["avg_rank"]["mse_only"] - piv["avg_rank"]["ridge"]).abs().mean())
    e51 = pd.DataFrame(json.loads((OUT / "results" / f"seed{seed}.json").read_text())["rows"])
    e51 = e51[(e51.hyp == "H1") & (e51.model == "cca_ridge") & (e51.rep == "CCA")].set_index(["source", "target"])
    mine = df[(df.hyp == "H1") & (df.combo == "ridge")].set_index(["source", "target"])
    consist = float((mine.demeaned_pearson - e51.demeaned_pearson.reindex(mine.index)).abs().max()) if len(mine) else 0.0
    tol = CFG["gates"]["g1_tol"]
    g1 = dict(mse_only_vs_ridge_dr=g1_dr, mse_only_vs_ridge_rank=g1_rank, ridge_vs_e51_dr_max=consist,
              **{"pass": bool(g1_dr < tol and g1_rank < tol and consist < 1e-3)})
    log(f"G1 {'PASS' if g1['pass'] else 'FAIL'} {g1}")
    out = dict(seed=seed, part=part, nparts=nparts, g1=g1, combos=CC["combos"], scales=scales, curves=curves, rows=rows, seconds=time.time() - T0)
    name = f"composite_seed{seed}" + (f"_part{part}of{nparts}" if nparts > 1 else "")
    (OUT / "h3" / f"{name}.json").write_text(json.dumps(out, default=float))
    if examples:
        test_idx = splits["test"][:ex["n_subjects"]]
        np.savez(OUT / "h3" / f"examples_seed{seed}.npz", test_index=test_idx, **examples)
    log(f"wrote h3/{name}.json ({len(rows)} rows)")
    return g1["pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    raise SystemExit(0 if run(a.seed, a.part, a.nparts) else 1)
