"""Diagnostic (E4.2): does FM-lin / FM-nl learn a linear-Gaussian synthetic map? Training curves without early stopping."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fm import SynthData, ridge_cell, row_r
from fm_common import sample, train

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lin = SynthData(64, dev, nonlinear=False, seed=1)
cells = [(0, 1, False), (8, 2, True), (3, 5, True)]
def probe(net):
    out = []
    for c in cells:
        X = sample(net, lin, "test", *c, 64, 1.0).mean(0).cpu().numpy()
        P_r, _ = ridge_cell(lin, *c)
        Y = lin.fc["test"][c[1]].cpu().numpy(); mu = lin.fc["train"][c[1]].cpu().numpy().mean(0)
        out.append((np.mean([np.corrcoef(p, q)[0, 1] for p, q in zip(X, P_r)]), row_r(X, Y, mu).mean(), row_r(P_r, Y, mu).mean()))
    return np.array(out).mean(0)
import sys as _s
only = _s.argv[1] if len(_s.argv) > 1 else None
for model, hp in (("fm_lin", dict(k_sc=64, rank=128)), ("fm_nl", dict(width=256, depth=2, dropout=0.0, emb_dim=32, k_sc=64))):
    if only and model != only:
        continue
    for lr in (1e-3, 3e-3, 1e-2):
        for ep in (200, 1000):
            net, info = train(model, dict(hp, lr=lr, wd=0.0), lin, 0, max_epochs=ep, early_stop=False, restore_best=False)
            h = info["history"]
            r = probe(net)
            print(f"{model} lr {lr:g} epochs {ep}: train loss {h[0][1]:.3f} -> {h[-1][1]:.3f}, val {h[-1][2]:.3f} | "
                  f"r(mean, ridge) {r[0]:.3f}, r(mean, truth) {r[1]:.3f}, r(ridge, truth) {r[2]:.3f}", flush=True)
