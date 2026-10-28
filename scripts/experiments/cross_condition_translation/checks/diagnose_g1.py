"""E4.2 diagnosis of the failed checks: (a) is G1's planted nonlinearity learnable at this n? an MLP regression (MSE, no
flow) vs ridge per cell; (b) O3: FM-lin's generated spread vs the closed-form Gaussian's."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fm import SynthData, ridge_cell, row_r, HP
from fm_common import NONE, eval_cells, sample, train

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nl = SynthData(64, dev, nonlinear=True, seed=3)
def mlp_cell(a, t, on, width=256, epochs=600, wd=1e-3):
    Xs = {s: np.concatenate(([] if a == NONE else [nl.fc[s][a].cpu().numpy()]) + ([nl.sc[s].cpu().numpy()[:, :64]] if on else []), 1) for s in ("train", "val", "test")}
    X = {s: torch.tensor(Xs[s], dtype=torch.float32, device=dev) for s in Xs}; Y = {s: nl.fc[s][t] for s in Xs}
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X["train"].shape[1], width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 64)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd); best, st = 1e9, None
    for ep in range(epochs):
        net.train(); loss = ((net(X["train"]) - Y["train"]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = ((net(X["val"]) - Y["val"]) ** 2).mean().item()
        if v < best: best, st = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(st)
    with torch.no_grad():
        return net(X["test"]).cpu().numpy()
wins, d = 0, []
cells = [c for c in eval_cells() if not c[2] and c[0] != NONE][::4]
for a, t, on in cells:
    P_m = mlp_cell(a, t, on); P_r, _ = ridge_cell(nl, a, t, on)
    Y = nl.fc["test"][t].cpu().numpy(); mu = nl.fc["train"][t].cpu().numpy().mean(0)
    rm, rr = row_r(P_m, Y, mu).mean(), row_r(P_r, Y, mu).mean()
    wins += rm > rr; d.append(rm - rr)
print(f"(a) G1 data, FC -> FC cells: MLP regression beats ridge in {wins}/{len(cells)} cells; mean r diff {np.mean(d):+.4f} "
      f"(ridge mean r {np.mean([row_r(ridge_cell(nl, *c)[0], nl.fc['test'][c[1]].cpu().numpy(), nl.fc['train'][c[1]].cpu().numpy().mean(0)).mean() for c in cells]):.3f})", flush=True)

lin = SynthData(64, dev, nonlinear=False, seed=1)
net, info = train("fm_lin", HP["fm_lin"], lin, 2, max_epochs=1500)
for a, t, on in [c for c in eval_cells()][::24]:
    X = sample(net, lin, "test", a, t, on, 64, 1.0).cpu().numpy()
    P_r, cov = ridge_cell(lin, a, t, on)
    print(f"(b) cell {a}->{t} sc {on}: FM-lin spread (trace of sample cov) {X.var(0, ddof=1).sum(1).mean():.2f}, Gaussian residual trace {np.trace(cov):.2f}, "
          f"epochs {info['epochs_run']} best {info['best_epoch']}", flush=True)
