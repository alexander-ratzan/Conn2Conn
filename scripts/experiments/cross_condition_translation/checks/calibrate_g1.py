"""E4.2: calibrate G1's synthetic nonlinearity so a supervised MLP (MSE) clearly beats ridge at n = 659 (otherwise G1
cannot discriminate). Prints the MLP - ridge gain per (noise, strength, latent) setting."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fm import SynthData, ridge_cell, row_r
from fm_common import NONE, eval_cells

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def mlp(data, a, t, on, width=256, epochs=800):
    Xs = {s: np.concatenate(([] if a == NONE else [data.fc[s][a].cpu().numpy()]) + ([data.sc[s].cpu().numpy()[:, :64]] if on else []), 1) for s in ("train", "val", "test")}
    X = {s: torch.tensor(Xs[s], dtype=torch.float32, device=dev) for s in Xs}; Y = {s: data.fc[s][t] for s in Xs}
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X["train"].shape[1], width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, Y["train"].shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3); best, st = 1e9, None
    for ep in range(epochs):
        loss = ((net(X["train"]) - Y["train"]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = ((net(X["val"]) - Y["val"]) ** 2).mean().item()
        if v < best: best, st = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(st)
    with torch.no_grad():
        return net(X["test"]).cpu().numpy()
cells = [c for c in eval_cells() if not c[2] and c[0] != NONE][::4]
for noise, strength, latent in ((0.5, 2.0, 8), (0.2, 3.0, 8), (0.1, 3.0, 4), (0.1, 5.0, 4), (0.05, 5.0, 2)):
    d = SynthData(64, dev, nonlinear=True, seed=3, noise=noise, strength=strength, latent=latent)
    g, rr = [], []
    for a, t, on in cells:
        Y = d.fc["test"][t].cpu().numpy(); mu = d.fc["train"][t].cpu().numpy().mean(0)
        r_r = row_r(ridge_cell(d, a, t, on)[0], Y, mu).mean(); r_m = row_r(mlp(d, a, t, on), Y, mu).mean()
        g.append(r_m - r_r); rr.append(r_r)
    g = np.array(g)
    print(f"noise {noise} strength {strength} latent {latent}: ridge r {np.mean(rr):.3f}, MLP - ridge {g.mean():+.4f} "
          f"(wins {int((g > 0).sum())}/{len(g)})", flush=True)
