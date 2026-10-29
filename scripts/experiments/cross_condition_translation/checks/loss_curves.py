"""Loss curves of FM-nl and FM-lin on real data (seed 0, code size k), full schedule without early stopping, with the
val average rank traced during training on the baselines' cells (56 single-source cells without SC; E4.0c compares B1 /
B2 on the same cells), so the flow loss can be read against the selection metric.

FM-nl: the search's best trial (2026-10-28, val 0.7958 over all cells); FM-lin: check_fm.py's settings.

    python scripts/experiments/cross_condition_translation/checks/loss_curves.py --k 256
        -> checks/loss_curves.json, figures/checks/fig_loss_curves.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fm_common import NONE, FlowData, eval_cells, train, val_avg_rank  # noqa: E402
from e4_common import REPO_ROOT  # noqa: E402

HP = {"fm_nl": dict(lr=0.002857403782946872, wd=8.745940174972092e-06, k_sc=64, width=256, depth=1, dropout=0.1805844122352601),
      "fm_lin": dict(lr=1e-3, wd=1e-4, k_sc=64, rank=128)}
CELLS = [c for c in eval_cells() if c[0] != NONE and not c[2]]
T = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--every", type=int, default=50)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FlowData(0, a.k, dev)
    out = {}
    for model, hp in HP.items():
        trace = []

        def cb(ep, net):
            trace.append((ep, val_avg_rank(net, data, T, cells=CELLS)))
            print(f"  {model} epoch {ep}: val avg rank {trace[-1][1]:.4f}", flush=True)

        net, info = train(model, hp, data, 0, early_stop=False, restore_best=False, callback=cb, every=a.every)
        out[model] = dict(hp=hp, history=info["history"], avg_rank=trace, best_epoch_by_val_loss=info["best_epoch"])
    dec = json.loads((HERE.parent / "tables" / "e40c" / "decision.json").read_text())["by_model"][str(a.k)]
    out["baselines_val_avg_rank"] = dec
    (HERE / "loss_curves.json").write_text(json.dumps(out, default=float))

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402
    fig_dir = HERE.parent / "figures" / "checks"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = {"fm_nl": "FM-nl (search best)", "fm_lin": "FM-lin"}
    with plt.rc_context({**mb.FIG_RC, "font.size": 14}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        for j, model in enumerate(HP):
            h = np.array(out[model]["history"])
            ax = axes[0, j]
            ax.plot(h[:, 0], h[:, 1], color="#7FB2E5", lw=1.2, label="train flow loss")
            ax.plot(h[:, 0], h[:, 2], color="#0F4D92", lw=1.2, label="val flow loss")
            ax.axvline(out[model]["best_epoch_by_val_loss"], color="#8C8C8C", ls="--", lw=1, label="best val loss (after epoch 300)")
            lo, hi = np.percentile(h[:, 1:], 0.5), np.percentile(h[:, 1:], 99.5)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_title(names[model], fontsize=16)
            ax.set_ylabel("flow-matching loss")
            ax.legend(frameon=False, fontsize=12)
            ax = axes[1, j]
            tr = np.array(out[model]["avg_rank"])
            ax.plot(tr[:, 0], tr[:, 1], "o-", color="#0F4D92", ms=3, lw=1.4, label=f"{names[model]} (T {T})")
            for key, lab, c in (("B1", "ridge (B1)", "#7FB2E5"), ("B2", "multiview CCA (B2)", "#42949E")):
                ax.axhline(dec[key], color=c, ls="--", lw=1.5, label=lab)
            ax.set_xlabel("epoch")
            ax.set_ylabel("val average rank\n(56 single-source cells, no SC)")
            ax.legend(frameon=False, fontsize=12, loc="lower right")
        fig.suptitle(f"E4 flows on real data (seed 0, k = {a.k}): loss and val average rank over training", fontsize=17)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_loss_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", fig_dir / "fig_loss_curves.png")


if __name__ == "__main__":
    main()
