"""Real-data test of the E2.1-form linear flow (fm_shared: one input map for every pair; fm_shared_tgt: one per target)
before benchmarking it (2026-10-28). Seed 0, code size k, trained as the pipeline trains (val flow-loss early stopping
after epoch 50, best weights restored), val average rank traced every 50 epochs at T in {0.25, 0.5, 1}.

Cells: the 56 single-source cells without SC, the cells E4.0c compares B1 / B2 on. Pass (per model): the selected
weights at their best T reach seed 0's ridge (B1) val average rank on those cells. The best passing model is benchmarked
(run_flows.py search + final); if neither passes, E4's flows stop at FM-nl / FM-lin.

    python scripts/experiments/cross_condition_translation/checks/shared_flow_test.py --k 256
        -> checks/shared_flow_test.json, figures/checks/fig_shared_flow_test.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fm_common import NONE, FC, FlowData, eval_cells, n_params, train, val_avg_rank  # noqa: E402
from e4_common import REPO_ROOT  # noqa: E402

MODELS = ("fm_shared", "fm_shared_tgt")
HP = dict(lr=1e-3, wd=1e-4, k_sc=64)
CELLS = [c for c in eval_cells() if c[0] != NONE and not c[2]]
TEMPS = (0.25, 0.5, 1.0)


def baselines(k, seed=0):
    rows = json.loads((HERE.parent / "results" / "e40c" / f"k{k}" / f"seed{seed}.json").read_text())["rows"]
    v = [r for r in rows if r["split"] == "val" and not r["with_sc"] and r["source"] in FC]
    return {m: float(np.mean([r["avg_rank"] for r in v if r["model"] == m])) for m in ("B0-codes", "B1", "B2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FlowData(0, a.k, dev)
    ref = baselines(a.k)
    print("seed 0 val average rank, 56 cells:", {m: round(v, 4) for m, v in ref.items()}, flush=True)
    out = dict(baselines_seed0=ref, hp=HP)
    for model in MODELS:
        trace = []

        def cb(ep, net):
            trace.append([ep] + [val_avg_rank(net, data, T, cells=CELLS) for T in TEMPS])
            print(f"  {model} epoch {ep}: " + ", ".join(f"T {T} {v:.4f}" for T, v in zip(TEMPS, trace[-1][1:])), flush=True)

        net, info = train(model, HP, data, 0, callback=cb, every=50)
        sel = {str(T): val_avg_rank(net, data, T, cells=CELLS) for T in TEMPS}
        best_T = max(sel, key=sel.get)
        ok = sel[best_T] >= ref["B1"]
        out[model] = dict(n_params=n_params(net), history=info["history"], trace=trace, best_epoch=info["best_epoch"],
                          epochs_run=info["epochs_run"], selected=sel, best_T=float(best_T), passed=bool(ok))
        print(f"{model} ({n_params(net):,} params): best epoch {info['best_epoch']} of {info['epochs_run']}; selected weights "
              + ", ".join(f"T {T} {v:.4f}" for T, v in sel.items()) + f" -> {'PASS' if ok else 'FAIL'} vs ridge {ref['B1']:.4f}", flush=True)
    passed = [m for m in MODELS if out[m]["passed"]]
    out["choice"] = max(passed, key=lambda m: max(out[m]["selected"].values())) if passed else None
    print("CHOICE", out["choice"], flush=True)
    (HERE / "shared_flow_test.json").write_text(json.dumps(out, default=float))

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402
    prev = json.loads((HERE / "loss_curves.json").read_text()) if (HERE / "loss_curves.json").exists() else {}
    names = {"fm_shared": "linear flow, shared input map", "fm_shared_tgt": "linear flow, input map per target"}
    tcol = {0.25: "#0F4D92", 0.5: "#7FB2E5", 1.0: "#8BCF8B"}
    with plt.rc_context({**mb.FIG_RC, "font.size": 14}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
        for j, model in enumerate(MODELS):
            h = np.array(out[model]["history"])
            ax = axes[0, j]
            ax.plot(h[:, 0], h[:, 1], color="#7FB2E5", lw=1.2, label="train flow loss")
            ax.plot(h[:, 0], h[:, 2], color="#0F4D92", lw=1.2, label="val flow loss")
            ax.axvline(out[model]["best_epoch"], color="#8C8C8C", ls="--", lw=1, label="selected epoch (val loss)")
            lo, hi = np.percentile(h[:, 1:], 0.5), np.percentile(h[:, 1:], 99.5)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_title(f"{names[model]} ({out[model]['n_params'] / 1e6:.2f} M params)", fontsize=15)
            ax.set_ylabel("flow-matching loss")
            ax.legend(frameon=False, fontsize=12)
            ax = axes[1, j]
            tr = np.array(out[model]["trace"])
            for i, T in enumerate(TEMPS):
                ax.plot(tr[:, 0], tr[:, i + 1], "o-", ms=3, lw=1.3, color=tcol[T], label=f"T {T}")
            for m, lab, c in (("B1", "ridge (B1)", "#CFCECE"), ("B2", "multiview CCA (B2)", "#42949E")):
                ax.axhline(ref[m], color=c, ls="--", lw=1.6, label=f"{lab}, seed 0")
            for pm, lab in (("fm_nl", "FM-nl"), ("fm_lin", "FM-lin")):
                if pm in prev:
                    p = np.array(prev[pm]["avg_rank"])
                    ax.plot(p[:, 0], p[:, 1], color="#8C8C8C", lw=1, ls=":" if pm == "fm_lin" else "-", label=f"{lab} (T 0.5, earlier run)")
            ax.set_xlim(-20, h[-1, 0] + 20)
            ax.set_ylim(0.70, max(0.83, tr[:, 1:].max() + 0.01))
            ax.set_xlabel("epoch")
            ax.set_ylabel("val average rank\n(56 single-source cells, no SC)")
            ax.legend(frameon=False, fontsize=11, loc="lower right", ncol=2)
        fig.suptitle(f"E4: E2.1-form linear flow on real data (seed 0, k = {a.k}); choice: {out['choice']}", fontsize=17)
        fig.tight_layout()
        d = HERE.parent / "figures" / "checks"
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_shared_flow_test.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
