"""Spec v4 E6: is SC coupled to the fingerprint directions? Held-out R² of SC's predictions per pooled-map direction.

The pooled map's directions v_j solve 8 S_f v = w (8 S_f + S_d) v (generalised symmetric eigenproblem; w_j = fingerprint
share of the direction's variance). Each subject's codes are projected onto them (c_j = z v_j) and SC's held-out R² is
computed per direction, for a fixed target:
  route 1  SC -> rest codes (ridge, k_SC and penalty on val average rank against measured rest FC, as run_checks.py),
           scored against the subject's rest codes
  route 4  SC -> mean fingerprint (same selection), scored against the subject's mean fingerprint
If R² rises with w, SC predicts the fingerprint directions of FC best, whatever the training target.

    python scripts/experiments/fingerprint_embedding/r2_by_direction.py  -> results/r2_by_direction.json,
                                                                          figures/fig_r2_by_direction.png
"""
import json
import sys

import numpy as np
from scipy.linalg import eigh

from fe_common import FC, HERE, K, REPO_ROOT, SEEDS, SeedData, codes, fingerprint, score
from run_e40c import ridge_path  # noqa: E402

ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
K_SC = [64, 128, 256]
ALL = ("train", "val", "test")


def directions(Z):
    f = fingerprint(Z)
    Sf = f.T @ f
    Sd = sum((Z[v] - f).T @ (Z[v] - f) for v in FC)
    w, V = eigh(8 * Sf, 8 * Sf + Sd)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def fit_route(sd, Z, target):
    Ytr = target(Z["train"])
    best = None
    for k in K_SC:
        X = {s: sd.code("SC", s, k).astype(np.float64) for s in ALL}
        for a, B in ridge_path(X["train"], Ytr, ALPHAS).items():
            v = score(sd.decode("rest", X["val"] @ B), sd.edges[("val", "rest")], sd.train_mean["rest"])["avg_rank"]
            if best is None or v > best[0]:
                best = (v, X["test"] @ B)
    return best[1]


def r2_per_dir(pred, true, train_mean, V):
    cp, ct, cm = pred @ V, true @ V, train_mean @ V
    return 1 - ((ct - cp) ** 2).sum(0) / ((ct - cm) ** 2).sum(0)


def main():
    out = {}
    for s in SEEDS:
        sd = SeedData(s, "main")
        Z = {sp: codes(sd, sp) for sp in ALL}
        w, V = directions(Z["train"])
        rec = dict(w=w.tolist())
        for name, tgt in (("route 1: SC -> rest codes", lambda Zs: Zs["rest"]), ("route 4: SC -> mean fingerprint", fingerprint)):
            P = fit_route(sd, Z, tgt)
            rec[name] = r2_per_dir(P, tgt(Z["test"]), tgt(Z["train"]).mean(0), V).tolist()
        out[str(s)] = rec
        print(f"seed {s}: done", flush=True)
    (HERE / "results" / "r2_by_direction.json").write_text(json.dumps(out))
    figure(out)


def figure(out):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import run as mb  # noqa: E402

    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    cols = {"route 1: SC -> rest codes": "#8C8C8C", "route 4: SC -> mean fingerprint": "#7FB2E5"}
    labs = {"route 1: SC -> rest codes": "1  SC → rest codes, scored on rest codes",
            "route 4: SC -> mean fingerprint": "4  SC → mean fingerprint, scored on the fingerprint"}
    own = pd.read_csv(HERE / "tables" / "sc_routes_stage1_own_target.csv") if (HERE / "tables" / "sc_routes_stage1_own_target.csv").exists() else None
    with plt.rc_context({**mb.FIG_RC, "font.size": 13}):
        f_, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
        ax = axes[0]
        for name, c in cols.items():
            per_seed = []
            for rec in out.values():
                w, r = np.array(rec["w"]), np.array(rec[name])
                idx = np.digitize(w, bins) - 1
                per_seed.append([r[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
            per_seed = np.array(per_seed)
            m, e = np.nanmean(per_seed, 0), np.nanstd(per_seed, 0, ddof=1) / np.sqrt(len(per_seed))
            mid = (bins[:-1] + bins[1:]) / 2
            ax.errorbar(mid, m, yerr=e, color=c, marker="o", lw=2, ms=6, capsize=3, label=labs[name])
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("fingerprint share w of the direction (pooled map)", fontsize=14)
        ax.set_ylabel("held-out R² of SC's prediction\n(mean over the directions in the bin)", fontsize=14)
        ax.set_title("Which FC directions does SC predict? (test, 5 seeds; directions binned by w)", fontsize=14)
        ax.legend(frameon=False, fontsize=12, loc="upper left")
        ax = axes[1]
        if own is not None:
            order = ["1  Rest codes", "2  CCA, SC as a view", "3  CCA consensus", "4  Mean fingerprint", "5  Fingerprint embedding"]
            q = own.set_index("route").reindex(order)
            colours = ["#8C8C8C", "#2A7F87", "#7CC4C9", "#7FB2E5", "#B2182B"]
            x = np.arange(len(order))
            vals = q.r2.where(q.index != "2  CCA, SC as a view")
            ax.bar(x, vals, 0.7, yerr=q.r2_se.where(q.index != "2  CCA, SC as a view"), color=colours, edgecolor="black", lw=1, capsize=3)
            ax.text(1, 0.002, "not a\nregression:\nR² n/a", ha="center", va="bottom", fontsize=10)
            ax.set_xticks(x, [o_.split("  ")[0] for o_ in order])
            ax.set_xlabel("route (as in fig_sc_routes)", fontsize=13)
            ax.set_ylabel("held-out R² of the route's own target", fontsize=14)
            ax.set_title("R² of each route's own target\n(different targets: not comparable across routes)", fontsize=14)
        f_.suptitle("SC → FC: variance explained by direction of FC (left) and by route (right)", fontsize=16)
        f_.tight_layout()
        f_.savefig(HERE / "figures" / "fig_r2_by_direction.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figure-only":
        figure(json.loads((HERE / "results" / "r2_by_direction.json").read_text()))
    else:
        main()
