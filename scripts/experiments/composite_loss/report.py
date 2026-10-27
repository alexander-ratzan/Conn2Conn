"""Composite-loss protocol report (spec v2 E1.5): tables and figures for one instance, rebuilt from its per-run outputs.

Called by `protocol.py report --instance <name>`. Deterministic: everything is recomputed from runs/ (and the Stage 1
tables), so re-running on unchanged runs rewrites byte-identical tables and figures.

Tables  (<instance>/tables/): seed_records.csv, combo_summary.csv, epoch_history.csv.gz
Figures (<instance>/figures/, PNG 300 dpi, figure-making skill style; the interactive view of every instance is
viewers/cross_model/):
    tradeoff_scatter.png         test demeaned_pearson vs avg_rank, one point per combination (mean ± SE over seeds)
    term_trajectories.png        val raw value of each term over epochs (factorial combinations)
    loss_composition.png         each active term's share of the weighted training loss over epochs
    val_trajectories.png         val demeaned r over epochs (factorial combinations)
    dose_response.png            weight -> paired Δ test demeaned_pearson / avg_rank vs MSE-only (single terms + three-term mixtures; symlog)
    grad_cosine.png              per-term gradient cosines on the latent map over epochs (if recorded)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.results_utils import loss_grid as lg  # noqa: E402

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_strong": "#B64342",
           "teal": "#42949E", "violet": "#9A4D8E", "neutral": "#CFCECE", "grey": "#4D4D4D", "highlight": "#FFD700"}
TERM_COLOR = {"mse": PALETTE["blue_main"], "varmatch": PALETTE["teal"], "correye": PALETTE["violet"],
              "correye_dm": "#E3962B", "neidist": PALETTE["red_strong"]}
FACTORIAL_BLOCKS = ("factorial", "factorial_dm")
TERM_LABEL = {"mse": "MSE", "varmatch": "Var-match", "correye": "Corr-eye", "correye_dm": "Demeaned corr-eye",
              "neidist": "Neighbor dist"}


TERM_SHORT = {"varmatch": "VM", "correye": "CE", "correye_dm": "dCE", "neidist": "ND"}
SHORT_KEY = "VM Var-match · CE Corr-eye · dCE Demeaned corr-eye · ND Neighbor dist"


def combo_label(c, with_mse=True, sep=" + ", short=False):
    """Readable name for a combination (dict or summary row): 'MSE + neidist 0.5 + correye (de-meaned) 0.5'."""
    get = c.get if hasattr(c, "get") else (lambda k, d=None: c[k] if k in c else d)
    names = TERM_SHORT if short else TERM_LABEL
    parts = [f"{names[t]} {float(get(t, get(f'w_{t}', 0.0)) or 0.0):g}" for t in lg.TERMS
             if float(get(t, get(f"w_{t}", 0.0)) or 0.0) > 0]
    if not parts:
        return "MSE only"
    return sep.join((["MSE"] if with_mse else []) + parts)


def cos_label(col):
    """'cos_correye_dm_neidist' -> 'correye (de-meaned) vs neidist'."""
    body = col[len("cos_"):]
    for a in sorted(lg.ALL_TERMS, key=len, reverse=True):
        if body.startswith(a + "_") and body[len(a) + 1:] in lg.ALL_TERMS:
            return f"{TERM_LABEL[a]} vs {TERM_LABEL[body[len(a) + 1:]]}"
    return body
RC = {"font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "font.size": 14, "axes.spines.right": False,
      "axes.spines.top": False, "axes.linewidth": 2, "legend.frameon": False, "svg.fonttype": "none"}
DPI = 300
TEST_METRICS = ("demeaned_pearson", "avg_rank", "pearson", "mse", "top1_acc")
PAIRED_METRICS = ("demeaned_pearson", "avg_rank", "mse")


def _terms(df):
    """Protocol terms present in a table (v1 tables have no correye_dm column)."""
    return [t for t in lg.TERMS if f"w_{t}" in df]


def _mix_color(row):
    """Colour by which terms are active: MSE-only blue, single terms their own colour, mixtures grey shades."""
    active = [t for t in lg.TERMS if row.get(f"w_{t}", 0) > 0]
    if not active:
        return PALETTE["blue_main"]
    if len(active) == 1:
        return TERM_COLOR[active[0]]
    return PALETTE["grey"] if len(active) == 3 else PALETTE["green_3"]


def _save(fig, path):
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- tables
def build_tables(cfg, records, epochs):
    combos = {c["id"]: c for c in cfg["grid"]["combos"]}
    rec = records.copy()
    rec["block"] = rec["combo_id"].map(lambda c: combos.get(c, {}).get("block", "consensus"))
    for t in lg.TERMS:
        rec[f"w_{t}"] = rec["combo_id"].map(lambda c, t=t: float(combos.get(c, {}).get(t, 0.0)))
    rec = rec.sort_values(["stage", "combo_id", "seed"]).reset_index(drop=True)
    grid = rec[rec["stage"] == "stage2"]
    agg = {}
    for m in TEST_METRICS + ("val_demeaned_r_last",):
        col = f"test_{m}" if m in TEST_METRICS else m
        agg[f"{m}_mean"] = (col, "mean")
        agg[f"{m}_se"] = (col, lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan)
    summary = grid.groupby(["combo_id", "block"] + [f"w_{t}" for t in lg.TERMS], sort=False).agg(
        n_seeds=("seed", "nunique"), loss_signature=("loss_signature", "first"), **agg).reset_index()
    # Paired effects: every combination runs on the same seeds (= splits) as mse_only, so Δ per seed removes the
    # split-to-split variance that dominates the unpaired SE.
    base = grid[grid["combo_id"] == "mse_only"].set_index("seed")
    for m in PAIRED_METRICS:
        d = grid.assign(_d=grid[f"test_{m}"] - grid["seed"].map(base[f"test_{m}"]))
        g = d.groupby("combo_id")["_d"]
        summary[f"d_{m}_mean"] = summary["combo_id"].map(g.mean())
        summary[f"d_{m}_se"] = summary["combo_id"].map(g.std(ddof=1) / np.sqrt(g.count()))
    order = {c: i for i, c in enumerate(combos)}
    summary = summary.sort_values("combo_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return rec, summary


# --------------------------------------------------------------------------------------------- figures
def fig_tradeoff(summary, path):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for _, r in summary.iterrows():
        color = _mix_color(r)
        is_base = r["combo_id"] == "mse_only"
        ax.errorbar(r["avg_rank_mean"], r["demeaned_pearson_mean"], xerr=r["avg_rank_se"], yerr=r["demeaned_pearson_se"],
                    fmt="o", ms=12 if is_base else 8, color=color, ecolor=color, elinewidth=1.2, capsize=3,
                    mec="black" if is_base else color, mew=2 if is_base else 0.5, zorder=3 if is_base else 2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=l) for l, c in
               [("MSE only", PALETTE["blue_main"]), ("+ Var-match", TERM_COLOR["varmatch"]), ("+ Corr-eye", TERM_COLOR["correye"]),
                ("+ Demeaned corr-eye", TERM_COLOR["correye_dm"]), ("+ Neighbor dist", TERM_COLOR["neidist"]),
                ("+ two terms", PALETTE["green_3"]), ("+ three terms", PALETTE["grey"])]]
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    _place_labels(ax, summary)
    ax.text(0.99, 0.01, SHORT_KEY, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=PALETTE["grey"])
    ax.set_xlabel("Average rank (test, max)")
    ax.set_ylabel("Demeaned corr. (test, max)")
    ax.grid(alpha=0.25, ls="--")
    _save(fig, path)


def _place_labels(ax, summary):
    """Greedy, deterministic label placement: try a few offsets per point and skip a label that would overlap one
    already placed (every point stays identifiable in the cross-model viewer)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = [ax.get_legend().get_window_extent(renderer)] if ax.get_legend() else []
    pts = ax.transData.transform(summary[["avg_rank_mean", "demeaned_pearson_mean"]].to_numpy())
    from matplotlib.transforms import Bbox
    placed += [Bbox.from_extents(x - 9, y - 9, x + 9, y + 9) for x, y in pts]  # keep labels off the markers
    offsets = [(10, 0), (-10, 0), (10, 9), (10, -9), (-10, 9), (-10, -9)]
    rows = summary.sort_values(["demeaned_pearson_mean", "avg_rank_mean"], ascending=False)
    for _, r in rows.iterrows():
        for dx, dy in offsets:
            txt = ax.annotate(combo_label(r, with_mse=False, short=True), (r["avg_rank_mean"], r["demeaned_pearson_mean"]), xytext=(dx, dy),
                              textcoords="offset points", fontsize=8, color=PALETTE["grey"],
                              ha="left" if dx > 0 else "right", va="center")
            bb = txt.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in placed) and ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1):
                placed.append(bb)
                break
            txt.remove()


def _mean_over_seeds(ep, cols):
    return ep.groupby(["combo_id", "epoch"])[cols].mean().reset_index()


def fig_term_trajectories(ep, combos, path):
    plt.rcParams.update(RC)
    fac = [c["id"] for c in combos if c["block"] in FACTORIAL_BLOCKS]
    by_id = {c["id"]: c for c in combos}
    cols = [f"val_loss_raw_{t}" for t in lg.ALL_TERMS if f"val_loss_raw_{t}" in ep]
    if not cols:
        return False
    m = _mean_over_seeds(ep[ep["combo_id"].isin(fac)], cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.4))
    cmap = plt.get_cmap("tab20")
    for ax, col in zip(np.atleast_1d(axes), cols):
        for i, cid in enumerate(fac):
            g = m[m["combo_id"] == cid]
            ax.plot(g["epoch"], g[col], lw=2.2 if cid == "mse_only" else 1.4, color="black" if cid == "mse_only" else cmap(i),
                    label=combo_label(by_id[cid]))
        ax.set_title(f"Validation {TERM_LABEL[col.replace('val_loss_raw_', '')]} (min)", fontsize=12)
        ax.set_xlabel("Epoch")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, title="Training loss")
    _save(fig, path)
    return True


def fig_loss_composition(ep, combos, scales, path):
    plt.rcParams.update(RC)
    fac = [c for c in combos if c["block"] in FACTORIAL_BLOCKS and c["id"] != "mse_only"]
    ncol = min(4, len(fac))
    nrow = int(np.ceil(len(fac) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), sharey=True, squeeze=False)
    for ax in axes.flat[len(fac):]:
        ax.axis("off")
    for ax, c in zip(axes.flat, fac):
        g = _mean_over_seeds(ep[ep["combo_id"] == c["id"]], [f"train_loss_raw_{t}" for t in lg.ALL_TERMS if f"train_loss_raw_{t}" in ep])
        parts = {"mse": g["train_loss_raw_mse"].abs()}
        for t in lg.TERMS:
            if float(c.get(t, 0)) > 0:
                parts[t] = (float(c[t]) * g[f"train_loss_raw_{t}"] / float(scales[t])).abs()
        tot = sum(parts.values())
        ax.stackplot(g["epoch"], *[parts[k] / tot for k in parts], colors=[TERM_COLOR[k] for k in parts],
                     labels=[TERM_LABEL[k] for k in parts])
        ax.set_title(combo_label(c, sep="\n+ "), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
    for row in axes:
        row[0].set_ylabel("Share of |weighted loss|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM_COLOR[k]) for k in lg.ALL_TERMS if f"train_loss_raw_{k}" in ep]
    fig.legend(handles, [TERM_LABEL[k] for k in lg.ALL_TERMS if f"train_loss_raw_{k}" in ep], loc="lower center",
               ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    _save(fig, path)


def fig_val_trajectories(ep, combos, path):
    plt.rcParams.update(RC)
    fac = [c["id"] for c in combos if c["block"] in FACTORIAL_BLOCKS]
    by_id = {c["id"]: c for c in combos}
    m = _mean_over_seeds(ep[ep["combo_id"].isin(fac)], ["val_demeaned_r"])
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(fac):
        g = m[m["combo_id"] == cid]
        ax.plot(g["epoch"], g["val_demeaned_r"], lw=2.2 if cid == "mse_only" else 1.4, color="black" if cid == "mse_only" else cmap(i),
                label=combo_label(by_id[cid]))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation demeaned r (max)")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Training loss")
    _save(fig, path)


def fig_dose_response(summary, path):
    """Weight -> test metric for each single term, and for the equal-weight mixtures of three terms (one line per
    correye variant). Symlog x axis: the v2 correye ladders run to w = 50."""
    plt.rcParams.update(RC)
    terms = _terms(summary)
    W = summary[[f"w_{u}" for u in terms]]
    base = summary[summary["combo_id"] == "mse_only"].iloc[0]
    lines = {t: summary[(W.drop(columns=f"w_{t}").sum(axis=1) == 0) & (W[f"w_{t}"] > 0)] for t in terms}
    for variant in ("correye", "correye_dm"):
        if variant not in terms:
            continue
        others = [u for u in terms if u not in ("correye", "correye_dm")] + [variant]
        sel = (summary[[f"w_{u}" for u in others]] > 0).all(axis=1) & (summary[[f"w_{u}" for u in others]].nunique(axis=1) == 1)
        sel &= W[[f"w_{u}" for u in terms if u not in others]].sum(axis=1) == 0
        if sel.any():
            lines[f"three terms ({variant})"] = summary[sel]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, label in zip(axes, ("demeaned_pearson", "avg_rank"),
                                 ("Δ demeaned corr. vs MSE-only (test, max)", "Δ average rank vs MSE-only (test, max)")):
        for name, g in lines.items():
            if g.empty:
                continue
            wcol = f"w_{name}" if name in terms else f"w_{terms[0]}"
            xs = np.r_[0.0, g[wcol].to_numpy()]
            order = np.argsort(xs)
            ys = np.r_[0.0, g[f"d_{metric}_mean"].to_numpy()][order]
            es = np.r_[0.0, g[f"d_{metric}_se"].to_numpy()][order]
            three = name.startswith("three terms")
            color = PALETTE["grey"] if three else TERM_COLOR[name]
            shown = (f"Var-match + {TERM_LABEL[name.split('(')[1][:-1]]} + Neighbor dist (equal weights)" if three else TERM_LABEL[name])
            ax.errorbar(xs[order], ys, yerr=es, marker="o", lw=2, capsize=3, color=color, label=shown,
                        ls="--" if three and name.endswith("_dm)") else "-")
        ax.axhline(0, color=PALETTE["blue_main"], ls=":", lw=1.5)
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
        ax.set_xlim(-0.03, float(summary[[f"w_{u}" for u in terms]].to_numpy().max()) * 1.4)
        ax.set_xlabel("Term weight (scaled; MSE = 1; symlog)")
        ax.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=10, title="Training loss = MSE +")
    fig.suptitle("Mean ± SE of the per-seed difference from MSE-only (paired by seed / split)", fontsize=11, color=PALETTE["grey"])
    _save(fig, path)


def fig_grad_cosine(ep, combos, path):
    cos_cols = [c for c in ep.columns if c.startswith("cos_")]
    if not cos_cols:
        return False
    plt.rcParams.update(RC)
    show = [c for c in ("mse_only", "all_0.5", "alldm_0.5") if c in set(ep["combo_id"])]
    by_id = {c["id"]: c for c in combos}
    fig, axes = plt.subplots(1, len(show), figsize=(6.5 * len(show), 4.6), sharey=True, squeeze=False)
    for ax, cid in zip(axes[0], show):
        g = ep[(ep["combo_id"] == cid)].dropna(subset=cos_cols).groupby("epoch")[cos_cols].mean()
        for col in cos_cols:
            ax.plot(g.index, g[col], marker=".", lw=1.6, label=cos_label(col))
        ax.axhline(0, color=PALETTE["neutral"], lw=1)
        ax.set_title(f"Gradient cosine on the latent map\ntraining loss: {combo_label(by_id.get(cid, {'id': cid}))}", fontsize=10)
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("cosine")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, title="Gradient pair")
    _save(fig, path)
    return True


# --------------------------------------------------------------------------------------------- entry point
def build(instance):
    cfg = lg.load_instance(instance)
    frames = [lg.collect_runs(instance, stage) for stage in ("consensus", "stage2")]
    records = pd.concat([f[0] for f in frames if not f[0].empty], ignore_index=True) if any(not f[0].empty for f in frames) else pd.DataFrame()
    epochs = pd.concat([f[1].assign(stage=s) for f, s in zip(frames, ("consensus", "stage2")) if not f[1].empty], ignore_index=True) \
        if any(not f[1].empty for f in frames) else pd.DataFrame()
    if records.empty:
        print(f"{instance}: no runs yet")
        return 1
    # Only the instance's current grid: runs of combinations from an older grid stay in runs/ (and W&B) but are not
    # mixed into this grid's tables and figures.
    grid_ids = {c["id"] for c in cfg["grid"]["combos"]}
    keep = (records["stage"] != "stage2") | records["combo_id"].isin(grid_ids)
    dropped = sorted(set(records.loc[~keep, "combo_id"]))
    if dropped:
        print(f"{instance}: excluding {len(dropped)} combinations not in grid {cfg['grid_version']}: {dropped}")
    records = records[keep]
    if not epochs.empty:
        epochs = epochs[(epochs["stage"] != "stage2") | epochs["combo_id"].isin(grid_ids)]
    d = lg.instance_dir(instance)
    (d / "tables").mkdir(exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    seed_records, summary = build_tables(cfg, records, epochs)
    seed_records.to_csv(d / "tables" / "seed_records.csv", index=False, float_format="%.6g")
    epochs.to_csv(d / "tables" / "epoch_history.csv.gz", index=False, float_format="%.6g")
    written = ["tables/seed_records.csv", "tables/epoch_history.csv.gz"]
    if not summary.empty:
        summary.to_csv(d / "tables" / "combo_summary.csv", index=False, float_format="%.6g")
        written.append("tables/combo_summary.csv")
        combos = cfg["grid"]["combos"]
        ep2 = epochs[epochs["stage"] == "stage2"]
        fig_tradeoff(summary, d / "figures" / "tradeoff_scatter.png")
        written += ["figures/tradeoff_scatter.png"]
        if fig_term_trajectories(ep2, combos, d / "figures" / "term_trajectories.png"):
            written.append("figures/term_trajectories.png")
        fig_loss_composition(ep2, combos, cfg["state"]["reference_scales"]["c"], d / "figures" / "loss_composition.png")
        fig_val_trajectories(ep2, combos, d / "figures" / "val_trajectories.png")
        fig_dose_response(summary, d / "figures" / "dose_response.png")
        written += ["figures/loss_composition.png", "figures/val_trajectories.png", "figures/dose_response.png"]
        if fig_grad_cosine(ep2, combos, d / "figures" / "grad_cosine.png"):
            written.append("figures/grad_cosine.png")
    check = cfg["state"].get("consensus_check", {})
    if check.get("note"):
        (d / "tables" / "consensus_note.txt").write_text(f"basis: {check.get('basis')}\n{check['note']}\n")
        written.append("tables/consensus_note.txt")
        print(f"NOTE ({check.get('basis')}): {check['note']}")
    for w in written:
        print("wrote", d / w)
    return 0


if __name__ == "__main__":
    sys.exit(build(sys.argv[1]))
