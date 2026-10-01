"""Composite-loss protocol report (spec v2 E1.5): tables and figures for one instance, rebuilt from its per-run outputs.

Called by `protocol.py report --instance <name>`. Deterministic: everything is recomputed from runs/ (and the Stage 1
tables), so re-running on unchanged runs rewrites byte-identical tables and figures.

Tables  (<instance>/tables/): seed_records.csv, combo_summary.csv, epoch_history.csv.gz
Figures (<instance>/figures/, PNG 300 dpi, figure-making skill style):
    tradeoff_scatter.png         test demeaned_pearson vs avg_rank, one point per combination (mean ± SE over seeds)
    tradeoff_interactive.html    same scatter, self-contained; hover/click shows the weights and per-seed values
    term_trajectories.png        val raw value of each term over epochs (factorial combinations)
    loss_composition.png         each active term's share of the weighted training loss over epochs
    val_trajectories.png         val demeaned r over epochs (factorial combinations)
    dose_response.png            weight -> test demeaned_pearson / avg_rank per term (single-term and all-three lines)
    grad_cosine.png              per-term gradient cosines on the latent map over epochs (if recorded)
"""
import html
import json
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
TERM_COLOR = {"mse": PALETTE["blue_main"], "varmatch": PALETTE["teal"], "correye": PALETTE["violet"], "neidist": PALETTE["red_strong"]}
RC = {"font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "font.size": 14, "axes.spines.right": False,
      "axes.spines.top": False, "axes.linewidth": 2, "legend.frameon": False, "svg.fonttype": "none"}
DPI = 300
TEST_METRICS = ("demeaned_pearson", "avg_rank", "pearson", "mse", "top1_acc")


def _mix_color(row):
    """Colour by which terms are active: MSE-only blue, single terms their own colour, mixtures grey shades."""
    active = [t for t in lg.TERMS if row[f"w_{t}"] > 0]
    if not active:
        return PALETTE["blue_main"]
    if len(active) == 1:
        return TERM_COLOR[active[0]]
    return PALETTE["grey"] if len(active) == 3 else PALETTE["green_3"]


def _save(fig, path):
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, facecolor="white")
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
        ax.annotate(r["combo_id"], (r["avg_rank_mean"], r["demeaned_pearson_mean"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=PALETTE["grey"])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=l) for l, c in
               [("MSE only", PALETTE["blue_main"]), ("+ varmatch", TERM_COLOR["varmatch"]), ("+ correye", TERM_COLOR["correye"]),
                ("+ neidist", TERM_COLOR["neidist"]), ("two terms", PALETTE["green_3"]), ("all three", PALETTE["grey"])]]
    ax.legend(handles=handles, loc="best", fontsize=10)
    ax.set_xlabel("Average rank (test)")
    ax.set_ylabel("Demeaned corr. (test)")
    ax.grid(alpha=0.25, ls="--")
    _save(fig, path)


def fig_interactive(summary, records, path, title):
    """Self-contained SVG scatter with hover/click details (no external scripts)."""
    x, y = summary["avg_rank_mean"].to_numpy(), summary["demeaned_pearson_mean"].to_numpy()
    xe, ye = summary["avg_rank_se"].fillna(0).to_numpy(), summary["demeaned_pearson_se"].fillna(0).to_numpy()
    pad = lambda lo, hi: (lo - 0.08 * (hi - lo or 1), hi + 0.08 * (hi - lo or 1))
    x0, x1 = pad(float(np.min(x - xe)), float(np.max(x + xe)))
    y0, y1 = pad(float(np.min(y - ye)), float(np.max(y + ye)))
    W, H, L, B = 720, 520, 70, 60
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - 20)
    sy = lambda v: H - B - (v - y0) / (y1 - y0) * (H - B - 20)
    points = []
    for i, r in summary.iterrows():
        seeds = records[(records["stage"] == "stage2") & (records["combo_id"] == r["combo_id"])].sort_values("seed")
        info = {"id": r["combo_id"], "block": r["block"], "signature": r["loss_signature"],
                "weights": {"mse": 1.0, **{t: float(r[f"w_{t}"]) for t in lg.TERMS}},
                "demeaned_pearson": f"{r['demeaned_pearson_mean']:.4f} ± {r['demeaned_pearson_se']:.4f}",
                "avg_rank": f"{r['avg_rank_mean']:.4f} ± {r['avg_rank_se']:.4f}",
                "per_seed": [{"seed": int(s.seed), "demeaned_pearson": round(s.test_demeaned_pearson, 4),
                              "avg_rank": round(s.test_avg_rank, 4)} for s in seeds.itertuples()]}
        cx, cy = sx(x[i]), sy(y[i])
        color = _mix_color(r)
        points.append(
            f'<line x1="{sx(x[i]-xe[i]):.1f}" x2="{sx(x[i]+xe[i]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{color}"/>'
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{sy(y[i]-ye[i]):.1f}" y2="{sy(y[i]+ye[i]):.1f}" stroke="{color}"/>'
            f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="{9 if r["combo_id"] == "mse_only" else 7}" fill="{color}" '
            f'stroke="{"#000" if r["combo_id"] == "mse_only" else "#fff"}" stroke-width="1.5" data-info="{html.escape(json.dumps(info))}"/>')
    ticks = []
    for v in np.linspace(x0, x1, 5):
        ticks.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" text-anchor="middle">{v:.3f}</text>')
    for v in np.linspace(y0, y1, 5):
        ticks.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.3f}</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
           f'<line x1="{L}" y1="{H-B}" x2="{W-20}" y2="{H-B}" class="ax"/><line x1="{L}" y1="20" x2="{L}" y2="{H-B}" class="ax"/>'
           + "".join(ticks) + "".join(points) +
           f'<text x="{(W+L)/2:.0f}" y="{H-15}" text-anchor="middle" class="lab">Average rank (test)</text>'
           f'<text transform="translate(18 {(H-B)/2:.0f}) rotate(-90)" text-anchor="middle" class="lab">Demeaned corr. (test)</text></svg>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#fff;--fg:#1f2328;--muted:#57606a;--card:#f6f8fa}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--card:#161b22}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 "DejaVu Sans",Helvetica,Arial,sans-serif}}
main{{max-width:1100px;margin:auto;display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:16px}}
@media (max-width:760px){{main{{grid-template-columns:1fr}}}}
svg{{width:100%;height:auto}} svg text{{fill:var(--muted);font-size:11px}} .lab{{font-size:13px;fill:var(--fg)}}
.ax{{stroke:var(--muted);stroke-width:1.5}} .pt{{cursor:pointer}} .pt:hover,.pt.sel{{stroke:var(--fg);stroke-width:3}}
#panel{{background:var(--card);border-radius:8px;padding:12px;min-height:200px}} table{{border-collapse:collapse;width:100%}}
td,th{{padding:2px 6px;text-align:left;border-bottom:1px solid var(--muted)}} h1{{font-size:18px;max-width:1100px;margin:0 auto 12px}}
</style></head><body><h1>{html.escape(title)}</h1><main><div>{svg}</div>
<div id="panel"><em>Hover or click a point to see its loss weights and per-seed results.</em></div></main>
<script>
const panel=document.getElementById('panel');
function show(el){{const d=JSON.parse(el.dataset.info);
 let w=Object.entries(d.weights).map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
 let s=d.per_seed.map(r=>`<tr><td>${{r.seed}}</td><td>${{r.demeaned_pearson}}</td><td>${{r.avg_rank}}</td></tr>`).join('');
 panel.innerHTML=`<h3>${{d.id}} <small>(${{d.block}})</small></h3><p><code>${{d.signature}}</code></p>
 <table><tr><th>term</th><th>weight</th></tr>${{w}}</table>
 <p>demeaned corr: ${{d.demeaned_pearson}}<br>avg rank: ${{d.avg_rank}}</p>
 <table><tr><th>seed</th><th>demeaned</th><th>avg rank</th></tr>${{s}}</table>`;}}
document.querySelectorAll('.pt').forEach(el=>{{el.addEventListener('mouseenter',()=>show(el));
 el.addEventListener('click',()=>{{document.querySelectorAll('.pt').forEach(p=>p.classList.remove('sel'));el.classList.add('sel');show(el);}});}});
</script></body></html>"""
    Path(path).write_text(page)


def _mean_over_seeds(ep, cols):
    return ep.groupby(["combo_id", "epoch"])[cols].mean().reset_index()


def fig_term_trajectories(ep, combos, path):
    plt.rcParams.update(RC)
    fac = [c["id"] for c in combos if c["block"] == "factorial"]
    cols = [f"val_loss_raw_{t}" for t in lg.ALL_TERMS if f"val_loss_raw_{t}" in ep]
    if not cols:
        return False
    m = _mean_over_seeds(ep[ep["combo_id"].isin(fac)], cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4))
    cmap = plt.get_cmap("tab10")
    for ax, col in zip(np.atleast_1d(axes), cols):
        for i, cid in enumerate(fac):
            g = m[m["combo_id"] == cid]
            ax.plot(g["epoch"], g[col], lw=2.2 if cid == "mse_only" else 1.4, color="black" if cid == "mse_only" else cmap(i), label=cid)
        ax.set_title(col.replace("val_loss_raw_", "val "))
        ax.set_xlabel("Epoch")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _save(fig, path)
    return True


def fig_loss_composition(ep, combos, scales, path):
    plt.rcParams.update(RC)
    fac = [c for c in combos if c["block"] == "factorial" and c["id"] != "mse_only"]
    fig, axes = plt.subplots(1, len(fac), figsize=(3.2 * len(fac), 3.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), fac):
        g = _mean_over_seeds(ep[ep["combo_id"] == c["id"]], [f"train_loss_raw_{t}" for t in lg.ALL_TERMS if f"train_loss_raw_{t}" in ep])
        parts = {"mse": g["train_loss_raw_mse"].abs()}
        for t in lg.TERMS:
            if float(c.get(t, 0)) > 0:
                parts[t] = (float(c[t]) * g[f"train_loss_raw_{t}"] / float(scales[t])).abs()
        tot = sum(parts.values())
        ax.stackplot(g["epoch"], *[parts[k] / tot for k in parts], colors=[TERM_COLOR[k] for k in parts], labels=list(parts))
        ax.set_title(c["id"], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Share of |weighted loss|")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="lower right")
    _save(fig, path)


def fig_val_trajectories(ep, combos, path):
    plt.rcParams.update(RC)
    fac = [c["id"] for c in combos if c["block"] == "factorial"]
    m = _mean_over_seeds(ep[ep["combo_id"].isin(fac)], ["val_demeaned_r"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, cid in enumerate(fac):
        g = m[m["combo_id"] == cid]
        ax.plot(g["epoch"], g["val_demeaned_r"], lw=2.2 if cid == "mse_only" else 1.4, color="black" if cid == "mse_only" else cmap(i), label=cid)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val demeaned r")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def fig_dose_response(summary, path):
    plt.rcParams.update(RC)
    base = summary[summary["combo_id"] == "mse_only"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in zip(axes, ("demeaned_pearson", "avg_rank"), ("Demeaned corr. (test)", "Average rank (test)")):
        lines = {t: summary[(summary[[f"w_{u}" for u in lg.TERMS if u != t]].sum(axis=1) == 0) & (summary[f"w_{t}"] > 0)] for t in lg.TERMS}
        lines["all three"] = summary[(summary[[f"w_{u}" for u in lg.TERMS]] > 0).all(axis=1)
                                     & (summary[[f"w_{u}" for u in lg.TERMS]].nunique(axis=1) == 1)]
        for name, g in lines.items():
            w = g[f"w_{lg.TERMS[0]}"] if name == "all three" else g[f"w_{name}"]
            xs = np.r_[0.0, w.to_numpy()]
            order = np.argsort(xs)
            ys = np.r_[base[f"{metric}_mean"], g[f"{metric}_mean"].to_numpy()][order]
            es = np.r_[base[f"{metric}_se"], g[f"{metric}_se"].to_numpy()][order]
            color = PALETTE["grey"] if name == "all three" else TERM_COLOR[name]
            ax.errorbar(xs[order], ys, yerr=es, marker="o", lw=2, capsize=3, color=color, label=name)
        ax.axhline(base[f"{metric}_mean"], color=PALETTE["blue_main"], ls=":", lw=1.5)
        ax.set_xlabel("Term weight (scaled; MSE = 1)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=10)
    _save(fig, path)


def fig_grad_cosine(ep, path):
    cos_cols = [c for c in ep.columns if c.startswith("cos_")]
    if not cos_cols:
        return False
    plt.rcParams.update(RC)
    show = [c for c in ("mse_only", "all_0.5") if c in set(ep["combo_id"])]
    fig, axes = plt.subplots(1, len(show), figsize=(6.5 * len(show), 4.2), sharey=True, squeeze=False)
    for ax, cid in zip(axes[0], show):
        g = ep[(ep["combo_id"] == cid)].dropna(subset=cos_cols).groupby("epoch")[cos_cols].mean()
        for col in cos_cols:
            ax.plot(g.index, g[col], marker=".", lw=1.6, label=col.replace("cos_", "").replace("_", " vs ", 1))
        ax.axhline(0, color=PALETTE["neutral"], lw=1)
        ax.set_title(f"{cid}: gradient cosine on the latent map", fontsize=11)
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("cosine")
    axes[0][0].legend(fontsize=8, ncol=2)
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
        fig_interactive(summary, seed_records, d / "figures" / "tradeoff_interactive.html",
                        f"{cfg['model']}: composite-loss trade-off (grid {cfg['grid_version']})")
        written += ["figures/tradeoff_scatter.png", "figures/tradeoff_interactive.html"]
        if fig_term_trajectories(ep2, combos, d / "figures" / "term_trajectories.png"):
            written.append("figures/term_trajectories.png")
        fig_loss_composition(ep2, combos, cfg["state"]["reference_scales"]["c"], d / "figures" / "loss_composition.png")
        fig_val_trajectories(ep2, combos, d / "figures" / "val_trajectories.png")
        fig_dose_response(summary, d / "figures" / "dose_response.png")
        written += ["figures/loss_composition.png", "figures/val_trajectories.png", "figures/dose_response.png"]
        if fig_grad_cosine(ep2, d / "figures" / "grad_cosine.png"):
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
