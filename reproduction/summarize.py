#!/usr/bin/env python3
"""Summarize the grid into reports/reproduction_findings.md (metric reporting order).

Reconstruction leads with demeaned_pearson (+ avg_rank); downstream leads with
lift_over_bvdemo (+ median paired-permutation p). Seed-means with std across the 10 seeds,
split by parcellation so the 4S456 replication of F1-F5 is read directly.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import OUTPUTS_DIR, CONFIGS_DIR  # noqa: E402

REPORTS = OUTPUTS_DIR.parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def fmt_mean_std(g):
    return f"{g.mean():.3f}±{g.std():.3f}"


def recon_section(df):
    out = ["## Reconstruction (demeaned_pearson, mean±std over seeds)\n"]
    pls = df[df.estimator == "pca_pls"]
    for parc in sorted(df.parcellation.unique()):
        d = pls[pls.parcellation == parc]
        if d.empty:
            continue
        out.append(f"### {parc} (estimator=pca_pls)\n")
        out.append("| input → target | demeaned_r | avg_rank | top1 |")
        out.append("|---|---|---|---|")
        order = [("FC", "SC"), ("SC", "FC"), ("bv", "SC"), ("bv", "FC"),
                 ("demo", "SC"), ("demo", "FC"), ("bv+demo", "SC"), ("bv+demo", "FC"),
                 ("FC+bv+demo", "SC"), ("SC+bv+demo", "FC"), ("FC", "FC"), ("SC", "SC")]
        for (i, t) in order:
            c = d[(d.input_set == i) & (d.target == t)]
            if c.empty:
                continue
            out.append(f"| {i} → {t} | {fmt_mean_std(c.demeaned_pearson)} | "
                       f"{fmt_mean_std(c.avg_rank)} | {fmt_mean_std(c.top1_acc)} |")
        out.append("")
        # asymmetry call-out
        fcsc = d[(d.input_set == "FC") & (d.target == "SC")].demeaned_pearson.mean()
        scfc = d[(d.input_set == "SC") & (d.target == "FC")].demeaned_pearson.mean()
        if scfc and not np.isnan(scfc) and scfc != 0:
            out.append(f"**Asymmetry FC→SC / SC→FC = {fcsc/scfc:.2f}×** "
                       f"(FC→SC={fcsc:.3f}, SC→FC={scfc:.3f}).\n")
        # oracle / Ceiling B across estimators
        orc = df[(df.parcellation == parc) & (df.input_set == "FC") & (df.target == "FC")]
        if not orc.empty:
            best = orc.groupby("estimator").demeaned_pearson.mean()
            out.append("**Ceiling B (FC→FC oracle):** " +
                       ", ".join(f"{e}={v:.3f}" for e, v in best.items()) + "\n")
    return "\n".join(out)


def downstream_section(df):
    out = ["## Downstream cognition (lift over bv+demo, mean±std; median paired-perm p)\n"]
    cog = df[df.target.isin(["CogTotal", "CogFluid", "CogCryst"])]
    br = cog[cog.estimator == "bayesian_ridge"]
    for parc in sorted(df.parcellation.unique()):
        d = br[br.parcellation == parc]
        if d.empty:
            continue
        out.append(f"### {parc} (estimator=bayesian_ridge)\n")
        out.append("| input | target | pearson | lift_over_bvdemo | median perm p | residualized_r |")
        out.append("|---|---|---|---|---|---|")
        for inp in ["bv+demo", "obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
                    "obs_FC+bv+demo", "obs_SC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo"]:
            for tgt in ["CogTotal", "CogFluid", "CogCryst"]:
                c = d[(d.input_set == inp) & (d.target == tgt)]
                if c.empty:
                    continue
                pm = c.lift_perm_p.median() if "lift_perm_p" in c else np.nan
                rr = fmt_mean_std(c.residualized_pearson) if "residualized_pearson" in c else "-"
                out.append(f"| {inp} | {tgt} | {fmt_mean_std(c.pearson)} | "
                           f"{fmt_mean_std(c.lift_over_bvdemo)} | {pm:.3g} | {rr} |")
        out.append("")
    return "\n".join(out)


def leak_section():
    f = OUTPUTS_DIR / "leak_verdict.csv"
    if not f.exists():
        return "## Leak checks\n(no leak_verdict.csv)\n"
    lk = pd.read_csv(f)
    n_fail = int((lk.verdict == "LEAK_FAIL").sum())
    n_flag = int((lk.verdict == "EXEMPT_FLAGGED").sum())
    n_exp = int((lk.verdict == "EXPECTED_SIGNAL").sum()) if "verdict" in lk else 0
    out = ["## Leak checks\n",
           f"- LEAK_FAIL (genuine: demographic-free, non-connectome input over threshold): **{n_fail}**",
           f"- EXPECTED_SIGNAL (raw connectomes predict sex/age — real biology, not a leak): {n_exp}",
           f"- EXEMPT_FLAGGED (contain bv+demo; cognition-only): {n_flag}",
           f"- ok: {int((lk.verdict=='ok').sum())}\n"]
    if n_fail:
        out.append("**⚠ NON-EXEMPT LEAK DETECTED:**")
        out.append(lk[lk.verdict == "LEAK_FAIL"][["parcellation", "input_set", "target",
                   "leak_score"]].drop_duplicates().to_string(index=False))
    return "\n".join(out)


def main():
    recon = OUTPUTS_DIR / "reconstruction.csv"
    down = OUTPUTS_DIR / "downstream.csv"
    md = ["# Reproduction Grid — Findings\n",
          "Generated by summarize.py. Metric reporting order: reconstruction leads with "
          "demeaned_r/avg_rank; downstream with lift_over_bvdemo + paired-permutation p. "
          "**Inspect the 4S456 F1–F5 cells first** (the genuinely-new cross-parcellation evidence).\n"]
    if recon.exists():
        md.append(recon_section(pd.read_csv(recon)))
    if down.exists():
        md.append(downstream_section(pd.read_csv(down)))
    md.append(leak_section())
    out = REPORTS / "reproduction_findings.md"
    out.write_text("\n".join(md))
    print(f"[summarize] wrote {out}")


if __name__ == "__main__":
    main()
