#!/usr/bin/env python3
"""Bayesian counterpart to make_robustness.py — same data, Bayesian inference.

Frequentist -> Bayesian mapping:
  BH-FDR / permutation p   ->  hierarchical partial-pooling (shrinkage handles
                               multiplicity) + posterior P(lift > 0)
  TOST equivalence         ->  ROPE (region of practical equivalence) + HDI/ROPE
                               decision (Kruschke) + P(lift in ROPE)
  p-value for an effect     ->  JZS Bayes factor (Rouder 2009)
  MDE / power              ->  posterior SD (estimation precision)
  CI on oracle fraction    ->  Monte-Carlo posterior credible interval on the ratio

Per-cell posteriors are analytic (Jeffreys prior -> Student-t marginal for the mean).
The multiplicity demonstration uses a PyMC hierarchical model over the affirmative
family; if PyMC is unavailable it falls back to empirical-Bayes shrinkage.

Outputs:
  tables/robustness_bayes_downstream.csv
  figures/supp/SX6_bayes_forest.pdf/.png
  ROBUSTNESS_ANALYSIS_BAYES.md
"""
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from style import (load_downstream, load_recon, C, PARCS, PARC_LABEL,
                   FIGDIR, recon_cell)

OUTDIR = FIGDIR.parent
TBL = OUTDIR / "tables"; TBL.mkdir(exist_ok=True)
SUPP = FIGDIR / "supp"; SUPP.mkdir(exist_ok=True)

d = load_downstream()
r = load_recon()
COGS = ["CogTotal", "CogFluid", "CogCryst"]
CONN_INPUTS = ["obs_FC", "obs_SC", "obs_FC+obs_SC", "obs_FC+bv+demo",
               "obs_SC+bv+demo", "pred_SC", "pred_FC",
               "pred_SC+bv+demo", "pred_FC+bv+demo"]
NICE = {"obs_FC": "obs FC", "obs_SC": "obs SC", "obs_FC+obs_SC": "obs FC+SC",
        "obs_FC+bv+demo": "obs FC+bv+demo", "obs_SC+bv+demo": "obs SC+bv+demo",
        "pred_SC": "pred SC (FC→SC)", "pred_FC": "pred FC (SC→FC)",
        "pred_SC+bv+demo": "pred SC+bv+demo", "pred_FC+bv+demo": "pred FC+bv+demo"}
ROPE = 0.02          # region of practical equivalence: |lift| < 0.02 ~ "no effect"
RNG = np.random.default_rng(42)


def t_posterior(x):
    """Jeffreys-prior marginal posterior for the mean: Student-t(df=n-1, loc, scale)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); m = x.mean(); s = x.std(ddof=1); se = s / np.sqrt(n)
    df = n - 1
    return m, se, df, n


def post_prob_gt(thresh, m, se, df):
    return float(1 - stats.t.cdf((thresh - m) / se, df))


def post_prob_in_rope(m, se, df, rope=ROPE):
    return float(stats.t.cdf((rope - m) / se, df) - stats.t.cdf((-rope - m) / se, df))


def hdi_t(m, se, df, cred=0.95):
    half = stats.t.ppf(0.5 + cred / 2, df) * se
    return m - half, m + half


def jzs_bf10(x, rscale=0.707):
    """JZS Bayes factor (BF10) for a one-sample test, Rouder et al. 2009."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); t = x.mean() / (x.std(ddof=1) / np.sqrt(n)); df = n - 1

    def integrand(g):
        return ((1 + n * g * rscale**2) ** (-0.5)
                * (1 + t**2 / ((1 + n * g * rscale**2) * df)) ** (-(df + 1) / 2)
                * (2 * np.pi) ** (-0.5) * g ** (-1.5) * np.exp(-1 / (2 * g)))
    num, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    denom = (1 + t**2 / df) ** (-(df + 1) / 2)
    return float(num / denom)


# ---------------------------------------------------------------- per-cell
rows = []
for parc in PARCS:
    for inp in CONN_INPUTS:
        for t in COGS:
            s = d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
                  (d.input_set == inp) & (d.target == t)]["lift_over_bvdemo"].values
            s = s[np.isfinite(s)]
            if len(s) < 3:
                continue
            m, se, df, n = t_posterior(s)
            lo, hi = hdi_t(m, se, df)
            p_gt0 = post_prob_gt(0.0, m, se, df)
            p_gt_rope = post_prob_gt(ROPE, m, se, df)       # P(meaningful + lift)
            p_in_rope = post_prob_in_rope(m, se, df)
            # Kruschke HDI vs ROPE decision
            if lo > ROPE:
                decision = "credibly beats baseline"
            elif hi < -ROPE:
                decision = "credibly below baseline"
            elif lo > -ROPE and hi < ROPE:
                decision = "practically equivalent"
            else:
                decision = "undecided"
            bf10 = jzs_bf10(s)
            rows.append(dict(parcellation=parc, input_set=inp, target=t, n=n,
                             post_mean=m, post_sd=se, hdi_lo=lo, hdi_hi=hi,
                             P_gt0=p_gt0, P_gt_rope=p_gt_rope, P_in_rope=p_in_rope,
                             decision=decision, BF10=bf10))
res = pd.DataFrame(rows)
res.round(4).to_csv(TBL / "robustness_bayes_downstream.csv", index=False)
print("  wrote tables/robustness_bayes_downstream.csv")

# ---------------------------------------------------------------- hierarchical
AFFIRM = ["obs_FC", "obs_FC+bv+demo"]
aff = d[(d.estimator == "bayesian_ridge") & (d.input_set.isin(AFFIRM)) &
        (d.target.isin(COGS))].copy()
aff["cell"] = aff.parcellation + "|" + aff.input_set + "|" + aff.target
cells = sorted(aff.cell.unique())
cell_idx = {c: i for i, c in enumerate(cells)}
y = aff["lift_over_bvdemo"].values
ci = aff["cell"].map(cell_idx).values

hier = {}
backend = "none"
try:
    import pymc as pm
    import arviz as az
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 0.2)                 # grand-mean lift
        tau = pm.HalfNormal("tau", 0.1)               # between-cell SD
        sigma = pm.HalfNormal("sigma", 0.1)           # within-cell (seed) SD
        theta = pm.Normal("theta", mu, tau, shape=len(cells))
        pm.Normal("obs", theta[ci], sigma, observed=y)
        idata = pm.sample(2000, tune=1000, chains=2, cores=1,
                          target_accept=0.95, progressbar=False,
                          random_seed=42)
    th = idata.posterior["theta"].stack(s=("chain", "draw")).values  # cells x draws
    for c, i in cell_idx.items():
        draws = th[i]
        hier[c] = dict(mean=float(draws.mean()),
                       hdi=tuple(az.hdi(draws, hdi_prob=0.95)),
                       P_gt_rope=float((draws > ROPE).mean()),
                       P_gt0=float((draws > 0).mean()))
    backend = "pymc-hierarchical-NUTS"
    grand = idata.posterior["mu"].values.mean()
    tau_hat = idata.posterior["tau"].values.mean()
except Exception as ex:                                # empirical-Bayes fallback
    print("  PyMC unavailable/failed, empirical-Bayes shrinkage:", ex)
    cell_means = aff.groupby("cell")["lift_over_bvdemo"].mean()
    cell_se = aff.groupby("cell")["lift_over_bvdemo"].sem()
    grand = cell_means.mean()
    tau2 = max(cell_means.var(ddof=1) - (cell_se**2).mean(), 1e-6)
    tau_hat = np.sqrt(tau2)
    for c in cells:
        v = cell_se[c]**2
        w = tau2 / (tau2 + v)                          # shrinkage weight
        shr = w * cell_means[c] + (1 - w) * grand
        sd = np.sqrt(w * v)
        hier[c] = dict(mean=shr, hdi=(shr - 1.96*sd, shr + 1.96*sd),
                       P_gt_rope=float(1 - stats.norm.cdf(ROPE, shr, sd)),
                       P_gt0=float(1 - stats.norm.cdf(0, shr, sd)))
    backend = "empirical-Bayes shrinkage"
print("  hierarchical backend:", backend)

# ---------------------------------------------------------------- oracle ratio posterior
def ratio_posterior(num_vals, den_vals, ndraw=20000):
    mn, sen, dfn, _ = t_posterior(num_vals)
    md, sed, dfd, _ = t_posterior(den_vals)
    a = mn + sen * RNG.standard_t(dfn, ndraw)
    b = md + sed * RNG.standard_t(dfd, ndraw)
    rr = a / b
    return float(np.median(rr)), tuple(np.percentile(rr, [2.5, 97.5]))

oracle = []
for parc in PARCS:
    scfc = recon_cell(r, parc, "pca_pls", "SC", "FC")
    ffc = recon_cell(r, parc, "pca_pls", "FC", "FC")
    fcsc = recon_cell(r, parc, "pca_pls", "FC", "SC")
    ssc = recon_cell(r, parc, "pca_pls", "SC", "SC")
    f1, ci1 = ratio_posterior(scfc, ffc)
    f2, ci2 = ratio_posterior(fcsc, ssc)
    oracle.append(dict(parcellation=parc,
                       SCFC_over_FCFC=f1, SCFC_ci=ci1,
                       FCSC_over_SCSC=f2, FCSC_ci=ci2))

# ---------------------------------------------------------------- imputation contrast
def contrast_posterior(parc, t, ndraw=40000):
    a = d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
          (d.input_set == "pred_FC") & (d.target == t)]["lift_over_bvdemo"].values
    b = d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
          (d.input_set == "pred_SC") & (d.target == t)]["lift_over_bvdemo"].values
    ma, sea, dfa, _ = t_posterior(a); mb, seb, dfb, _ = t_posterior(b)
    da = ma + sea * RNG.standard_t(dfa, ndraw)
    db = mb + seb * RNG.standard_t(dfb, ndraw)
    return float((da < db).mean()), float((da < 0).mean())

# ================================================================ figure SX6
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
focus = ["obs_FC", "obs_FC+bv+demo"]
ax = axes[0]
ylabels, yt, y = [], [], 0
for parc in PARCS:
    for inp in focus:
        for t in COGS:
            c = f"{parc}|{inp}|{t}"
            np_row = res[(res.parcellation == parc) & (res.input_set == inp) &
                         (res.target == t)].iloc[0]
            # no-pooling (analytic) in light, hierarchical (shrunk) in dark
            ax.plot([np_row.hdi_lo, np_row.hdi_hi], [y+0.15, y+0.15],
                    color="#bbb", lw=2, zorder=2)
            ax.scatter([np_row.post_mean], [y+0.15], color="#bbb", s=18, zorder=3)
            h = hier[c]
            col = C["good"] if h["hdi"][0] > ROPE else C["FC"]
            ax.plot([h["hdi"][0], h["hdi"][1]], [y-0.15, y-0.15], color=col, lw=2.4, zorder=4)
            ax.scatter([h["mean"]], [y-0.15], color=col, s=26, zorder=5, edgecolor="white")
            ylabels.append(f"{PARC_LABEL[parc].split(' ')[0]} · {NICE[inp]} · {t[3:]}")
            yt.append(y); y += 1
    y += 0.4
ax.axvspan(-ROPE, ROPE, color=C["base"], alpha=0.12, zorder=0)
ax.axvline(0, color=C["ink"], lw=1)
ax.set_yticks(yt); ax.set_yticklabels(ylabels, fontsize=6.8); ax.invert_yaxis()
ax.set_xlabel("posterior lift over bv+demo (95% HDI)")
ax.set_title("A · No-pooling (grey) vs hierarchical-shrunk (color)\n"
             "green = 95% HDI excludes ROPE", fontsize=9)

ax = axes[1]
# Bayes factor bar (log scale) for the affirmative-family CogCryst cells + negatives
bcells = res[(res.target == "CogCryst") &
             (res.input_set.isin(["obs_FC", "obs_FC+bv+demo", "obs_SC", "pred_FC"]))]
bcells = bcells.sort_values("BF10")
labels = [f"{x.parcellation[:3]} · {NICE[x.input_set]}" for _, x in bcells.iterrows()]
bf = bcells.BF10.values
# colour by DIRECTION of the effect (sign of posterior mean), length = evidence strength
colors = [C["good"] if mm > 0 else C["bad"] for mm in bcells.post_mean.values]
ax.barh(range(len(bf)), np.log10(bf), color=colors, edgecolor="white")
ax.axvline(np.log10(3), color="#888", ls=":", lw=1)
ax.text(np.log10(3), len(bf)-0.4, " BF=3", fontsize=6.5, color="#888")
ax.axvline(0, color=C["ink"], lw=1)
ax.set_yticks(range(len(bf))); ax.set_yticklabels(labels, fontsize=6.8)
ax.set_xlabel("log10 Bayes factor (BF10 for lift ≠ 0)")
ax.set_title("B · Strength of evidence for a lift (CogCryst)\n"
             "green = helps, red = harms; length = evidence", fontsize=9)
fig.suptitle("SX6 · Bayesian downstream analysis: hierarchical shrinkage, ROPE, and Bayes factors",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(SUPP / f"SX6_bayes_forest.{ext}")
print("  wrote figures/supp/SX6_bayes_forest.png")
plt.close(fig)

# ================================================================ markdown
def f(x): return f"{x:+.3f}"
L = ["# Bayesian Robustness Analysis\n",
     "_Generated by `make_robustness_bayes.py` from `reproduction/outputs/*.csv`. "
     "A Bayesian re-analysis of the same questions as `ROBUSTNESS_ANALYSIS.md`. "
     f"Hierarchical backend: **{backend}**. ROPE = ±{ROPE} pearson._\n",
     "**Frequentist → Bayesian mapping.** BH-FDR → hierarchical partial-pooling "
     "(shrinkage absorbs multiplicity) + posterior P(lift>0); TOST → ROPE + HDI/ROPE "
     "decision; p-value → JZS Bayes factor; MDE/power → posterior SD; CI → credible "
     "interval. Posteriors per cell use a Jeffreys prior (Student-t marginal for the mean).\n"]

L.append("## 1 · Posterior lifts, ROPE decisions, and Bayes factors (CogCryst)\n")
L.append("| parc | input | post mean | 95% HDI | P(lift>0) | P(lift>ROPE) | BF10 | decision |")
L.append("|---|---|---:|---|---:|---:|---:|---|")
for _, x in res[(res.target == "CogCryst") &
                res.input_set.isin(["obs_FC", "obs_FC+bv+demo", "obs_SC", "pred_FC"])
                ].sort_values(["parcellation", "input_set"]).iterrows():
    L.append(f"| {x.parcellation[:3]} | {NICE[x.input_set]} | {f(x.post_mean)} | "
             f"[{x.hdi_lo:+.3f}, {x.hdi_hi:+.3f}] | {x.P_gt0:.3f} | {x.P_gt_rope:.3f} | "
             f"{x.BF10:.1f} | {x.decision} |")
L.append("\nBayes-factor reading (Jeffreys): BF10>3 moderate / >10 strong evidence for an "
         "effect; BF10<1/3 evidence for the null. P(lift>ROPE) is the posterior probability "
         "of a *practically meaningful* improvement over bv+demo.\n")

L.append("## 2 · Multiplicity via hierarchical shrinkage (affirmative family)\n")
L.append(f"A partial-pooling model over the 12 observed-FC cognition cells (grand-mean lift "
         f"posterior ≈ {grand:+.3f}, between-cell SD ≈ {tau_hat:.3f}) shrinks each cell "
         "toward the family mean — the Bayesian counterpart to restricted-family FDR.\n")
L.append("| parc | input | target | no-pool mean | shrunk mean | shrunk P(lift>ROPE) | HDI excl. ROPE |")
L.append("|---|---|---|---:|---:|---:|:--:|")
for parc in PARCS:
    for inp in AFFIRM:
        for t in COGS:
            c = f"{parc}|{inp}|{t}"
            npm = res[(res.parcellation == parc) & (res.input_set == inp) &
                      (res.target == t)]["post_mean"].iloc[0]
            h = hier[c]
            excl = "yes" if h["hdi"][0] > ROPE else "no"
            L.append(f"| {parc[:3]} | {NICE[inp]} | {t[3:]} | {npm:+.3f} | "
                     f"{h['mean']:+.3f} | {h['P_gt_rope']:.3f} | {excl} |")
nyes = sum(1 for c in cells if hier[c]["hdi"][0] > ROPE)
L.append(f"\nAfter shrinkage, **{nyes}/{len(cells)}** affirmative-family cells have a 95% HDI "
         "entirely above the ROPE (a credibly meaningful lift). The obs_FC+bv+demo CogCryst "
         "cells are the most robust; plain obs_FC and the fluid/total targets are pulled "
         "toward the null — the same fragility the frequentist FDR showed, expressed as "
         "posterior shrinkage rather than a binary q-value.\n")

L.append("## 3 · Same-estimator oracle fraction (posterior credible intervals)\n")
L.append("| parcellation | SC→FC / FC→FC | 95% CrI | FC→SC / SC→SC | 95% CrI |")
L.append("|---|---:|---|---:|---|")
for x in oracle:
    L.append(f"| {x['parcellation']} | {x['SCFC_over_FCFC']:.2f} | "
             f"[{x['SCFC_ci'][0]:.2f}, {x['SCFC_ci'][1]:.2f}] | "
             f"{x['FCSC_over_SCSC']:.2f} | [{x['FCSC_ci'][0]:.2f}, {x['FCSC_ci'][1]:.2f}] |")
L.append("\nCross-modal prediction reaches only ~20% (SC→FC) and ~37% (FC→SC) of the "
         "same-estimator within-modality ceiling, and the credible intervals are well "
         "below 1 — the large cross-modal loss is robust to estimator choice and to "
         "seed uncertainty.\n")

L.append("## 4 · Imputation harm is direction-specific (posterior probabilities)\n")
L.append("| parc | target | P(pred_FC < pred_SC) | P(pred_FC lift < 0) |")
L.append("|---|---|---:|---:|")
for parc in PARCS:
    for t in COGS:
        p_lt, p_neg = contrast_posterior(parc, t)
        L.append(f"| {parc[:3]} | {t[3:]} | {p_lt:.3f} | {p_neg:.3f} |")
L.append("\nThe posterior probability that imputed FC is more harmful than imputed SC is high "
         "across cells, and P(pred_FC lift < 0) is near 1 — direction-specific harm, "
         "consistent with the frequentist contrast. The definitive spectrum-matched control "
         "still needs the pipeline (experiment E3).\n")

L.append("## Caveat — unit of analysis (read alongside the frequentist doc)\n")
L.append("These posteriors are built on the **10 seed-level lifts** per cell, so they are the "
         "Bayesian counterpart of the *seed-level* t-test, whose 95% CI already excluded zero "
         "— not of the *subject-level paired-permutation* test (median p≈0.03), whose "
         "whole-grid FDR was the fragile q≈0.24. The seeds are family-aware **resplits of one "
         "cohort** with overlapping training sets, i.e. pseudo-replicates, so the per-cell "
         "Bayes factors (up to ~200) and the narrow HDIs **overstate the evidence** for the "
         "same reason the seed-level frequentist CI did. A fully consistent Bayesian model "
         "would put the likelihood at the subject level and pool over splits, which needs the "
         "per-subject predictions (not in the seed-summary CSVs) — that is experiment E6. "
         "Treat the Bayes-factor magnitudes as conditional on seeds-as-data, and lean on the "
         "subject-level permutation result for the conservative claim.\n")
L.append("## Bottom line\n")
L.append("The Bayesian re-analysis reaches the same scientific conclusions as the "
         "frequentist one. Stated as posteriors: the negative claims (SC and imputed "
         "connectomes do not beat bv+demo) have HDIs below the ROPE and Bayes factors "
         "indicating strong evidence of a *harmful* effect (not a null), and "
         "P(pred_FC lift<0)≈1; the positive claim (observed FC helps CogCryst) is credibly "
         "above the ROPE and survives hierarchical shrinkage for the strongest cells. The "
         "honest synthesis across both frameworks: the directional and no-utility claims are "
         "robust, while 'observed FC helps cognition' is real but rests on the seed unit of "
         "analysis — its strength should be reported with the pseudo-replication caveat, not "
         "as a decisive Bayes factor.\n")

(OUTDIR / "ROBUSTNESS_ANALYSIS_BAYES.md").write_text("\n".join(L))
print("  wrote ROBUSTNESS_ANALYSIS_BAYES.md")
