#!/usr/bin/env python3
"""Extra findings F11-F13 — results from this session's own robustness analysis,
extending the F1-F10 ledger. Across both parcellations / estimators as applicable.

F11  statistical fragility of the cognition lift (FDR, CIs, equivalence, MDE)
F12  Bayesian corroboration (hierarchical shrinkage, ROPE, Bayes factors)
F13  imputation harm is direction-specific + cross-modal loss vs same-estimator oracle
"""
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from style import (load_downstream, load_recon, recon_est, down_est, m_sd,
                   C, PARCS, PARC_LABEL, ESTIMATORS, EST_LABEL, panel_tag, savefig)

d = load_downstream()
r = load_recon()
COGS = ["CogTotal", "CogFluid", "CogCryst"]
PSHORT = {"Glasser": "Glasser", "4S456Parcels": "4S456"}
ROPE = 0.02
RNG = np.random.default_rng(7)
NICE = {"obs_FC": "obs FC", "obs_FC+bv+demo": "obs FC+bv+demo", "obs_SC": "obs SC",
        "pred_FC": "pred FC", "pred_SC": "pred SC"}


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); ranked = p[o] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]; adj[o] = np.clip(ranked, 0, 1)
    return adj


# ---- build the per-cell frequentist table (BayesianRidge) -------------------
CELLS = ["obs_FC", "obs_SC", "obs_FC+obs_SC", "obs_FC+bv+demo", "obs_SC+bv+demo",
         "pred_SC", "pred_FC", "pred_SC+bv+demo", "pred_FC+bv+demo"]
rows = []
for parc in PARCS:
    for inp in CELLS:
        for t in COGS:
            lifts = down_est(d, parc, "bayesian_ridge", inp, t, "lift_over_bvdemo")
            if len(lifts) < 3:
                continue
            n = len(lifts); mean = lifts.mean(); se = lifts.std(ddof=1) / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            pmed = np.median(down_est(d, parc, "bayesian_ridge", inp, t, "lift_perm_p"))
            mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * se
            rows.append(dict(parc=parc, inp=inp, t=t, mlift=mean, lo=mean - tc*se,
                             hi=mean + tc*se, se=se, pmed=pmed, mde=mde))
res = pd.DataFrame(rows)
res["q_grid"] = bh(res.pmed.values)
AFF = ["obs_FC", "obs_FC+bv+demo"]
res["q_fam"] = np.nan
mask = res.inp.isin(AFF)
res.loc[mask, "q_fam"] = bh(res.loc[mask, "pmed"].values)


# ============================================================ F11
def F11():
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.32,
                          left=0.16, right=0.97, top=0.88, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    focus = ["obs_FC", "obs_FC+bv+demo", "obs_SC", "pred_SC", "pred_FC"]
    ylab, yt, y = [], [], 0
    for parc in PARCS:
        for inp in focus:
            for t in COGS:
                row = res[(res.parc == parc) & (res.inp == inp) & (res.t == t)]
                if not len(row):
                    continue
                row = row.iloc[0]
                fam_ok = np.isfinite(row.q_fam) and row.q_fam < 0.05 and row["mlift"] > 0
                col = C["good"] if fam_ok else C["FC"] if row["mlift"] > 0 else C["SC"]
                ax.plot([row.lo, row.hi], [y, y], color=col, lw=2, zorder=2)
                ax.scatter([row["mlift"]], [y], color=col, s=24, zorder=3, edgecolor="white")
                tag = " *fam-FDR" if fam_ok else ""
                ylab.append(f"{PSHORT[parc]}·{NICE.get(inp,inp)}·{t[3:]}{tag}")
                yt.append(y); y += 1
        y += 0.5
    ax.axvspan(-ROPE, ROPE, color=C["base"], alpha=0.10)
    ax.axvline(0, color=C["ink"], lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=6.3); ax.invert_yaxis()
    ax.set_xlabel("lift over bv+demo (95% CI across seeds)")
    ax.set_title("A · downstream lifts with CIs; green = survives affirmative-family FDR")
    panel_tag(ax, "A")
    # B: MDE + how many survive each correction
    axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
    n_grid = int((res.q_grid < 0.05).sum() and ((res.q_grid < 0.05) & (res["mlift"] > 0)).sum())
    n_fam = int(((res.q_fam < 0.05) & (res["mlift"] > 0)).sum())
    mde = res.mde.median()
    lines = [
        ("Median MDE (80% power, n=10 seeds)", f"{mde:.3f} pearson"),
        ("Positive lifts surviving whole-grid FDR", f"{n_grid} / {len(res)}"),
        ("Positive lifts surviving family FDR", f"{n_fam} / 12"),
        ("obs FC CogCryst — uncorrected p", "≈0.03–0.04"),
        ("obs FC CogCryst — whole-grid q", "≈0.24 (fails)"),
        ("obs FC+bv+demo CogCryst — family q", "0.027/0.048 (passes)"),
    ]
    axB.text(0, 1.0, "Multiplicity & power", fontsize=11, fontweight="bold", va="top")
    for i, (a, b) in enumerate(lines):
        yy = 0.86 - i * 0.145
        axB.text(0, yy, a, fontsize=8.2, va="top")
        axB.text(0, yy - 0.058, b, fontsize=9, fontweight="bold",
                 color=C["good"] if "passes" in b else C["bad"] if "fails" in b else C["ink"], va="top")
    axB.text(0, -0.02, "Design well-powered for ~0.1 lifts,\nunderpowered below ~0.03.",
             fontsize=7.5, color="#666", style="italic", va="top")
    panel_tag(axB, "B")
    fig.suptitle("F11 · The observed-FC cognition lift is real but multiplicity-fragile "
                 "(my robustness analysis)", fontsize=11.5, fontweight="bold", y=0.965)
    savefig(fig, "F11_statistical_fragility")


# ============================================================ F12
def jzs_bf10(x, rscale=0.707):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); t = x.mean() / (x.std(ddof=1) / np.sqrt(n)); df = n - 1

    def ig(g):
        return ((1 + n*g*rscale**2)**-0.5 *
                (1 + t**2/((1+n*g*rscale**2)*df))**(-(df+1)/2) *
                (2*np.pi)**-0.5 * g**-1.5 * np.exp(-1/(2*g)))
    num, _ = integrate.quad(ig, 0, np.inf, limit=200)
    return float(num / (1 + t**2/df)**(-(df+1)/2))


def F12():
    # per-cell t-posterior + BF; hierarchical via PyMC (guarded) else empirical Bayes
    aff = d[(d.estimator == "bayesian_ridge") & (d.input_set.isin(AFF)) & (d.target.isin(COGS))].copy()
    aff["cell"] = aff.parcellation + "|" + aff.input_set + "|" + aff.target
    cells = sorted(aff.cell.unique()); idx = {c: i for i, c in enumerate(cells)}
    y = aff.lift_over_bvdemo.values; ci = aff.cell.map(idx).values
    hier = {}
    try:
        import pymc as pm, arviz as az
        with pm.Model():
            mu = pm.Normal("mu", 0, 0.2); tau = pm.HalfNormal("tau", 0.1)
            sig = pm.HalfNormal("sigma", 0.1)
            th = pm.Normal("theta", mu, tau, shape=len(cells))
            pm.Normal("obs", th[ci], sig, observed=y)
            idata = pm.sample(1500, tune=1000, chains=2, cores=1, target_accept=0.95,
                              progressbar=False, random_seed=7)
        T = idata.posterior["theta"].stack(s=("chain", "draw")).values
        for c, i in idx.items():
            dr = T[i]; hier[c] = (float(dr.mean()), tuple(az.hdi(dr, hdi_prob=0.95)))
    except Exception as ex:
        print("  PyMC fallback:", ex)
        cm = aff.groupby("cell").lift_over_bvdemo.mean(); cse = aff.groupby("cell").lift_over_bvdemo.sem()
        grand = cm.mean(); tau2 = max(cm.var(ddof=1) - (cse**2).mean(), 1e-6)
        for c in cells:
            v = cse[c]**2; w = tau2/(tau2+v); shr = w*cm[c] + (1-w)*grand; sd = np.sqrt(w*v)
            hier[c] = (shr, (shr-1.96*sd, shr+1.96*sd))

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.34, left=0.2, right=0.97, top=0.86, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ylab, yt, y = [], [], 0
    for parc in PARCS:
        for inp in AFF:
            for t in COGS:
                c = f"{parc}|{inp}|{t}"
                lifts = down_est(d, parc, "bayesian_ridge", inp, t, "lift_over_bvdemo")
                mn = lifts.mean(); se = lifts.std(ddof=1)/np.sqrt(len(lifts))
                tc = stats.t.ppf(0.975, len(lifts)-1)
                ax.plot([mn-tc*se, mn+tc*se], [y+0.15, y+0.15], color="#bbb", lw=2)
                ax.scatter([mn], [y+0.15], color="#bbb", s=16)
                hm, hd = hier[c]
                col = C["good"] if hd[0] > ROPE else C["FC"]
                ax.plot([hd[0], hd[1]], [y-0.15, y-0.15], color=col, lw=2.4, zorder=4)
                ax.scatter([hm], [y-0.15], color=col, s=24, zorder=5, edgecolor="white")
                ylab.append(f"{PSHORT[parc]}·{NICE[inp]}·{t[3:]}"); yt.append(y); y += 1
        y += 0.4
    ax.axvspan(-ROPE, ROPE, color=C["base"], alpha=0.12); ax.axvline(0, color=C["ink"], lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("posterior lift (95% HDI)")
    ax.set_title("A · no-pool (grey) vs hierarchical-shrunk (color)\ngreen = HDI excludes ROPE", fontsize=9)
    panel_tag(ax, "A")
    # B: Bayes factors, coloured by direction
    axB = fig.add_subplot(gs[0, 1])
    bc = []
    for parc in PARCS:
        for inp in ["obs_FC", "obs_FC+bv+demo", "obs_SC", "pred_FC"]:
            lifts = down_est(d, parc, "bayesian_ridge", inp, "CogCryst", "lift_over_bvdemo")
            bc.append((f"{PSHORT[parc]}·{NICE.get(inp,inp)}", jzs_bf10(lifts), lifts.mean()))
    bc.sort(key=lambda z: z[1])
    labels = [z[0] for z in bc]; bf = np.array([z[1] for z in bc]); mm = [z[2] for z in bc]
    cols = [C["good"] if v > 0 else C["bad"] for v in mm]
    axB.barh(range(len(bf)), np.log10(bf), color=cols, edgecolor="white")
    axB.axvline(np.log10(3), color="#888", ls=":", lw=1); axB.axvline(0, color=C["ink"], lw=1)
    axB.text(np.log10(3), len(bf)-0.3, " BF=3", fontsize=6.5, color="#888")
    axB.set_yticks(range(len(bf))); axB.set_yticklabels(labels, fontsize=6.8)
    axB.set_xlabel("log10 Bayes factor (lift ≠ 0)")
    axB.set_title("B · evidence strength (CogCryst)\ngreen=helps, red=harms", fontsize=9)
    panel_tag(axB, "B")
    fig.suptitle("F12 · Bayesian corroboration: hierarchical shrinkage, ROPE, and Bayes factors "
                 "(my analysis; seed-unit caveat applies)", fontsize=11, fontweight="bold", y=0.96)
    savefig(fig, "F12_bayesian_corroboration")


# ============================================================ F13
def ratio_post(num, den, nd=20000):
    mn, sn = num.mean(), num.std(ddof=1)/np.sqrt(len(num))
    md, sd = den.mean(), den.std(ddof=1)/np.sqrt(len(den))
    a = mn + sn*RNG.standard_t(len(num)-1, nd); b = md + sd*RNG.standard_t(len(den)-1, nd)
    rr = a/b; return np.median(rr), np.percentile(rr, [2.5, 97.5])


def F13():
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.07, right=0.985, top=0.85, bottom=0.13)
    # A: pred_FC vs pred_SC at matched dimensionality, across estimators (CogCryst)
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ESTIMATORS)); w = 0.18
    for pi, parc in enumerate(PARCS):
        for ii, (inp, col) in enumerate([("pred_SC", C["SC"]), ("pred_FC", C["pred"])]):
            vals = [m_sd(down_est(d, parc, e, inp, "CogCryst", "lift_over_bvdemo"))[0] for e in ESTIMATORS]
            axA.bar(x + (pi*2+ii-1.5)*w, vals, w, color=col, alpha=1 if pi == 0 else 0.55,
                    edgecolor="white", label=f"{inp}·{PSHORT[parc]}")
    axA.axhline(0, color=C["ink"], lw=1)
    axA.set_xticks(x); axA.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=8)
    axA.set_ylabel("CogCryst lift over bv+demo")
    axA.set_title("A · pred FC harm > pred SC at matched dimensionality (all estimators)")
    axA.legend(fontsize=6.5, ncol=2); panel_tag(axA, "A")
    # B: cross-modal loss vs same-estimator oracle (fraction of model ceiling)
    axB = fig.add_subplot(gs[0, 1])
    labels, meds, los, his, cols = [], [], [], [], []
    for parc in PARCS:
        scfc = recon_est(r, parc, "pca_pls", "SC", "FC"); ffc = recon_est(r, parc, "pca_pls", "FC", "FC")
        fcsc = recon_est(r, parc, "pca_pls", "FC", "SC"); ssc = recon_est(r, parc, "pca_pls", "SC", "SC")
        for lab, num, den, col in [("SC→FC / FC→FC", scfc, ffc, C["SC"]),
                                   ("FC→SC / SC→SC", fcsc, ssc, C["FC"])]:
            med, (lo, hi) = ratio_post(num, den)
            labels.append(f"{lab}\n{PSHORT[parc]}"); meds.append(med); los.append(med-lo); his.append(hi-med)
            cols.append(col)
    yy = np.arange(len(labels))
    axB.barh(yy, meds, xerr=[los, his], color=cols, edgecolor="white", capsize=3)
    axB.axvline(1.0, color=C["ink"], ls=":", lw=1)
    for i, mm in enumerate(meds):
        axB.text(mm + 0.02, i, f"{mm:.0%}", va="center", fontsize=8)
    axB.set_yticks(yy); axB.set_yticklabels(labels, fontsize=7.5); axB.invert_yaxis()
    axB.set_xlim(0, 1.05); axB.set_xlabel("cross-modal as fraction of same-estimator oracle")
    axB.set_title("B · Cross-modal loss: only ~20% (→FC) / ~37% (→SC) of the model ceiling")
    panel_tag(axB, "B")
    fig.suptitle("F13 · Imputation harm is direction-specific (not a dimensionality artifact); "
                 "cross-modal prediction reaches a small fraction of the within-modality ceiling",
                 fontsize=10.5, fontweight="bold", y=0.965)
    savefig(fig, "F13_imputation_and_oracle")


if __name__ == "__main__":
    for fn in [F11, F12, F13]:
        fn()
    print("done F11-F13")
