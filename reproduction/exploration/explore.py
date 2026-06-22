#!/usr/bin/env python3
"""Deep exploration of the reproduction grid CSVs.

Goes beyond summarize.py: cross-parcellation consistency, estimator comparison,
seed stability, HP-flatness, oracle-ceiling structure, and anomaly hunting.
Writes a machine-readable digest to exploration/digest.txt and figures to figures/.
Run from reproduction/:  python exploration/explore.py
"""
from pathlib import Path
import pandas as pd, numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
DIG = HERE / "digest.txt"

r = pd.read_csv(OUT / "reconstruction.csv")
d = pd.read_csv(OUT / "downstream.csv")
l = pd.read_csv(OUT / "leak_verdict.csv")
PARCS = sorted(r.parcellation.unique())
lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)


def ms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) == 0 else (x.mean(), x.std())


# ---------------------------------------------------------------- 1. estimator comparison (oracle + cross-modal)
p("=" * 78); p("1. ESTIMATOR COMPARISON — within-modal oracle (Ceiling B) and cross-modal")
for parc in PARCS:
    p(f"\n  [{parc}] demeaned_pearson by estimator:")
    for tgtpair, lab in [(("FC", "FC"), "FC->FC oracle"), (("SC", "SC"), "SC->SC oracle"),
                         (("FC", "SC"), "FC->SC x-modal"), (("SC", "FC"), "SC->FC x-modal")]:
        row = []
        for est in ["pca_pls", "bayesian_ridge", "kernel_ridge"]:
            sub = r[(r.parcellation == parc) & (r.estimator == est) &
                    (r.input_set == tgtpair[0]) & (r.target == tgtpair[1])]
            m, s = ms(sub.demeaned_pearson)
            row.append(f"{est.split('_')[0]:>6}={m:.3f}")
        p(f"    {lab:16s} " + "  ".join(row))

# ---------------------------------------------------------------- 2. KR 3x3 HP flatness
p("\n" + "=" * 78); p("2. KERNEL_RIDGE 3x3 HP SENSITIVITY (does gamma/alpha matter?)")
kr = r[r.estimator == "kernel_ridge"]
for parc in PARCS:
    for pair in [("FC", "SC"), ("FC", "FC")]:
        sub = kr[(kr.parcellation == parc) & (kr.input_set == pair[0]) & (kr.target == pair[1])]
        piv = sub.groupby(["hp_gamma_mult", "hp_alpha"]).demeaned_pearson.mean().unstack()
        spread = sub.groupby("seed").demeaned_pearson.agg(lambda x: x.max() - x.min()).mean()
        p(f"\n  [{parc}] {pair[0]}->{pair[1]}  within-seed max-min spread over 9 HPs (mean over seeds) = {spread:.4f}")
        p("    grid (rows=gamma_mult, cols=alpha):"); p("    " + piv.round(3).to_string().replace("\n", "\n    "))

# ---------------------------------------------------------------- 3. cross-parcellation consistency
p("\n" + "=" * 78); p("3. CROSS-PARCELLATION CONSISTENCY (Glasser vs 4S456, pca_pls recon)")
pls = r[r.estimator == "pca_pls"]
pairs = [("FC", "SC"), ("SC", "FC"), ("bv", "SC"), ("bv", "FC"), ("demo", "SC"), ("demo", "FC"),
         ("bv+demo", "SC"), ("bv+demo", "FC"), ("FC+bv+demo", "SC"), ("SC+bv+demo", "FC")]
p(f"  {'pair':16s} {'Glasser':>10s} {'4S456':>10s} {'Δ':>8s} {'%diff':>8s}")
for a, b in pairs:
    g = ms(pls[(pls.parcellation == "Glasser") & (pls.input_set == a) & (pls.target == b)].demeaned_pearson)[0]
    s = ms(pls[(pls.parcellation == "4S456Parcels") & (pls.input_set == a) & (pls.target == b)].demeaned_pearson)[0]
    p(f"  {a+'->'+b:16s} {g:10.3f} {s:10.3f} {s-g:8.3f} {100*(s-g)/g:7.1f}%")

# ---------------------------------------------------------------- 4. dissociation F2 (bv=anatomy vs demo)
p("\n" + "=" * 78); p("4. F2 DISSOCIATION — bv(anatomy) vs demo(demographics) into each modality")
for parc in PARCS:
    p(f"\n  [{parc}]")
    for tgt in ["SC", "FC"]:
        bv = ms(pls[(pls.parcellation == parc) & (pls.input_set == "bv") & (pls.target == tgt)].demeaned_pearson)[0]
        de = ms(pls[(pls.parcellation == parc) & (pls.input_set == "demo") & (pls.target == tgt)].demeaned_pearson)[0]
        p(f"    ->{tgt}: bv={bv:.3f}  demo={de:.3f}  ratio bv/demo={bv/de:.2f}")

# ---------------------------------------------------------------- 5. does connectome add over bv+demo (recon side)?
p("\n" + "=" * 78); p("5. CONNECTOME OVER bv+demo (reconstruction) — does adding FC/SC help predict the other?")
for parc in PARCS:
    for src, tgt in [("FC", "SC"), ("SC", "FC")]:
        base = ms(pls[(pls.parcellation == parc) & (pls.input_set == "bv+demo") & (pls.target == tgt)].demeaned_pearson)[0]
        comb = ms(pls[(pls.parcellation == parc) & (pls.input_set == f"{src}+bv+demo") & (pls.target == tgt)].demeaned_pearson)[0]
        alone = ms(pls[(pls.parcellation == parc) & (pls.input_set == src) & (pls.target == tgt)].demeaned_pearson)[0]
        p(f"  [{parc}] ->{tgt}: bv+demo={base:.3f}  {src}+bv+demo={comb:.3f} (Δ={comb-base:+.3f})  {src} alone={alone:.3f}")

# ---------------------------------------------------------------- 6. downstream: full lift table w/ significance, both estimators
p("\n" + "=" * 78); p("6. DOWNSTREAM lift_over_bvdemo (bayesian_ridge), cognition, with sig fraction")
br = d[(d.estimator == "bayesian_ridge")]
cog = ["CogTotal", "CogFluid", "CogCryst"]
for parc in PARCS:
    p(f"\n  [{parc}]  (lift mean±std | frac seeds perm_p<.05 | median p)")
    for inp in ["obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
                "obs_FC+bv+demo", "pred_SC+bv+demo"]:
        cells = []
        for t in cog:
            sub = br[(br.parcellation == parc) & (br.input_set == inp) & (br.target == t)]
            m, s = ms(sub.lift_over_bvdemo)
            fsig = (sub.lift_perm_p < 0.05).mean()
            medp = sub.lift_perm_p.median()
            cells.append(f"{t[3:]:>5}:{m:+.3f}±{s:.3f}[{fsig:.0%},p={medp:.3f}]")
        p(f"    {inp:18s} " + " ".join(cells))

# ---------------------------------------------------------------- 7. estimator effect on downstream
p("\n" + "=" * 78); p("7. DOES DOWNSTREAM ESTIMATOR MATTER? obs_FC->CogCryst lift by estimator")
for parc in PARCS:
    row = []
    for est in ["pca_pls", "bayesian_ridge", "kernel_ridge"]:
        sub = d[(d.parcellation == parc) & (d.estimator == est) & (d.input_set == "obs_FC") & (d.target == "CogCryst")]
        m, _ = ms(sub.lift_over_bvdemo)
        row.append(f"{est.split('_')[0]}={m:+.3f}")
    p(f"  [{parc}] " + "  ".join(row))

# ---------------------------------------------------------------- 8. sex/age (leak targets) — biology signal strength
p("\n" + "=" * 78); p("8. SEX/AGE prediction (leak-checks, but = biology signal). bayesian_ridge")
for parc in PARCS:
    p(f"\n  [{parc}]")
    for inp in ["bv+demo", "obs_FC", "obs_SC", "pred_FC", "pred_SC"]:
        sx = ms(br[(br.parcellation == parc) & (br.input_set == inp) & (br.target == "sex")].balanced_acc)[0]
        ag = ms(br[(br.parcellation == parc) & (br.input_set == inp) & (br.target == "age")].pearson)[0]
        p(f"    {inp:12s} sex bal_acc={sx:.3f}  age pearson={ag:.3f}")

# ---------------------------------------------------------------- 9. identifiability structure (avg_rank / top1)
p("\n" + "=" * 78); p("9. IDENTIFIABILITY (avg_rank, top1_acc) — recon pca_pls")
for parc in PARCS:
    p(f"\n  [{parc}] (avg_rank / top1)")
    for a, b in [("FC", "SC"), ("SC", "FC"), ("bv+demo", "SC"), ("FC+bv+demo", "SC"), ("FC", "FC")]:
        sub = pls[(pls.parcellation == parc) & (pls.input_set == a) & (pls.target == b)]
        ar, _ = ms(sub.avg_rank); t1, _ = ms(sub.top1_acc)
        p(f"    {a+'->'+b:16s} avg_rank={ar:.3f}  top1={t1:.3f}")

# ---------------------------------------------------------------- 10. ANOMALY HUNT
p("\n" + "=" * 78); p("10. ANOMALY HUNT")
# a) where does demo BEAT bv (demographics beat anatomy)?
p("\n  (a) demo > bv (demographics beat anatomy proxies) cases [pca_pls]:")
for parc in PARCS:
    for tgt in ["SC", "FC"]:
        bv = ms(pls[(pls.parcellation == parc) & (pls.input_set == "bv") & (pls.target == tgt)].demeaned_pearson)[0]
        de = ms(pls[(pls.parcellation == parc) & (pls.input_set == "demo") & (pls.target == tgt)].demeaned_pearson)[0]
        flag = " <-- demo wins" if de > bv else ""
        p(f"    [{parc}] ->{tgt}: bv={bv:.3f} demo={de:.3f}{flag}")
# b) pred_FC actively HARMFUL downstream (lift well below 0)?
p("\n  (b) most-negative downstream lifts (pred_* hurting):")
harm = br[br.input_set.isin(["pred_FC", "pred_SC", "obs_SC"]) & br.target.isin(cog)]
g = harm.groupby(["parcellation", "input_set", "target"]).lift_over_bvdemo.mean().sort_values().head(8)
for (parc, inp, t), v in g.items():
    p(f"    {parc:13s} {inp:8s} {t:9s} lift={v:+.3f}")
# c) seed-fragile findings: largest std relative to mean among headline recon cells
p("\n  (c) seed stability (CV = std/|mean|) of headline recon cells [pca_pls]:")
for parc in PARCS:
    for a, b in [("FC", "SC"), ("SC", "FC"), ("FC", "FC"), ("SC", "SC")]:
        sub = pls[(pls.parcellation == parc) & (pls.input_set == a) & (pls.target == b)].demeaned_pearson
        m, s = ms(sub)
        p(f"    [{parc}] {a+'->'+b:8s} mean={m:.3f} std={s:.3f} CV={s/abs(m):.2%}")
# d) oracle gap: how far is FC->SC from the FC->FC ceiling? (fraction of ceiling achieved)
p("\n  (d) fraction of within-modal oracle (Ceiling B, BR) reached by cross-modal (pca_pls):")
for parc in PARCS:
    cb = ms(r[(r.parcellation == parc) & (r.estimator == "bayesian_ridge") & (r.input_set == "SC") & (r.target == "SC")].demeaned_pearson)[0]
    xm = ms(pls[(pls.parcellation == parc) & (pls.input_set == "FC") & (pls.target == "SC")].demeaned_pearson)[0]
    p(f"    [{parc}] FC->SC {xm:.3f} / SC->SC oracle {cb:.3f} = {xm/cb:.1%} of ceiling")

# ---------------------------------------------------------------- 11. leak verdict breakdown
p("\n" + "=" * 78); p("11. LEAK VERDICTS")
p("  " + str(l.verdict.value_counts().to_dict()))
ef = l[l.verdict == "EXPECTED_SIGNAL"]
p("  EXPECTED_SIGNAL rows (raw connectome predicts sex/age):")
for _, x in ef.iterrows():
    p(f"    {x.parcellation:13s} {x.input_set:8s} {x.target:5s} score={x.leak_score:.3f} thr={x.threshold}")

DIG.write_text("\n".join(lines) + "\n")
p("\n[written] " + str(DIG))
PY = None
