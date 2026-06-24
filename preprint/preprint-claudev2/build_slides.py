#!/usr/bin/env python3
"""Auto-assemble a Beamer slideshow from figures/slides/F*_*.pdf.
One section per finding, one slide per variant — pick-and-choose deck.
Writes slides.tex (compile with latexmk -pdf slides.tex).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SL = HERE / "figures" / "slides"

# (fid prefix, section title) in deck order
FINDINGS = [
    ("F1", "F1 · Directional asymmetry  (FC$\\rightarrow$SC $>$ SC$\\rightarrow$FC)"),
    ("F2", "F2 · Anatomy$\\rightarrow$SC / demographics$\\rightarrow$FC dissociation"),
    ("F3recon", "F3 · Imputed connectomes are measurable"),
    ("F3util", "F3 · …but their utility is asymmetric"),
    ("F4", "F4 · Observed FC: crystallized-cognition signal (narrow, FDR-surviving)"),
    ("F5", "F5 · SC / imputation utility wall"),
    ("F6", "F6 · Predicted connectomes carry family signal"),
    ("F7", "F7 · Reconstruction vs identification tradeoff"),
    ("F8", "F8 · FC-predictable low-variance SC mode (component index not stable)"),
    ("F9", "F9 · Richer tractography does not help"),
    ("F10", "F10 · The ceiling is structural"),
    ("F11", "F11 · The cognition lift is multiplicity-fragile"),
    ("F12", "F12 · Bayesian corroboration"),
    ("F13", "F13 · Imputation harm / cross-modal loss"),
]

VLABEL = {
    "groupedbar": "grouped bars", "hbar": "horizontal bars", "lollipop": "lollipop",
    "dumbbell": "dumbbell (parcellation gap)", "slope": "slope across parcellations",
    "heatmap": "heatmap", "box": "box + seed dots", "ratioline": "ratio across estimators",
    "Glasser": "Glasser", "4S456": "4S456Parcels",
    "fcr2": "FC→PC predictability per PC", "auc": "sibling AUC per PC",
    "conf": "sex/volume confound per PC", "recon_bar": "→FC bars", "recon_hbar": "→FC hbar",
    "downstream": "downstream cognition", "gap": "nonlinear gap vs n",
    "linvsnon": "linear vs nonlinear vs n", "ceiling": "vs FC reliability ceiling",
    "composite": "full composite panel",
}


def variants_for(prefix):
    # exact-prefix match: files are "<prefix>_v_<name>.pdf" or "<prefix>_<name>.pdf"
    out = []
    for p in sorted(SL.glob(f"{prefix}_*.pdf")):
        stem = p.stem[len(prefix) + 1:]
        if stem.startswith("v_"):
            stem = stem[2:]
        out.append((p.name, stem))
    return out


lines = [
    r"\documentclass[aspectratio=169]{beamer}",
    r"\usetheme{default}\usecolortheme{seahorse}",
    r"\setbeamertemplate{navigation symbols}{}",
    r"\setbeamertemplate{footline}[frame number]",
    r"\usepackage{graphicx}",
    r"\graphicspath{{figures/slides/}}",
    r"\title{Findings Ledger — Figure Variants (pick-and-choose)}",
    r"\subtitle{FC--SC connectome translation · F1--F13 · both parcellations, all estimators}",
    r"\author{Adel Sahuc}",
    r"\date{}",
    r"\begin{document}",
    r"\frame{\titlepage}",
    r"\begin{frame}{How to use this deck}",
    r"\begin{itemize}",
    r"\item One section per finding (F1--F13); each slide is one chart \emph{variant} of that finding.",
    r"\item Multiple chart types per finding (bars / hbar / lollipop / dumbbell / slope / heatmap / box, plus bespoke).",
    r"\item Pick your favourites; each variant is also saved standalone in figures/slides/.",
    r"\end{itemize}",
    r"\vfill\footnotesize",
    r"\textbf{Provenance:} F1--F5, F9--F13 from the frozen reproduction-grid CSVs + sanity outputs; "
    r"F6--F8 from the separate family-mechanism pipeline (same frozen splits, adds the sibling/twin task).\\[2pt]",
    r"\textbf{Estimators:} reconstruction uses all three; for \emph{scalar} downstream targets the RBF "
    r"KernelRidge rows are \emph{a priori} ill-conditioned (1-D target, $\alpha/\gamma$ grid) and shown "
    r"only to expose that instability --- BayesianRidge is reportable.",
    r"\end{frame}",
    r"\begin{frame}{The affirmative claim, stated at its defensible width}",
    r"\begin{itemize}",
    r"\item The durable contributions are the \textbf{asymmetry (F1)} and the \textbf{baseline/ceiling "
    r"negatives (F5/F9/F10/F13)}.",
    r"\item The affirmative cognition claim is \textbf{narrow}: observed FC carries a "
    r"\textbf{crystallized}-cognition signal that survives demographic residualization and "
    r"restricted-family FDR in one cell (obs FC+bv+demo $\rightarrow$ CogCryst, $q=0.027/0.048$).",
    r"\item The broader ``FC beats the baseline on cognition'' lift is real but "
    r"\textbf{multiplicity-fragile} (0/54 survive whole-grid FDR) --- reported, not hidden (F11/F12).",
    r"\end{itemize}",
    r"\end{frame}",
]

total = 0
for prefix, title in FINDINGS:
    vs = variants_for(prefix)
    if not vs:
        continue
    lines.append(r"\section{%s}" % title.split("·")[0].strip())
    lines.append(r"\begin{frame}\centering\Large\textbf{%s}\\[4pt]\normalsize %d chart variants\end{frame}"
                 % (title, len(vs)))
    for fname, stem in vs:
        vlab = VLABEL.get(stem, stem.replace("_", " "))
        lines.append(r"\begin{frame}{%s}{%s}" % (title, vlab))
        lines.append(r"\centering\includegraphics[height=0.82\textheight,width=\linewidth,keepaspectratio]{%s}" % fname)
        lines.append(r"\end{frame}")
        total += 1

lines.append(r"\end{document}")
(HERE / "slides.tex").write_text("\n".join(lines))
print(f"wrote slides.tex with {total} variant slides across {len(FINDINGS)} findings")
