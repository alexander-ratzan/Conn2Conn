"""Cross-model composite-loss comparison (spec v2 E1.10): one self-contained interactive HTML over every instance.

Reads each instance's `tables/seed_records.csv` (E1 schema: combo_id, w_<term>, seed, test_<metric>; the Krakencoder
instance adds `direction`), keeps SC -> FC Stage 2 rows, and draws test demeaned r vs avg_rank on fixed axes shared by
all models, with per-model toggles, a hover / click panel (weights, mean ± SE, paired Δ vs the model's MSE-only,
top-1, per-seed values) and the test-retest ceiling from `ceiling/test_retest.json` when present.
Adding an instance needs no code change: any `<instance>/tables/seed_records.csv` is picked up.

    python scripts/experiments/composite_loss/compare.py     # -> figures/cross_model_interactive.html, tables/cross_model_summary.csv
"""
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TERMS = ("varmatch", "correye", "correye_dm", "neidist")
TERM_LABEL = {"varmatch": "Var-match", "correye": "Corr-eye", "correye_dm": "Demeaned corr-eye", "neidist": "Neighbor dist"}
MODEL_LABEL = {"linear_backbone": "Linear backbone", "pca_pls_learnable": "PCA/PLS learnable",
               "pca_pls_covprojector": "PCA/PLS CovProjector (all covariates)", "krakencoder": "Krakencoder"}
# Krakencoder's native correye acts in a mean-centred PCA space, i.e. it corresponds to our Demeaned corr-eye (D5).
NATIVE_NOTE = {"krakencoder": "Krakencoder weights are its native loss weights (its corr-eye ≈ Demeaned corr-eye, D5); "
                              "every cell also keeps its fixed MSE / latent terms."}
COLORS = ["#0F4D92", "#B64342", "#42949E", "#E3962B", "#9A4D8E", "#4D4D4D", "#8BCF8B"]
METRICS = ("demeaned_pearson", "avg_rank", "top1_acc", "mse")


def _se(s):
    s = s.dropna()
    return float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0


def load_instance(d):
    path = d / "tables" / "seed_records.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "stage" in df:
        df = df[df["stage"] == "stage2"]
    if "direction" in df:
        keep = df["direction"].astype(str).str.replace(" ", "").str.upper()
        df = df[keep.isin({"SC->FC", "SC_TO_FC", "SCFC", "SC2FC", "SC→FC"})]
    if "random_seed" in df:  # Krakencoder noise reruns use other init seeds; the grid uses the default one
        df = df[df["random_seed"] == df["random_seed"].min()]
    if df.empty or "test_demeaned_pearson" not in df:
        return None
    for t in TERMS:
        if f"w_{t}" not in df:
            df[f"w_{t}"] = 0.0
    return df


def label(row):
    parts = [f"{TERM_LABEL[t]} {row[f'w_{t}']:g}" for t in TERMS if row[f"w_{t}"] > 0]
    if row["combo_id"] == "kraken_default":
        return "Krakencoder paper default (Corr-eye 1 + Neighbor dist 1)"
    return "MSE only" if not parts else "MSE + " + " + ".join(parts)


def summarize(name, df):
    base = df[df["combo_id"] == "mse_only"].set_index("seed")
    rows = []
    for cid, g in df.groupby("combo_id", sort=False):
        r = {"model": name, "combo_id": cid, "n_seeds": int(g["seed"].nunique()),
             **{f"w_{t}": float(g[f"w_{t}"].iloc[0]) for t in TERMS}}
        for m in METRICS:
            col = f"test_{m}"
            if col in g:
                r[f"{m}_mean"], r[f"{m}_se"] = float(g[col].mean()), _se(g[col])
                if len(base) and col in base:
                    d = g.set_index("seed")[col] - base[col].reindex(g["seed"].values).values
                    r[f"d_{m}_mean"], r[f"d_{m}_se"] = float(d.mean()), _se(d)
        r["label"] = label(r)
        r["per_seed"] = [{"seed": int(s.seed), **{m: round(float(getattr(s, f"test_{m}")), 4) for m in METRICS
                                                  if f"test_{m}" in g}} for s in g.sort_values("seed").itertuples()]
        rows.append(r)
    return rows


def build():
    models = {}
    for d in sorted(p for p in HERE.iterdir() if p.is_dir()):
        df = load_instance(d)
        if df is not None:
            models[d.name] = summarize(d.name, df)
    if not models:
        print("no instance tables yet")
        return 1
    ceiling = None
    cj = HERE / "ceiling" / "test_retest.json"
    if cj.exists():
        ceiling = json.loads(cj.read_text())
    flat = [r for rows in models.values() for r in rows]
    (HERE / "tables").mkdir(exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "per_seed"} for r in flat]).to_csv(
        HERE / "tables" / "cross_model_summary.csv", index=False, float_format="%.6g")
    # fixed axes: every model and the ceiling, so toggling never rescales
    xs = [r["avg_rank_mean"] for r in flat] + ([ceiling["avg_rank_mean"]] if ceiling else [])
    ys = [r["demeaned_pearson_mean"] for r in flat] + ([ceiling["demeaned_pearson_mean"]] if ceiling else [])
    x0, x1 = min(0.5, min(xs) - 0.02), min(1.0, max(xs) + 0.02)
    y0, y1 = min(0.0, min(ys) - 0.005), max(ys) * 1.06
    mx = [r["avg_rank_mean"] for r in flat]; my = [r["demeaned_pearson_mean"] for r in flat]
    zx, zy = (max(mx) - min(mx)) * 0.06 + 1e-3, (max(my) - min(my)) * 0.08 + 1e-3
    zoom = [min(mx) - zx, max(mx) + zx, min(my) - zy, max(my) + zy]
    data = {"zoom": zoom, "models": [{"id": k, "label": MODEL_LABEL.get(k, k), "color": COLORS[i % len(COLORS)],
                        "note": NATIVE_NOTE.get(k, ""), "points": v} for i, (k, v) in enumerate(models.items())],
            "ceiling": ceiling, "axes": [x0, x1, y0, y1]}
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    (out / "cross_model_interactive.html").write_text(PAGE.replace("__DATA__", json.dumps(data, sort_keys=True)))
    print(f"models: {list(models)}; ceiling: {'yes' if ceiling else 'not yet'}")
    print("wrote", out / "cross_model_interactive.html", "and", HERE / "tables" / "cross_model_summary.csv")
    return 0


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Composite-loss models</title><style>
:root{--bg:#fff;--fg:#1f2328;--muted:#57606a;--card:#f6f8fa;--grid:#d0d7de}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--card:#161b22;--grid:#30363d}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 "DejaVu Sans",Helvetica,Arial,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 12px}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:16px}
@media (max-width:820px){main{grid-template-columns:1fr}}
svg{width:100%;height:auto} svg text{fill:var(--muted);font-size:11px} .lab{font-size:13px;fill:var(--fg)}
.ax{stroke:var(--muted);stroke-width:1.5} .gl{stroke:var(--grid);stroke-dasharray:3 3}
.pt{cursor:pointer} .pt:hover,.pt.sel{stroke:var(--fg);stroke-width:3}
#toggles label{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;cursor:pointer}
.sw{width:12px;height:12px;border-radius:50%;display:inline-block}
#panel{background:var(--card);border-radius:8px;padding:12px;min-height:220px}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:2px 6px;text-align:left;border-bottom:1px solid var(--grid)}
small{color:var(--muted)}
</style></head><body>
<h1>Composite losses across models (SC → FC, test split)</h1>
<p class="sub">Mean over seeds 0–4 per loss combination. Axes are fixed across models (toggling a model never rescales them). Ringed points are each model's MSE-only fit; the dashed lines mark the test-retest ceiling.</p>
<div id="toggles"></div>
<main><div><svg id="plot" viewBox="0 0 760 560" role="img" aria-label="Test demeaned correlation against average rank, per model and loss combination"></svg></div>
<div id="panel"><em>Hover or click a point to see its loss weights, paired change from that model's MSE-only fit, and per-seed results.</em></div></main>
<script>
const D=__DATA__; const W=760,H=560,L=70,B=56,T=16,R=16; let [x0,x1,y0,y1]=D.axes; let zoomed=false;
const sx=v=>L+(v-x0)/(x1-x0)*(W-L-R), sy=v=>H-B-(v-y0)/(y1-y0)*(H-B-T);
const on={}; D.models.forEach(m=>on[m.id]=true);
let saved=null; try{saved=JSON.parse(localStorage.getItem('cl_models'))}catch(e){}
if(saved) Object.keys(on).forEach(k=>{if(k in saved) on[k]=saved[k]});
const f=(v,d=4)=>v==null?'–':Number(v).toFixed(d), pm=(m,s,d=4)=>m==null?'–':`${f(m,d)} ± ${f(s,d)}`;
const sg=v=>v==null?'–':(v>=0?'+':'')+Number(v).toFixed(4);
function ticks(a,b,n){const s=(b-a)/n;return Array.from({length:n+1},(_,i)=>a+i*s)}
function render(){
 [x0,x1,y0,y1]=zoomed?D.zoom:D.axes;
 const svg=document.getElementById('plot'); let o='';
 ticks(x0,x1,5).forEach(v=>{o+=`<line class="gl" x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${H-B}"/><text x="${sx(v)}" y="${H-B+18}" text-anchor="middle">${v.toFixed(3)}</text>`});
 ticks(y0,y1,5).forEach(v=>{o+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L-8}" y="${sy(v)+4}" text-anchor="end">${v.toFixed(3)}</text>`});
 o+=`<line class="ax" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="ax" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"/>`;
 o+=`<text class="lab" x="${(W+L)/2}" y="${H-12}" text-anchor="middle">Average rank (test, max)</text>`;
 o+=`<text class="lab" transform="translate(16 ${(H-B)/2}) rotate(-90)" text-anchor="middle">Demeaned corr. (test, max)</text>`;
 if(D.ceiling&&!zoomed){const c=D.ceiling,cx=sx(c.avg_rank_mean),cy=sy(c.demeaned_pearson_mean);
  o+=`<line x1="${L}" x2="${W-R}" y1="${cy}" y2="${cy}" stroke="currentColor" stroke-dasharray="6 4" opacity=".5"/>`;
  o+=`<line x1="${cx}" x2="${cx}" y1="${T}" y2="${H-B}" stroke="currentColor" stroke-dasharray="6 4" opacity=".5"/>`;
  o+=`<rect class="pt" data-c="1" x="${cx-7}" y="${cy-7}" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"/>`;
  o+=`<text x="${cx-10}" y="${cy-10}" text-anchor="end">test-retest ceiling</text>`;}
 D.models.forEach((m,mi)=>{ if(!on[m.id]) return; m.points.forEach((p,pi)=>{
  const cx=sx(p.avg_rank_mean), cy=sy(p.demeaned_pearson_mean), base=p.combo_id==='mse_only';
  o+=`<line x1="${sx(p.avg_rank_mean-p.avg_rank_se)}" x2="${sx(p.avg_rank_mean+p.avg_rank_se)}" y1="${cy}" y2="${cy}" stroke="${m.color}" opacity=".6"/>`;
  o+=`<line x1="${cx}" x2="${cx}" y1="${sy(p.demeaned_pearson_mean-p.demeaned_pearson_se)}" y2="${sy(p.demeaned_pearson_mean+p.demeaned_pearson_se)}" stroke="${m.color}" opacity=".6"/>`;
  o+=`<circle class="pt" data-m="${mi}" data-p="${pi}" cx="${cx}" cy="${cy}" r="${base?8:5.5}" fill="${m.color}" stroke="${base?'currentColor':'none'}" stroke-width="${base?2.5:0}"/>`;
 })});
 svg.innerHTML=o;
 svg.querySelectorAll('.pt').forEach(el=>{el.addEventListener('mouseenter',()=>show(el));
  el.addEventListener('click',()=>{svg.querySelectorAll('.pt').forEach(q=>q.classList.remove('sel'));el.classList.add('sel');show(el)})});
}
function show(el){
 const P=document.getElementById('panel');
 if(el.dataset.c){const c=D.ceiling;P.innerHTML=`<h3>Test-retest ceiling</h3><p>Session-1 FC predicted from the same subject's session-2 FC (no model), on the same splits. Subjects with both sessions only.</p>
  <p>Demeaned corr.: ${pm(c.demeaned_pearson_mean,c.demeaned_pearson_se)}<br>Average rank: ${pm(c.avg_rank_mean,c.avg_rank_se)}<br>Top-1 accuracy: ${pm(c.top1_acc_mean,c.top1_acc_se)}</p>`;return}
 const m=D.models[+el.dataset.m], p=m.points[+el.dataset.p];
 const w=['varmatch','correye','correye_dm','neidist'].filter(t=>p['w_'+t]>0).map(t=>`<tr><td>${({varmatch:'Var-match',correye:'Corr-eye',correye_dm:'Demeaned corr-eye',neidist:'Neighbor dist'})[t]}</td><td>${p['w_'+t]}</td></tr>`).join('');
 const s=p.per_seed.map(r=>`<tr><td>${r.seed}</td><td>${f(r.demeaned_pearson)}</td><td>${f(r.avg_rank)}</td><td>${f(r.top1_acc)}</td></tr>`).join('');
 P.innerHTML=`<h3><span class="sw" style="background:${m.color}"></span> ${m.label}</h3><p><b>${p.label}</b><br><small>id <code>${p.combo_id}</code> · ${p.n_seeds} seeds</small></p>
 <table><tr><th>Term (MSE = 1)</th><th>Weight</th></tr>${w||'<tr><td colspan="2">MSE only</td></tr>'}</table>
 <p>Demeaned corr.: ${pm(p.demeaned_pearson_mean,p.demeaned_pearson_se)} <small>(Δ ${sg(p.d_demeaned_pearson_mean)})</small><br>
 Average rank: ${pm(p.avg_rank_mean,p.avg_rank_se)} <small>(Δ ${sg(p.d_avg_rank_mean)})</small><br>
 Top-1 accuracy: ${pm(p.top1_acc_mean,p.top1_acc_se)} <small>(Δ ${sg(p.d_top1_acc_mean)})</small></p>
 <p><small>Δ = mean per-seed change from this model's MSE-only fit on the same split.${m.note?' '+m.note:''}</small></p>
 <table><tr><th>Seed</th><th>Demeaned corr.</th><th>Avg. rank</th><th>Top-1 acc.</th></tr>${s}</table>`;
}
const tg=document.getElementById('toggles');
D.models.forEach(m=>{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" ${on[m.id]?'checked':''}><span class="sw" style="background:${m.color}"></span>${m.label} <small>(${m.points.length})</small>`;
 l.querySelector('input').addEventListener('change',e=>{on[m.id]=e.target.checked;try{localStorage.setItem('cl_models',JSON.stringify(on))}catch(_){};render()});tg.appendChild(l)});
const zl=document.createElement('label');zl.innerHTML='<input type="checkbox"> Zoom to the models (axes fit all models, ceiling hidden)';
zl.querySelector('input').addEventListener('change',e=>{zoomed=e.target.checked;render()});tg.appendChild(zl);
render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(build())
