"""Cross-model composite-loss comparison (spec v2 E1.10): one self-contained interactive HTML over every instance.

Reads every instance's `tables/seed_records.csv` (E1 schema: combo_id, w_<term>, seed, test_<metric>), in either layout:
    <model>/<direction>/tables/seed_records.csv     direction folders `sc2fc` / `fc2sc` (spec v2 E3 layout)
    <model>/tables/seed_records.csv                  flat instance (pre-E3); direction from its `direction` column, else
                                                     its config.yml source/target, else SC -> FC
A model with direction folders is read from those only. Stage 2 rows, default init seed. Draws test demeaned r vs
avg_rank per direction on fixed axes shared by all models of that direction, with a direction switch (the selected
model toggles carry over), per-model toggles, a hover / click panel (weights, mean ± SE, paired Δ vs the model's
MSE-only, top-1, per-seed values) and the direction's ceiling from `ceiling/*.json` (field `direction`) when present.
Adding an instance or a direction needs no code change.

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


DIRECTIONS = {"sc2fc": "SC → FC", "fc2sc": "FC → SC"}


def direction_key(value):
    """'SC->FC', 'sc2fc', 'SC→FC', ('SC', 'FC') ... -> 'sc2fc' / 'fc2sc' (None if not one of the two)."""
    if isinstance(value, (tuple, list)):
        value = f"{value[0]}2{value[1]}"
    v = str(value).replace(" ", "").replace("->", "2").replace("→", "2").replace("_TO_", "2").lower()
    return v if v in DIRECTIONS else None


def discover():
    """[(model, direction, folder, from_flat)] for every instance table, direction folders preferred over flat."""
    found = []
    for d in sorted(p for p in HERE.iterdir() if p.is_dir()):
        nested = [d / k for k in DIRECTIONS if (d / k / "tables" / "seed_records.csv").exists()]
        if nested:
            found += [(d.name, n.name, n, False) for n in nested]
        elif (d / "tables" / "seed_records.csv").exists():
            found += [(d.name, k, d, True) for k in DIRECTIONS]
    return found


def flat_default_direction(d):
    cfg = d / "config.yml"
    if cfg.exists():
        import yaml
        c = yaml.safe_load(cfg.read_text()) or {}
        if "source" in c and "target" in c:
            return direction_key((c["source"], c["target"]))
    return "sc2fc"


def load_instance(d, direction, from_flat=False):
    path = d / "tables" / "seed_records.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "stage" in df:
        df = df[df["stage"] == "stage2"]
    if "direction" in df:
        df = df[df["direction"].map(direction_key) == direction]
    elif from_flat and flat_default_direction(d) != direction:
        return None
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


def load_ceilings():
    """{direction: ceiling dict} from ceiling/*.json (each has a `direction`; files without one are SC -> FC)."""
    out = {}
    for p in sorted((HERE / "ceiling").glob("*.json")):
        c = json.loads(p.read_text())
        k = direction_key(c.get("direction", "SC->FC"))
        if k and "demeaned_pearson_mean" in c and "avg_rank_mean" in c:
            out[k] = c
    return out


def direction_view(models, ceiling):
    flat = [r for rows in models.values() for r in rows]
    # fixed axes per direction: every model of this direction and its ceiling, so toggling never rescales
    xs = [r["avg_rank_mean"] for r in flat] + ([ceiling["avg_rank_mean"]] if ceiling else [])
    ys = [r["demeaned_pearson_mean"] for r in flat] + ([ceiling["demeaned_pearson_mean"]] if ceiling else [])
    x0, x1 = min(0.5, min(xs) - 0.02), min(1.0, max(xs) + 0.02)
    y0, y1 = min(0.0, min(ys) - 0.005), max(ys) * 1.06
    mx = [r["avg_rank_mean"] for r in flat]; my = [r["demeaned_pearson_mean"] for r in flat]
    zx, zy = (max(mx) - min(mx)) * 0.06 + 1e-3, (max(my) - min(my)) * 0.08 + 1e-3
    return {"zoom": [min(mx) - zx, max(mx) + zx, min(my) - zy, max(my) + zy], "axes": [x0, x1, y0, y1],
            "ceiling": ceiling, "points": {k: v for k, v in models.items()}}


def build():
    per_dir = {k: {} for k in DIRECTIONS}
    for model, direction, folder, from_flat in discover():
        df = load_instance(folder, direction, from_flat)
        if df is not None:
            per_dir[direction][model] = summarize(model, df)
    per_dir = {k: v for k, v in per_dir.items() if v}
    if not per_dir:
        print("no instance tables yet")
        return 1
    ceilings = load_ceilings()
    rows = [{"direction": DIRECTIONS[k], **{a: b for a, b in r.items() if a != "per_seed"}}
            for k, models in per_dir.items() for rows in models.values() for r in rows]
    (HERE / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "tables" / "cross_model_summary.csv", index=False, float_format="%.6g")
    model_ids = sorted({m for models in per_dir.values() for m in models})
    data = {"directions": {k: {"label": DIRECTIONS[k], **direction_view(v, ceilings.get(k))} for k, v in per_dir.items()},
            "models": [{"id": m, "label": MODEL_LABEL.get(m, m), "color": COLORS[i % len(COLORS)],
                        "note": NATIVE_NOTE.get(m, "")} for i, m in enumerate(model_ids)]}
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    (out / "cross_model_interactive.html").write_text(PAGE.replace("__DATA__", json.dumps(data, sort_keys=True)))
    for k, models in per_dir.items():
        print(f"{DIRECTIONS[k]}: models {list(models)}; ceiling: {'yes' if k in ceilings else 'not yet'}")
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
#toggles label,#dirs label{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;cursor:pointer}
#dirs{margin:0 0 8px;font-weight:600}
.sw{width:12px;height:12px;border-radius:50%;display:inline-block}
#panel{background:var(--card);border-radius:8px;padding:12px;min-height:220px}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:2px 6px;text-align:left;border-bottom:1px solid var(--grid)}
small{color:var(--muted)}
</style></head><body>
<h1>Composite losses across models (<span id="dirlab"></span>, test split)</h1>
<div id="dirs" role="radiogroup" aria-label="Prediction direction"></div>
<p class="sub">Mean over seeds 0–4 per loss combination. Axes are fixed across models within a direction (toggling a model never rescales them); each direction has its own axes. Ringed points are each model's MSE-only fit; the dashed lines mark the direction's ceiling when one is available.</p>
<div id="toggles"></div>
<main><div><svg id="plot" viewBox="0 0 760 560" role="img" aria-label="Test demeaned correlation against average rank, per model and loss combination"></svg></div>
<div id="panel"><em>Hover or click a point to see its loss weights, paired change from that model's MSE-only fit, and per-seed results.</em></div></main>
<script>
const DATA=__DATA__; const W=760,H=560,L=70,B=56,T=16,R=16; let x0,x1,y0,y1; let zoomed=false;
const dirKeys=['sc2fc','fc2sc'].filter(k=>DATA.directions[k]); let dir=dirKeys[0];
try{const s=localStorage.getItem('cl_dir'); if(s&&DATA.directions[s]) dir=s}catch(e){}
let D=null;
function useDir(k){dir=k; const v=DATA.directions[k];
 D={axes:v.axes, zoom:v.zoom, ceiling:v.ceiling, models:DATA.models.filter(m=>v.points[m.id]).map(m=>({...m, points:v.points[m.id]}))};
 document.getElementById('dirlab').textContent=v.label;}
const sx=v=>L+(v-x0)/(x1-x0)*(W-L-R), sy=v=>H-B-(v-y0)/(y1-y0)*(H-B-T);
const on={}; DATA.models.forEach(m=>on[m.id]=true);
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
  o+=`<text x="${cx-10}" y="${cy-10}" text-anchor="end">${c.label||'test-retest ceiling'}</text>`;}
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
 if(el.dataset.c){const c=D.ceiling;P.innerHTML=`<h3>${c.label||'Test-retest ceiling'}</h3><p>${c.description||"Session-1 FC predicted from the same subject's session-2 FC (no model), on the same splits. Subjects with both sessions only."}</p>
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
function buildToggles(){tg.innerHTML='';
 D.models.forEach(m=>{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" ${on[m.id]?'checked':''}><span class="sw" style="background:${m.color}"></span>${m.label} <small>(${m.points.length})</small>`;
  l.querySelector('input').addEventListener('change',e=>{on[m.id]=e.target.checked;try{localStorage.setItem('cl_models',JSON.stringify(on))}catch(_){};render()});tg.appendChild(l)});
 const zl=document.createElement('label');zl.innerHTML=`<input type="checkbox" ${zoomed?'checked':''}> Zoom to the models (axes fit all models, ceiling hidden)`;
 zl.querySelector('input').addEventListener('change',e=>{zoomed=e.target.checked;render()});tg.appendChild(zl);}
const dg=document.getElementById('dirs');
dirKeys.forEach(k=>{const l=document.createElement('label');l.innerHTML=`<input type="radio" name="dir" value="${k}" ${k===dir?'checked':''}> ${DATA.directions[k].label}`;
 l.querySelector('input').addEventListener('change',()=>{useDir(k);try{localStorage.setItem('cl_dir',k)}catch(_){};buildToggles();render();
  document.getElementById('panel').innerHTML='<em>Hover or click a point.</em>'});dg.appendChild(l)});
useDir(dir); buildToggles();
render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(build())
