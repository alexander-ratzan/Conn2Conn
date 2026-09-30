"""Search-space audit from collected Tune trials: sweep inventory, budget convergence, hyperparameter effects.

Input: the table written by collect_trials.py. Output (tracked, small CSVs) in audit/tables/:
    inventory.csv        sweeps / trials / seeds / time per trial / best val, per model × source × month
    convergence.csv      median best-so-far val (and gap to the sweep's final best) after k trials, per model
    hparam_effects.csv   per model × hyperparameter: categorical value medians (Kruskal-Wallis p) or continuous
                         Spearman rho and where the top-10% trials sit in the sampled range
    reference.csv        best-per-sweep val for reference models (e.g. the linear family)

Scope and caveats (also in multimodel_scfc.md): SC source only for the effect tables; val = last reported
`val_demeaned_r` of each trial (ASHA-stopped trials report fewer epochs); TPE-sampled trials are not uniform,
so effects are marginal and optimistic; for a model whose search space changed, only the dominant schema is used.

Run (kraken_env, CPU):
    python scripts/experiments/multimodel_scfc/audit/analyze_search_spaces.py \
        --models Sarwar2020MLP Chen2024GCN NodalGNN NodalMLP --reference CrossModal_PCA_PLS_learnable CrossModal_PCA_PLS
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

AUDIT_DIR = Path(__file__).resolve().parent
KS = [2, 4, 8, 12, 16, 24, 32]
LOSS_KEYS = ("loss_",)  # legacy loss hparams are reported but flagged


def dominant_schema(g):
    top = g["keyset"].value_counts().index[0]
    return g[g["keyset"] == top], top


def inventory(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["created"]).dt.strftime("%Y-%m")
    maxep = pd.to_numeric(d.get("p.max_epochs"), errors="coerce")
    d["completed"] = d["n_epochs"] >= maxep.fillna(np.inf) * 0.98
    return (d.groupby(["model", "source", "month"], dropna=False)
             .agg(sweeps=("sweep", "nunique"), trials=("trial", "size"),
                  seeds=("seed", lambda s: " ".join(str(int(x)) for x in sorted(set(s.dropna())))),
                  schemas=("keyset", "nunique"), completed_frac=("completed", "mean"),
                  median_min_per_trial=("time_s", lambda s: np.nanmedian(s) / 60),
                  best_val=("val_last", "max"), median_val=("val_last", "median"))
             .reset_index())


def convergence(sc):
    rows = []
    for model, g in sc.groupby("model"):
        g, _ = dominant_schema(g)
        per = []
        for _, s in g.groupby("sweep"):
            best = s.sort_values("trial_index")["val_last"].cummax().to_numpy()
            per.append({"n": len(best), "final": best[-1], **{k: (best[k - 1] if len(best) >= k else np.nan) for k in KS}})
        p = pd.DataFrame(per)
        seed_best = g.groupby("seed")["val_last"].max()
        row = {"model": model, "sweeps": len(p), "trials_per_sweep_median": p["n"].median(),
               "sweeps_ge16": int((p["n"] >= 16).sum()), "final_best_median": p["final"].median(),
               "final_best_q25": p["final"].quantile(0.25), "final_best_q75": p["final"].quantile(0.75),
               "across_seed_sd": seed_best.std()}
        long = p[p["n"] >= 16]
        for k in KS:
            row[f"gap_at_{k}"] = float(np.nanmedian(long["final"] - long[k])) if len(long) and long[k].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hparam_effects(sc):
    rows = []
    for model, g in sc.groupby("model"):
        g, schema = dominant_schema(g)
        y = g["val_last"].astype(float)
        top = y >= y.quantile(0.9)
        for col in sorted(c for c in g.columns if c.startswith("p.") and g[c].notna().any()):
            x, name = g[col], col[2:]
            xn = pd.to_numeric(x, errors="coerce")
            uniq = x.dropna().unique()
            base = {"model": model, "hparam": name, "n": int(x.notna().sum()), "legacy_loss_key": name.startswith(LOSS_KEYS)}
            if xn.notna().all() and len(uniq) > 8:
                rho, p = spearmanr(xn, y)
                lo, hi = float(xn.min()), float(xn.max())
                log = lo > 0 and hi / lo > 50
                pos = (lambda v: np.log(v / lo) / np.log(hi / lo)) if log else (lambda v: (v - lo) / (hi - lo))
                q = xn[top].quantile([0.1, 0.5, 0.9])
                rows.append({**base, "kind": "continuous", "p_value": p, "spearman_rho": rho, "range_lo": lo,
                             "range_hi": hi, "log_range": log, "top10_q10": q.iloc[0], "top10_q50": q.iloc[1],
                             "top10_q90": q.iloc[2], "top10_relpos_lo": pos(q.iloc[0]), "top10_relpos_hi": pos(q.iloc[2])})
            else:
                groups = {v: y[x == v] for v in uniq}
                testable = [s for s in groups.values() if len(s) > 1]
                p = kruskal(*testable).pvalue if len(testable) > 1 else np.nan
                ranked = sorted(groups.items(), key=lambda kv: -kv[1].median())
                rows.append({**base, "kind": "categorical", "p_value": p,
                             "values_by_median": " | ".join(f"{v}: n={len(s)} med={s.median():.4f} top10={top[x == v].mean():.0%}" for v, s in ranked)})
    return pd.DataFrame(rows)


def reference(df, models):
    rows = []
    for model in models:
        g = df[(df["model"] == model) & (df["source"] == "SC") & df["val_last"].notna()]
        if g.empty:
            continue
        best = g.groupby("sweep")["val_last"].max()
        rows.append({"model": model, "sweeps": len(best), "best_per_sweep_median": best.median(),
                     "best_q25": best.quantile(0.25), "best_q75": best.quantile(0.75), "trial_median": g["val_last"].median()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", default=str(AUDIT_DIR / "data" / "trials.csv.gz"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--out-dir", default=str(AUDIT_DIR / "tables"))
    args = ap.parse_args()
    df = pd.read_csv(args.trials, low_memory=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audited = df[df["model"].isin(args.models)]
    sc = audited[(audited["source"] == "SC") & audited["val_last"].notna()]
    tables = {"inventory": inventory(audited), "convergence": convergence(sc), "hparam_effects": hparam_effects(sc),
              "reference": reference(df, args.reference)}
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
        print(f"{name}: {len(t)} rows -> {out / (name + '.csv')}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tables["convergence"].round(4).to_string(index=False))
        print(tables["reference"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
