"""Cross-model composite-loss comparison (spec v2 E1.10): data and summary table for the cross-model viewer.

Reads every instance's `tables/seed_records.csv` (E1 schema: combo_id, w_<term>, seed, test_<metric>), in either layout:
    <model>/<direction>/tables/seed_records.csv     direction folders `sc2fc` / `fc2sc` (spec v2 E3 layout)
    <model>/tables/seed_records.csv                  flat instance (pre-E3); direction from its `direction` column, else
                                                     its config.yml source/target, else SC -> FC
A model with direction folders is read from those only. Stage 2 rows, default init seed. Draws test demeaned r vs
avg_rank per direction on fixed axes shared by all models of that direction, with a direction switch (the selected
model toggles carry over), per-model toggles, a hover / click panel (weights, mean ± SE, paired Δ vs the model's
MSE-only, top-1, per-seed values) and the direction's ceiling from `ceiling/*.json` (field `direction`) when present.
Adding an instance or a direction needs no code change. The page itself is built by viewers/cross_model/ (`collect()`
gives its data); this script writes the table.

    python scripts/experiments/composite_loss/compare.py     # -> tables/cross_model_summary.csv
    python scripts/experiments/composite_loss/viewers/cross_model/build_cross_model.py   # -> the viewer page
"""
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
             **{f"w_{t}": float(g[f"w_{t}"].iloc[0]) for t in TERMS},
             **{k: str(g[k].iloc[0]) for k in ("block", "loss_signature") if k in g and pd.notna(g[k].iloc[0])}}
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


def collect():
    """(per-direction summaries, ceilings, table rows, model ids); None if no instance table exists yet."""
    per_dir = {k: {} for k in DIRECTIONS}
    for model, direction, folder, from_flat in discover():
        df = load_instance(folder, direction, from_flat)
        if df is not None:
            per_dir[direction][model] = summarize(model, df)
    per_dir = {k: v for k, v in per_dir.items() if v}
    if not per_dir:
        return None
    rows = [{"direction": DIRECTIONS[k], **{a: b for a, b in r.items() if a not in ("per_seed", "block", "loss_signature")}}
            for k, models in per_dir.items() for rows in models.values() for r in rows]
    model_ids = sorted({m for models in per_dir.values() for m in models})
    return per_dir, load_ceilings(), rows, model_ids


def viewer_data():
    """(meta, sets) for viewers/cross_model: one data set per direction, models + colours in meta."""
    got = collect()
    if got is None:
        return None
    per_dir, ceilings, _, model_ids = got
    meta = {"models": [{"id": m, "label": MODEL_LABEL.get(m, m), "color": COLORS[i % len(COLORS)],
                        "note": NATIVE_NOTE.get(m, "")} for i, m in enumerate(model_ids)]}
    sets = {k: {"label": DIRECTIONS[k], **direction_view(v, ceilings.get(k))} for k, v in per_dir.items()}
    return meta, sets


def build():
    got = collect()
    if got is None:
        print("no instance tables yet")
        return 1
    per_dir, ceilings, rows, _ = got
    (HERE / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "tables" / "cross_model_summary.csv", index=False, float_format="%.6g")
    for k, models in per_dir.items():
        print(f"{DIRECTIONS[k]}: models {list(models)}; ceiling: {'yes' if k in ceilings else 'not yet'}")
    print("wrote", HERE / "tables" / "cross_model_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(build())
