"""windowed_fc_gain gates 1-6: are FC samples built from parts of the scan (halves, quarters) usable extra supervision?

Per condition (Glasser), on the timeseries cache and the fc/ combined matrices (the training target T):
  G1  estimator at full length: segment_fc over whole runs (concat / demean_concat / zscore_concat / fisher_mean) vs T
  G2  part vs part: samples of one subject from different parts identify each other (raw and demeaned identification)
  G3  part vs target: each part sample, and the mean of a subject's parts, against the full targets
  G4  systematic part effect: |mean(part k) - mean(part l)| relative to the subjects' spread around the mean target;
      for halves also a per-edge paired t-test (BH-FDR)
  G5  the same numbers grouped by LR / RL block-timing group (config timing_group)
  G6  inclusion: full-length expected runs (all conditions) and, for tasks, the standard acquisition pattern (RL run
      first with E-Prime script 1, LR second with script 2) from HCP's session summaries
A sample takes the same part of every run (both phase-encoding directions, and for rest both sessions), each segment
demeaned, then concatenated (config estimator). References: all LR runs vs all RL runs ("direction") and, for rest,
session 1 vs session 2 ("session"). Identification follows models/eval/metrics (rows = the matched set, avg rank 1 =
own sample ranked first, chance 0.5); demeaned versions subtract each sample type's mean over the included subjects.
All subjects of a condition enter (descriptive check, no train / test split).

    python scripts/experiments/windowed_fc_gain/run_gates.py --condition wm     # -> results/wm.json, wm_*.csv
    python scripts/experiments/windowed_fc_gain/run_gates.py --report           # -> tables/, figures/, tables/summary.md
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.timeseries import TR, TimeseriesCache, cache_dir_name, part_bounds, segment_fc  # noqa: E402
from models.eval.metrics import compute_corr_matrix, corr_avg_rank, corr_topn_accuracy  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yml").read_text())
RES, TAB, FIG = HERE / "results", HERE / "tables", HERE / "figures"
EXPECTED_RUNS = {"rest": ["LR1", "RL1", "LR2", "RL2"]}
SESSIONS_CSV = RES / "scan_sessions_fmri.csv"


# ---------- HCP scan sessions (gate 6) ----------
def scan_sessions():
    """One row per (subject, condition, run) fMRI acquisition from HCP's session summary CSVs; cached (tracked) as
    results/scan_sessions_fmri.csv so the zip is needed only once."""
    if SESSIONS_CSV.exists():
        return pd.read_csv(SESSIONS_CSV, dtype={"eprime_script": str, "scan_order": str})
    rows = []
    with zipfile.ZipFile(REPO_ROOT / CFG["session_zip"]) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            d = pd.read_csv(io.BytesIO(z.read(name)), dtype=str)
            d = d[d["Scan Type"].isin(["rfMRI", "tfMRI"])]
            m = d["Scan Description"].str.extract(r"^[rt]fMRI_([A-Z]+?)(\d?)_(LR|RL)$")
            d = d[m[0].notna()]
            m = m[m[0].notna()]
            cond = m[0].str.lower()
            rows.append(pd.DataFrame({
                "subject": int(Path(name).stem), "condition": cond.values,
                "run": (m[2] + m[1].where(cond == "rest", "1")).values,
                "eprime_script": d["E-Prime Script"].values, "scan_order": d["Scan Order"].values,
                "complete": d["Scan Complete"].values, "pct_complete": d["Percent Complete"].values,
                "session_day": d["Session Day"].values, "session": d["Session"].values,
                "acq_time": d["Acquisition Time"].values}))
    out = pd.concat(rows, ignore_index=True).sort_values(["subject", "condition", "run"])
    RES.mkdir(exist_ok=True)
    out.to_csv(SESSIONS_CSV, index=False)
    return pd.read_csv(SESSIONS_CSV, dtype={"eprime_script": str, "scan_order": str})


def inclusion(ts, condition, sessions):
    """Per-subject flags; `included` = usable for part samples."""
    expected = EXPECTED_RUNS.get(condition, ["LR1", "RL1"])
    modal = ts.runs.groupby("run")["n_tr"].agg(lambda s: int(s.mode()[0]))
    ses = sessions[sessions.condition == condition].set_index(["subject", "run"])
    rows = []
    for sid in map(int, ts.subject_ids):
        t = ts.run_table(sid).set_index("run")
        runs_ok = list(t.index) == expected and all(t.loc[r, "n_tr"] == modal[r] for r in expected)
        found = [(sid, r) in ses.index for r in expected]
        get = lambda r, c: ses.loc[(sid, r), c] if (sid, r) in ses.index else None
        standard = True
        if condition != "rest":
            standard = (get("RL1", "eprime_script") == "1" and get("RL1", "scan_order") == "1"
                        and get("LR1", "eprime_script") == "2" and get("LR1", "scan_order") == "2")
        rows.append({"subject": sid, "n_runs": len(t), "runs_full_length": runs_ok, "in_sessions": all(found),
                     "sessions_complete": all(get(r, "complete") == "Y" for r in expected) if all(found) else False,
                     "standard_pattern": standard,
                     "rest2_rl_first": (get("RL2", "scan_order") == "1") if condition == "rest" else None})
    df = pd.DataFrame(rows)
    df["included"] = df.runs_full_length & df.in_sessions & (df.standard_pattern | (condition in CFG.get("include_nonstandard", [])))
    return df, modal


# ---------- metrics ----------
def ident(rows, cols, demean=False):
    """Identification of row i's own column i (models/eval/metrics conventions) + own / other correlations."""
    if demean:
        rows, cols = rows - rows.mean(0), cols - cols.mean(0)
    cc = compute_corr_matrix(rows, cols)
    n = len(cc)
    own = np.diag(cc)
    other = (cc.sum(1) - own) / (n - 1)
    return {"top1": float(corr_topn_accuracy(cc=cc)), "avg_rank": float(corr_avg_rank(cc=cc)),
            "own_r": float(np.median(own)), "other_r": float(np.median(other)), "_own": own}


def both(rows, cols):
    raw, dm = ident(rows, cols), ident(rows, cols, demean=True)
    out = {k: v for k, v in raw.items() if not k.startswith("_")}
    out.update({f"{k}_dm": v for k, v in dm.items() if not k.startswith("_")})
    return out, dm["_own"]


def mean_dicts(ds):
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def bh_fraction(p, q=0.05):
    p = np.sort(p)
    ok = p <= q * np.arange(1, len(p) + 1) / len(p)
    return float(ok.nonzero()[0].max() + 1) / len(p) if ok.any() else 0.0


# ---------- one condition ----------
def run_condition(condition, limit=None, out_dir=RES):
    from scipy import stats
    P = CFG["parcellation"]
    ts = TimeseriesCache(condition, P)
    inc, modal = inclusion(ts, condition, scan_sessions())
    fcd = Path(CFG["fc_root"]) / cache_dir_name(condition, P)
    fc_ids = np.load(fcd / "subject_ids.npy")
    fc_up = np.load(fcd / "upper_triangles.npy", mmap_mode="r")
    inc["in_fc"] = np.isin(inc.subject, fc_ids)
    inc["included"] &= inc.in_fc
    sids = inc.loc[inc.included, "subject"].to_numpy()[:limit]
    n_reg = len(ts.labels)
    iu = np.triu_indices(n_reg, k=1)
    T = np.asarray(fc_up[np.searchsorted(fc_ids, sids)], dtype=np.float32)

    runs = EXPECTED_RUNS.get(condition, ["LR1", "RL1"])
    schemes = {name: [("part", k, s["n_parts"]) for k in range(s["n_parts"])] for name, s in CFG["schemes"].items()}
    schemes["direction"] = [("runs", ["LR1", "LR2"] if condition == "rest" else ["LR1"]),
                            ("runs", ["RL1", "RL2"] if condition == "rest" else ["RL1"])]
    if condition == "rest":
        schemes["session"] = [("runs", ["LR1", "RL1"]), ("runs", ["LR2", "RL2"])]
    S = {name: [np.empty_like(T) for _ in parts] for name, parts in schemes.items()}
    g1 = {m: np.empty((len(sids), 3)) for m in CFG["gate1_methods"]}
    est = CFG["estimator"]
    for i, sid in enumerate(sids):
        x = ts.subject(sid)
        b = ts.run_bounds(sid)
        names = list(ts.run_table(sid)["run"])
        for m in g1:
            v = segment_fc(x, b, m)[iu]
            d = v - T[i]
            g1[m][i] = [np.corrcoef(v, T[i])[0, 1], np.abs(d).max(), np.abs(d).mean()]
        for name, parts in schemes.items():
            for k, spec in enumerate(parts):
                bb = part_bounds(b, spec[1], spec[2]) if spec[0] == "part" else b[[names.index(r) for r in spec[1]]]
                S[name][k][i] = segment_fc(x, bb, est)[iu]
        if i % 100 == 0:
            print(f"{condition}: {i}/{len(sids)} subjects", flush=True)

    mu_T = T.mean(0)
    dev = np.linalg.norm(T - mu_T, axis=1)
    out = {"condition": condition, "parcellation": P, "estimator": est, "n_subjects": int(len(inc)),
           "n_included": int(len(sids)), "modal_run_tr": {k: int(v) for k, v in modal.items()},
           "gate1": {m: {"median_r": float(np.median(a[:, 0])), "min_r": float(a[:, 0].min()),
                         "median_max_abs": float(np.median(a[:, 1])), "median_mean_abs": float(np.median(a[:, 2]))}
                     for m, a in g1.items()},
           "gate2": {}, "gate3": {}, "gate4": {}, "duration_s": {}}
    subj = pd.DataFrame({"subject": sids, **{f"g1_r_{m}": a[:, 0] for m, a in g1.items()}})
    for name, parts in schemes.items():
        K = len(parts)
        n_tr = sum(modal[r] // parts[0][2] for r in runs) if parts[0][0] == "part" else sum(modal[r] for r in parts[0][1])
        out["duration_s"][name] = float(n_tr * TR)
        pair, pair_own = [], []
        for k in range(K):
            for l in range(K):
                if k != l:
                    m, own = both(S[name][l], S[name][k])
                    pair.append(m)
                    pair_own.append(own)
        out["gate2"][name] = mean_dicts(pair)
        subj[f"g2_own_dm_{name}"] = np.mean(pair_own, axis=0)
        per = [both(T, S[name][k]) for k in range(K)]
        g3 = mean_dicts([m for m, _ in per])
        g3["rel_err"] = float(np.mean([np.median(np.linalg.norm(S[name][k] - T, axis=1) / dev) for k in range(K)]))
        M = np.mean(S[name], axis=0)
        mm, _ = both(T, M)
        g3.update({f"mean_of_parts_{k}": v for k, v in mm.items()})
        g3["mean_of_parts_rel_err"] = float(np.median(np.linalg.norm(M - T, axis=1) / dev))
        out["gate3"][name] = g3
        subj[f"g3_own_dm_{name}"] = np.mean([own for _, own in per], axis=0)
        mus = [s.mean(0) for s in S[name]]
        ratios = {f"{k}-{l}": float(np.linalg.norm(mus[k] - mus[l]) / np.median(dev)) for k in range(K) for l in range(k + 1, K)}
        g4 = {"max_shift_ratio": max(ratios.values()), "shift_ratios": ratios,
              "within_subject_diff_ratio": float(np.median(np.linalg.norm(S[name][0] - S[name][-1], axis=1)) / np.median(dev))}
        if K == 2:
            d = (S[name][0] - S[name][1]).astype(np.float64)
            t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))
            p = 2 * stats.t.sf(np.abs(t), df=len(d) - 1)
            g4.update({"frac_edges_fdr05": bh_fraction(p), "median_abs_dz": float(np.median(np.abs(d.mean(0) / d.std(0, ddof=1))))})
        out["gate4"][name] = g4
        print(f"{condition} {name}: {out['duration_s'][name]:.0f} s | G2 avg_rank_dm {out['gate2'][name]['avg_rank_dm']:.3f} "
              f"| G3 avg_rank_dm {g3['avg_rank_dm']:.3f} | G4 shift {g4['max_shift_ratio']:.3f}", flush=True)
    res = out_dir
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{condition}.json").write_text(json.dumps(out, indent=1))
    inc.to_csv(res / f"{condition}_inclusion.csv", index=False)
    subj.to_csv(res / f"{condition}_subjects.csv", index=False, float_format="%.5g")
    print(f"wrote results/{condition}.json, {condition}_inclusion.csv, {condition}_subjects.csv")


# ---------- report ----------
def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import run as mb  # noqa: E402  (FIG_RC: shared figure style)

    G = CFG["gates"]
    R = {c: json.loads((RES / f"{c}.json").read_text()) for c in CFG["conditions"] if (RES / f"{c}.json").exists()}
    grp = CFG["timing_group"]
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    g1 = pd.DataFrame([{"condition": c, "method": m, **v} for c, r in R.items() for m, v in r["gate1"].items()])
    rows = []
    for c, r in R.items():
        for s in r["gate2"]:
            rows.append({"condition": c, "timing_group": grp[c], "scheme": s, "duration_s": r["duration_s"][s],
                         **{f"g2_{k}": v for k, v in r["gate2"][s].items()},
                         **{f"g3_{k}": v for k, v in r["gate3"][s].items()},
                         **{f"g4_{k}": v for k, v in r["gate4"][s].items() if k != "shift_ratios"}})
    parts = pd.DataFrame(rows)
    inc = []
    for c in R:
        d = pd.read_csv(RES / f"{c}_inclusion.csv")
        inc.append({"condition": c, "n_subjects": len(d), "included": int(d.included.sum()),
                    "frac_included": d.included.mean(), "not_full_length": int((~d.runs_full_length).sum()),
                    "not_in_sessions": int((~d.in_sessions).sum()), "nonstandard_pattern": int((~d.standard_pattern).sum()),
                    "sessions_incomplete_flag": int((~d.sessions_complete).sum()),
                    "rest2_rl_first": int(d.rest2_rl_first.fillna(False).astype(bool).sum()) if c == "rest" else None})
    inc = pd.DataFrame(inc)
    g1.to_csv(TAB / "gate1_estimator.csv", index=False, float_format="%.5g")
    parts.to_csv(TAB / "gates2to5_parts.csv", index=False, float_format="%.5g")
    inc.to_csv(TAB / "gate6_inclusion.csv", index=False, float_format="%.4g")
    g5 = parts[parts.scheme.isin(["halves", "quarters", "direction"])].groupby(["timing_group", "scheme"])[
        ["g2_avg_rank_dm", "g3_avg_rank_dm", "g4_max_shift_ratio"]].mean().reset_index()
    g5.to_csv(TAB / "gate5_timing_groups.csv", index=False, float_format="%.4g")

    est = CFG["estimator"]
    h = parts[parts.scheme == "halves"].set_index("condition")
    verdict = []
    for c in R:
        r1 = R[c]["gate1"][est]["median_r"]
        v = {"condition": c, "timing_group": grp[c],
             "G1 r(est, target)": r1, "G1": r1 >= G["g1_min_r"],
             "G2 halves avg_rank_dm": h.loc[c, "g2_avg_rank_dm"], "G2": h.loc[c, "g2_avg_rank_dm"] >= G["g2_min_avg_rank_dm"],
             "G3 halves avg_rank_dm": h.loc[c, "g3_avg_rank_dm"], "G3": h.loc[c, "g3_avg_rank_dm"] >= G["g3_min_avg_rank_dm"],
             "G4 halves shift": h.loc[c, "g4_max_shift_ratio"], "G4": h.loc[c, "g4_max_shift_ratio"] <= G["g4_max_shift_ratio"],
             "G6 included": inc.set_index("condition").loc[c, "frac_included"],
             "G6": inc.set_index("condition").loc[c, "frac_included"] >= G["g6_min_included"]}
        verdict.append(v)
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(TAB / "gate_verdicts.csv", index=False, float_format="%.4g")

    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else ("pass" if x is True else "FAIL" if x is False else str(x))
    md = ["# windowed_fc_gain — gates 1-6 summary", "",
          f"Estimator `{est}`; thresholds (config, fixed before results): G1 r ≥ {G['g1_min_r']}, G2 halves demeaned avg rank ≥ "
          f"{G['g2_min_avg_rank_dm']}, G3 ≥ {G['g3_min_avg_rank_dm']}, G4 shift ratio ≤ {G['g4_max_shift_ratio']}, "
          f"G6 included ≥ {G['g6_min_included']}.", "",
          "| " + " | ".join(verdict.columns) + " |", "|" + "---|" * len(verdict.columns)]
    md += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in verdict.itertuples(index=False)]
    md += ["", "Part schemes (G2 part vs part, G3 part vs full target; demeaned identification, chance 0.5):", "",
           "| condition | scheme | s | G2 avg rank dm | G2 top-1 dm | G3 avg rank dm | G3 top-1 dm | G3 own r dm | mean of parts G3 avg rank dm | G4 shift |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in parts.itertuples(index=False):
        md.append(f"| {r.condition} | {r.scheme} | {r.duration_s:.0f} | {r.g2_avg_rank_dm:.3f} | {r.g2_top1_dm:.3f} | "
                  f"{r.g3_avg_rank_dm:.3f} | {r.g3_top1_dm:.3f} | {r.g3_own_r_dm:.3f} | {r.g3_mean_of_parts_avg_rank_dm:.3f} | "
                  f"{r.g4_max_shift_ratio:.3f} |")
    (TAB / "summary.md").write_text("\n".join(md) + "\n")

    colors = dict(zip(CFG["conditions"], ["#0F4D92", "#B64342", "#42949E", "#E3962B", "#9A4D8E", "#4D4D4D", "#8BCF8B", "#3775BA"]))
    with plt.rc_context(mb.FIG_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
        for ax, key, title in ((axes[0], "g2_own_r_dm", "Sample vs another sample (G2)"), (axes[1], "g3_own_r_dm", "Sample vs full target (G3)")):
            for c in R:
                d = parts[(parts.condition == c) & parts.scheme.isin(["quarters", "halves"])].sort_values("duration_s")
                ax.plot(d.duration_s, d[key], "-o", color=colors[c], lw=2, ms=6, label=c)
                for s, mk in (("direction", "s"), ("session", "^")):
                    e = parts[(parts.condition == c) & (parts.scheme == s)]
                    if len(e):
                        ax.plot(e.duration_s, e[key], mk, color=colors[c], ms=8, mfc="none", mew=2)
            ax.set_xscale("log")
            ax.set_xlabel("Sample length (s)")
            ax.set_title(title, fontsize=14)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Own-subject demeaned r (median)")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, title="circles: quarters → halves\n"
                       "square: LR vs RL split\ntriangle: session split", title_fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "fig_identification_vs_length.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        sch = ["halves", "quarters", "direction", "session"]
        w = 0.2
        for j, s in enumerate(sch):
            d = parts[parts.scheme == s].set_index("condition").reindex(list(R))
            ax.bar(np.arange(len(R)) + (j - 1.5) * w, d.g4_max_shift_ratio, w, label=s, color=["#0F4D92", "#3775BA", "#B64342", "#E3962B"][j])
        ax.axhline(G["g4_max_shift_ratio"], color="0.5", ls="--", lw=1.2)
        ax.set_xticks(np.arange(len(R)), list(R), rotation=30, ha="right")
        ax.set_ylabel("Systematic shift / subject spread")
        ax.set_title("G4: mean FC difference between sample types", fontsize=14)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "fig_part_shift.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print((TAB / "summary.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CFG["conditions"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, help="smoke test: first N included subjects, written to a scratch folder")
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    if a.condition:
        run_condition(a.condition, a.limit, a.out_dir)
    if a.report:
        report()


if __name__ == "__main__":
    main()
