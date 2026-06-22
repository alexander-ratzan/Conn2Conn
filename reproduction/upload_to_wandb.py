#!/usr/bin/env python3
"""Replay the grid CSVs into Weights & Biases as interactive Tables.

The full grid ran with --no-wandb (CSV is the source of truth). This logs those CSVs to W&B
WITHOUT recomputation — as wandb.Table objects (one run, filterable/plottable in the UI),
plus a few headline summary scalars. Much better than 13k individual runs.

RUN THIS FROM A MACHINE WITH A W&B API KEY (e.g. your Mac):
    pip install wandb pandas        # if needed
    wandb login                     # paste your API key
    python upload_to_wandb.py                       # online (uploads to wandb.ai)
    python upload_to_wandb.py --offline             # local only, `wandb sync` later
    python upload_to_wandb.py --project my-proj --entity my-team

It reads outputs/{reconstruction,downstream,leak_verdict}.csv relative to this file.
"""
from pathlib import Path
import argparse
import pandas as pd
import wandb

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def headline_summary(recon, down):
    """A few load-bearing scalars (mean over seeds) for the run summary / quick glance."""
    s = {}
    pls = recon[recon.estimator == "pca_pls"]
    for parc in sorted(recon.parcellation.unique()):
        d = pls[pls.parcellation == parc]
        fcsc = d[(d.input_set == "FC") & (d.target == "SC")].demeaned_pearson.mean()
        scfc = d[(d.input_set == "SC") & (d.target == "FC")].demeaned_pearson.mean()
        s[f"summary/{parc}/asymmetry_FC2SC_over_SC2FC"] = float(fcsc / scfc) if scfc else float("nan")
        s[f"summary/{parc}/FC2SC_demeaned_r"] = float(fcsc)
        s[f"summary/{parc}/SC2FC_demeaned_r"] = float(scfc)
        orc = recon[(recon.parcellation == parc) & (recon.input_set == "FC") & (recon.target == "FC")]
        s[f"summary/{parc}/ceilingB_FC2FC_oracle"] = float(
            orc[orc.estimator == "bayesian_ridge"].demeaned_pearson.mean())
    br = down[(down.estimator == "bayesian_ridge") & (down.target == "CogCryst")]
    for parc in sorted(down.parcellation.unique()):
        fc = br[(br.parcellation == parc) & (br.input_set == "obs_FC")]
        s[f"summary/{parc}/obs_FC_CogCryst_lift"] = float(fc.lift_over_bvdemo.mean())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="conn2conn-fc-to-sc-reproduction")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    recon = pd.read_csv(OUT / "reconstruction.csv")
    down = pd.read_csv(OUT / "downstream.csv")
    leak = pd.read_csv(OUT / "leak_verdict.csv")
    git_commit = str(recon["git_commit"].iloc[0]) if "git_commit" in recon else "unknown"

    run = wandb.init(
        project=args.project, entity=args.entity,
        mode="offline" if args.offline else "online",
        name=f"grid-replay-{git_commit}",
        tags=["reproduction_2026_06", "grid_replay", "from_csv"],
        config={"git_commit": git_commit, "n_recon_rows": len(recon),
                "n_downstream_rows": len(down), "source": "merged CSV (source of truth)"},
    )
    run.log({
        "reconstruction": wandb.Table(dataframe=recon),
        "downstream": wandb.Table(dataframe=down),
        "leak_verdict": wandb.Table(dataframe=leak),
    })
    run.summary.update(headline_summary(recon, down))
    run.finish()
    print(f"[wandb] logged reconstruction({len(recon)}) + downstream({len(down)}) + "
          f"leak({len(leak)}) tables to project '{args.project}' "
          f"({'offline' if args.offline else 'online'}).")
    if args.offline:
        print("[wandb] sync later with:  wandb sync wandb/offline-run-*")


if __name__ == "__main__":
    main()
