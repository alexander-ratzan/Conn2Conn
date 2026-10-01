"""
Score every saved checkpoint of a retrained Krakencoder run with Conn2Conn's metrics and loss terms.

    python -m models.architectures.krakencoder.checkpoint_eval --run-dir results/krakencoder/<tag>/seed<S>
    python -m models.architectures.krakencoder.checkpoint_eval --run-dir ... --parcellation Glasser --batch-size 64

Writes <run-dir>/epoch_history.csv: one row per (checkpoint epoch, direction) with
  - {train,val,test}_{demeaned_pearson,pearson,avg_rank,top1_acc,mse}: models/eval/metrics.compute_basic_regression_metrics,
    the same definitions as the evaluator behind every benchmark table (avg_rank / top1 on the raw correlation matrix,
    demeaned_pearson on predictions and targets minus the training mean);
  - {train,val}_loss_raw_{mse,varmatch,correye,neidist}: models/train/loss.py term functions in original edge space,
    averaged over consecutive subject batches of --batch-size (default: the run's recipe batch size), so the training
    curves are in the same units as the composite-loss experiments (E1) rather than Krakencoder's PCA space.

Predictions are made in-process from the vendored package: input adaptation (`adaptmode`), input transform, encode,
decode, inverse transform, as run_model.py does for a single input -> output path. Note: upstream
`generate_adapt_transformer` resets its subject-mask arguments, so the adaptation is fit on all subjects in run_model
and here alike (the March runs too); with our inputs the fit is near identity (R2 = 1.000). The last
checkpoint is checked against the run's run_model predictions file (per-subject r reported in the log).
Upstream's own per-epoch record (`kraken_trainrecord_*.mat`) holds total loss per path but not per term.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from models.architectures.krakencoder import _vendor_entry  # noqa: E402
from models.architectures.krakencoder.precomputed import _KRAKEN_FLAVOR_KEY, krakencoder_prediction_path  # noqa: E402
from models.eval.metrics import compute_basic_regression_metrics, compute_corr_matrix  # noqa: E402
from models.train.loss import compute_correye_loss, compute_neidist_loss, compute_var_match_loss  # noqa: E402

DIRECTIONS = (("SC", "FC"), ("FC", "SC"))
TERMS = ("mse", "varmatch", "correye", "neidist")
METRICS = ("demeaned_pearson", "pearson", "avg_rank", "top1_acc", "mse")


def log(msg: str) -> None:
    print(f"[krakencoder.checkpoint_eval] {msg}", flush=True)


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in glob.glob(str(run_dir / "kraken_chkpt_*_ep*.pt")):
        m = re.search(r"_ep(\d+)\.pt$", p)
        out.append((int(m.group(1)), Path(p)))
    stamps = {re.search(r"_(\d{8}_\d{6})_ep\d+\.pt$", p.name).group(1) for _, p in out}
    if len(stamps) > 1:
        sys.exit(f"{run_dir} holds checkpoints from {len(stamps)} training runs; expected one")
    return sorted(out)


def term_losses(pred: torch.Tensor, true: torch.Tensor, batch_size: int) -> dict:
    """Mean of each raw term over consecutive full batches (a single partial batch if the split is smaller)."""
    n = pred.shape[0]
    starts = list(range(0, n - batch_size + 1, batch_size)) or [0]
    acc = {t: [] for t in TERMS}
    for s in starts:
        p, y = pred[s:s + batch_size], true[s:s + batch_size]
        acc["mse"].append(float(torch.mean((p - y) ** 2)))
        acc["varmatch"].append(float(compute_var_match_loss(p, y)))
        acc["correye"].append(float(compute_correye_loss(p, y)))
        acc["neidist"].append(float(compute_neidist_loss(p, y)))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="results/krakencoder/<tag>/seed<S>")
    ap.add_argument("--parcellation", default="Glasser", help="evaluation parcellation (must be among the trained flavors)")
    ap.add_argument("--batch-size", type=int, help="batch size for the loss terms (default: recipe batch_size)")
    ap.add_argument("--epochs", help="comma-separated checkpoint epochs to score (default: all)")
    ap.add_argument("--out", type=Path, help="default: <run-dir>/epoch_history.csv")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    recipe, seed, tag = manifest["recipe"], int(manifest["seed"]), manifest["tag"]
    parc = args.parcellation
    if parc not in recipe["parcellations"]:
        sys.exit(f"{parc} was not trained (flavors: {recipe['parcellations']})")
    batch_size = int(args.batch_size or recipe.get("batch_size") or 41)

    kc = _vendor_entry.activate()
    from krakencoder.data import generate_adapt_transformer, load_transformers_from_file
    from krakencoder.model import Krakencoder
    from krakencoder.utils import numpyvar, torchfloat, torchint

    from data.hcp_dataset import HCP_Base

    base = HCP_Base(parcellation=parc, hemi="both", shuffle_seed=seed, source="SC", target="FC",
                    data_load_mode="precomputed")
    split = {k: np.asarray(v) for k, v in base.trainvaltest_partition_indices.items()}
    data = {"SC": np.asarray(base.sc_upper_triangles, dtype=np.float32),
            "FC": np.asarray(base.fc_upper_triangles, dtype=np.float32)}
    train_mean = {m: data[m][split["train"]].mean(axis=0) for m in data}

    ckpts = checkpoints(run_dir)
    if args.epochs:
        keep = {int(e) for e in args.epochs.split(",")}
        ckpts = [c for c in ckpts if c[0] in keep]
    if not ckpts:
        sys.exit(f"no checkpoints in {run_dir}")
    stamp = re.search(r"_(\d{8}_\d{6})_ep\d+\.pt$", ckpts[0][1].name).group(1)
    ioxfm = glob.glob(str(run_dir / f"kraken_ioxfm_*_{stamp}.npy"))
    if len(ioxfm) != 1:
        sys.exit(f"expected one kraken_ioxfm_*_{stamp}.npy in {run_dir}")
    flavors = {m: _KRAKEN_FLAVOR_KEY[m].format(parc=parc) for m in data}
    transformers, transformer_info = load_transformers_from_file(ioxfm, input_names=list(flavors.values()), quiet=True)
    # Input adaptation as run_model --adaptmode, fit on the training subjects.
    adapted = {}
    for m, x in data.items():
        adxfm = generate_adapt_transformer(input_data=x, target_data=transformer_info[flavors[m]],
                                           adapt_mode=recipe["adaptmode"])  # upstream ignores fit masks (see top)
        adapted[m] = numpyvar(adxfm.transform(x))
    log(f"{tag} seed {seed}: {len(ckpts)} checkpoints, {parc}, loss batch {batch_size}, vendor {kc.__version__}")

    rows, final_preds = [], {}
    for epoch, ckpt in ckpts:
        net, extra = Krakencoder.load_checkpoint(str(ckpt), eval_mode=True)
        names = list(extra["input_name_list"])
        enc = {m: names.index(flavors[m]) for m in data}
        net.eval()
        for src, tgt in DIRECTIONS:
            with torch.no_grad():
                x = torchfloat(transformers[flavors[src]].transform(adapted[src]))
                _, y = net(x, torchint(enc[src]), torchint(enc[tgt]))
            pred = np.asarray(numpyvar(transformers[flavors[tgt]].inverse_transform(numpyvar(y))), dtype=np.float32)
            final_preds[(src, tgt)] = pred
            row = {"model": "Krakencoder", "tag": tag, "seed": seed, "parcellation": parc, "direction": f"{src}->{tgt}",
                   "epoch": epoch}
            for name, idx in split.items():
                p, t = pred[idx], data[tgt][idx]
                metrics = compute_basic_regression_metrics(
                    p, t, corr_matrix=compute_corr_matrix(t, p),
                    corr_matrix_demeaned=compute_corr_matrix(t - train_mean[tgt], p - train_mean[tgt]))
                row.update({f"{name}_{k}": float(metrics[k]) for k in METRICS})
                if name in ("train", "val"):
                    losses = term_losses(torch.as_tensor(p), torch.as_tensor(t), batch_size)
                    row.update({f"{name}_loss_raw_{k}": v for k, v in losses.items()})
            rows.append(row)
        log(f"epoch {epoch}: " + ", ".join(f"{r['direction']} val dr {r['val_demeaned_pearson']:.4f}"
                                           f" rank {r['val_avg_rank']:.3f}" for r in rows[-2:]))

    # The last checkpoint should reproduce the run's run_model predictions.
    if ckpts[-1][0] == int(recipe["epochs"]):
        for src, tgt in DIRECTIONS:
            path = krakencoder_prediction_path(seed, parc, src, tag=tag, predictions_root=run_dir.parent.parent)
            if path.exists():
                ref = loadmat(str(path), simplify_cells=True)["predicted_alltypes"][flavors[src]][flavors[tgt]]
                r = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(np.asarray(ref, dtype=np.float32),
                                                                    final_preds[(src, tgt)])])
                log(f"final-epoch check vs run_model {src}->{tgt}: mean per-subject r = {r:.6f}")

    out = args.out or run_dir / "epoch_history.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
