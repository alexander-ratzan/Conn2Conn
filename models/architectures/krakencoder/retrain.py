"""
Retrain Krakencoder for one seed with the vendored upstream code and write predictions for the benchmark loader.

    python -m models.architectures.krakencoder.retrain --config models/configs/Krakencoder.yml --seed 0
    python -m models.architectures.krakencoder.retrain --config ... --seed 0 --tag smoke --epochs 20   # quick check

Stages (all by default; each is skipped when its output exists, so a requeued job resumes):
  inputs  connectome .mat files per flavor, built from HCP_Base (identical to the March 2026 inputs to float32
          precision) -> results/krakencoder/_inputs/ (shared by every tag; data do not depend on the recipe)
  split   the seed's HCP_Base train/val/test partition -> results/krakencoder/_inputs/subject_splits_seed<S>.mat
  train   upstream run_training.py (via _vendor_entry.py) -> results/krakencoder/<tag>/seed<S>/kraken_*.pt
  infer   upstream run_model.py once per source flavor -> .../seed<S>/predictions_source_<parc>.<SC|FC>.mat
          (each holds every input -> output type, so SC -> FC and FC -> SC both come from one fit)

The recipe is the config's `retrain:` block (defaults = the March runs); `default.model.tag` names the output folder.
Evaluate afterwards with `main.py --model Krakencoder --config <same config> --source SC|FC --target FC|SC`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.io import savemat  # noqa: E402

from models.architectures.krakencoder.precomputed import (  # noqa: E402
    RETRAINED_PREDICTIONS_ROOT,
    _KRAKEN_FLAVOR_KEY,
    krakencoder_prediction_path,
)

VENDOR_ENTRY = Path(__file__).resolve().parent / "_vendor_entry.py"
VENDOR_COMMIT = "b57e39c2771c36ab39d2f24a5d4355f3d375624d"  # vendor/VENDOR.md
# Recipe keys that may be absent from older configs (upstream default when unset).
OPTIONAL_RECIPE_KEYS = {"batch_size", "random_seed", "infer_parcellations"}
# Input file names per modality (as in the March runs) and the matrix field each file holds.
INPUT_FILE = {"FC": "mydata_{parc}_FCcorrhpf.mat", "SC": "mydata_{parc}_SCifod2actvolnorm.mat"}


def log(msg: str) -> None:
    print(f"[krakencoder.retrain {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _atomic_savemat(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mat")
    savemat(str(tmp), data, format="5", do_compression=False)
    os.replace(tmp, path)


def _hcp_base(parc: str, seed: int):
    from data.hcp_dataset import HCP_Base

    return HCP_Base(parcellation=parc, hemi="both", shuffle_seed=seed, source="SC", target="FC",
                    data_load_mode="precomputed")


def _cell(arrays) -> np.ndarray:
    """1 x N MATLAB cell array (object array) from an iterable of matrices."""
    arrays = list(arrays)
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def build_inputs(recipe: dict, seed: int, inputs_dir: Path) -> dict:
    """Write missing flavor files and this seed's split file; return {modality@parc: path} plus the split path."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths = {(m, parc): inputs_dir / INPUT_FILE[m].format(parc=parc)
             for parc in recipe["parcellations"] for m in recipe["modalities"]}
    split_path = inputs_dir / f"subject_splits_seed{seed}.mat"
    subjects_ref = None
    for parc in recipe["parcellations"]:
        need = [m for m in recipe["modalities"] if not paths[(m, parc)].exists()]
        if not need and split_path.exists():
            continue
        base = _hcp_base(parc, seed)
        # HCP_Base reorders every array to the canonical (sorted, intersected) subject list but keeps
        # sc_/fc_subject_ids in their raw load order; metadata_df is reordered with the arrays.
        sc_ids = [str(s) for s in base.metadata_df["subject"]]
        if not (len(sc_ids) == len(base.sc_matrices) == len(base.fc_matrices)):
            raise RuntimeError(f"{parc}: subject count mismatch between metadata and connectomes")
        if subjects_ref is None:
            subjects_ref = sc_ids
        elif subjects_ref != sc_ids:
            raise RuntimeError(f"{parc}: subject order differs from {recipe['parcellations'][0]}")
        subjects_cell = np.empty((len(sc_ids), 1), dtype=object)
        subjects_cell[:, 0] = sc_ids
        for m in need:
            mats = base.fc_matrices if m == "FC" else base.sc_matrices
            log(f"writing {paths[(m, parc)].name} ({len(sc_ids)} subjects, {mats.shape[1]} regions)")
            _atomic_savemat(paths[(m, parc)], {m: _cell(np.asarray(x, dtype=np.float32) for x in mats),
                                               "subjects": subjects_cell})
        if not split_path.exists():
            part = base.trainvaltest_partition_indices
            _atomic_savemat(split_path, {
                "subjects": np.asarray([float(s) for s in sc_ids])[None, :],
                **{f"subjidx_{k}": np.asarray(part[k], dtype=np.int64)[None, :] for k in ("train", "val", "test")},
                "subjidx_retest": np.zeros((0, 0), dtype=np.int64),
                "subjidx_bad_data": np.zeros((0, 0), dtype=np.int64),
            })
            log(f"wrote {split_path.name}: " + ", ".join(f"{k} {len(part[k])}" for k in ("train", "val", "test")))
    return {"inputs": paths, "split": split_path}


def flavor(m: str, parc: str) -> str:
    return _KRAKEN_FLAVOR_KEY[m].format(parc=parc)


def run_vendor(script: str, args: list, log_path: Path) -> None:
    cmd = [sys.executable, str(VENDOR_ENTRY), script] + [str(a) for a in args]
    log(f"{script}: {' '.join(cmd[3:])[:400]}")
    with open(log_path, "a") as fh:
        fh.write(f"\n# {dt.datetime.now().isoformat()} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(log_path.parent), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        sys.exit(f"{script} failed (exit {proc.returncode}); last lines of {log_path}:\n" + "\n".join(tail))


def find_checkpoint(seed_dir: Path, epochs: int):
    hits = sorted(glob.glob(str(seed_dir / f"kraken_chkpt_*_ep{epochs:06d}.pt")))
    return Path(hits[-1]) if hits else None


def find_input_transform(ckpt: Path) -> Path:
    """The run's input-transform file. Upstream names it differently from the checkpoint (long vs short form), so
    run_model cannot always derive it; match on the run timestamp both names end with."""
    m = re.search(r"_(\d{8}_\d{6})_ep\d+\.pt$", ckpt.name)
    hits = sorted(glob.glob(str(ckpt.parent / f"kraken_ioxfm_*_{m.group(1)}.npy"))) if m else []
    if len(hits) != 1:
        sys.exit(f"expected one kraken_ioxfm_*.npy for {ckpt.name}, found {len(hits)}")
    return Path(hits[0])


def train(recipe: dict, files: dict, seed_dir: Path, epochs: int) -> Path:
    ckpt = find_checkpoint(seed_dir, epochs)
    if ckpt:
        log(f"checkpoint exists, skipping training: {ckpt.name}")
        return ckpt
    args = ["--subjectfile", files["split"], "--inputdata"]
    args += [f"[{flavor(m, parc)}]@{m}={path}" for (m, parc), path in files["inputs"].items()]
    args += ["--datagroups", recipe["datagroups"], "--latentsize", recipe["latentsize"]]
    if recipe.get("latentunit"):
        args += ["--latentunit"]
    if recipe.get("hiddenlayers"):
        args += ["--hiddenlayersizes"] + list(recipe["hiddenlayers"])
    args += ["--transformation", recipe["transformation"], "--dropout", recipe["dropout"],
             "--losstype", recipe["losstype"],
             "--trainvalsplitfrac", recipe["trainvalsplitfrac"], "--valsplitfrac", recipe["valsplitfrac"],
             "--outputprefix", seed_dir / "kraken", "--epochs", epochs,
             "--checkpointepochsevery", min(int(recipe["checkpoint_every"]), epochs),
             "--displayepochs", min(int(recipe["display_every"]), epochs)]
    if recipe.get("batch_size") is not None:
        args += ["--batchsize", int(recipe["batch_size"])]
    if recipe.get("random_seed") is not None:
        args += ["--randseed", int(recipe["random_seed"])]
    args += list(recipe.get("extra_train_args") or [])
    run_vendor("run_training", args, seed_dir / "train.log")
    ckpt = find_checkpoint(seed_dir, epochs)
    if not ckpt:
        sys.exit(f"training finished but no ep{epochs:06d} checkpoint in {seed_dir}")
    return ckpt


def infer(recipe: dict, files: dict, seed: int, tag: str, root: Path, ckpt: Path) -> list:
    written = []
    infer_parcs = recipe.get("infer_parcellations") or recipe["parcellations"]
    for (m, parc), path in files["inputs"].items():
        if parc not in infer_parcs:
            continue
        out = krakencoder_prediction_path(seed, parc, m, tag=tag, predictions_root=root)
        if out.exists():
            log(f"predictions exist, skipping: {out.name}")
            written.append(out)
            continue
        partial = out.with_name(f"{out.stem}.partial.mat")
        run_vendor("run_model", ["--inputdata", f"{flavor(m, parc)}={path}", "--subjectfile", files["split"],
                                 "--adaptmode", recipe["adaptmode"], "--checkpoint", ckpt,
                                 "--inputxform", find_input_transform(ckpt),
                                 "--outputname", "all", "--output", partial,
                                 "--fusioninclude", "fusion=all", "fusionSC=SC", "fusionFC=FC",
                                 "--fusionnoself", "--fusionnoatlas"], out.parent / "infer.log")
        os.replace(partial, out)
        written.append(out)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "models" / "configs" / "Krakencoder.yml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", help="override default.model.tag (use a new tag for checks, e.g. smoke)")
    ap.add_argument("--epochs", type=int, help="override retrain.epochs (quick checks)")
    ap.add_argument("--stage", choices=["all", "inputs", "train", "infer"], default="all")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a retrain recipe key (YAML value), e.g. --set losstype=mse.w1000 --set batch_size=64")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    recipe = dict(cfg["retrain"])
    overrides = {}
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in recipe.keys() | OPTIONAL_RECIPE_KEYS:
            sys.exit(f"--set {item!r}: expected KEY=VALUE with KEY in {sorted(recipe.keys() | OPTIONAL_RECIPE_KEYS)}")
        overrides[key] = yaml.safe_load(value)
    recipe.update(overrides)
    model_cfg = cfg["default"]["model"]
    tag = args.tag or model_cfg["tag"]
    root = Path(model_cfg.get("predictions_root") or RETRAINED_PREDICTIONS_ROOT)
    epochs = int(args.epochs or recipe["epochs"])
    if epochs % min(int(recipe["checkpoint_every"]), epochs):
        sys.exit(f"epochs ({epochs}) must be a multiple of checkpoint_every ({recipe['checkpoint_every']})")
    unknown = [m for m in recipe["modalities"] if m not in INPUT_FILE]
    if unknown:
        sys.exit(f"unsupported modalities {unknown}; expected {list(INPUT_FILE)}")

    seed_dir = root / tag / f"seed{args.seed}"
    t0 = time.time()
    log(f"tag={tag} seed={args.seed} epochs={epochs} flavors={recipe['parcellations']} x {recipe['modalities']}")
    files = build_inputs(recipe, args.seed, root / "_inputs")
    if args.stage == "inputs":
        return
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt = train(recipe, files, seed_dir, epochs) if args.stage in ("all", "train") else find_checkpoint(seed_dir, epochs)
    if args.stage == "train":
        return
    if ckpt is None:
        sys.exit(f"no ep{epochs:06d} checkpoint in {seed_dir}; run the train stage first")
    preds = infer(recipe, files, args.seed, tag, root, ckpt)
    manifest = {
        "tag": tag, "seed": args.seed, "config": str(args.config.resolve().relative_to(REPO_ROOT))
        if args.config.resolve().is_relative_to(REPO_ROOT) else str(args.config),
        "recipe": {**recipe, "epochs": epochs}, "overrides": overrides, "vendor_commit": VENDOR_COMMIT, "repo_commit": git_commit(),
        "checkpoint": ckpt.name, "predictions": [p.name for p in preds],
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"), "wall_s_this_call": round(time.time() - t0, 1),
    }
    (seed_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"done in {time.time() - t0:.0f} s -> {seed_dir}")


if __name__ == "__main__":
    main()
