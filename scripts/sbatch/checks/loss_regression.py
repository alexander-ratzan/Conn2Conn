"""Loss-path regression against an earlier commit: same loss class, identical outputs, identical Lightning hparams.

For every learned YAML in models/configs (default trainer section, plus each `loss_type` search choice), builds the
loss with the old commit's `models/train/loss.py` + `lightning_module.py` and with the working tree's, then compares:
the loss module class, outputs over 30 train steps + 1 eval step on fixed random tensors (bit-exact), and the
module hparams. Uses the post-v1:M2 API on both sides (`create_loss_fn(resolve_loss_config(tc))`).

Run (kraken_env, CPU, from the repo root):
    python scripts/sbatch/checks/loss_regression.py --old-ref 626f37d
Exit code 1 on any mismatch.
"""
import argparse
import copy
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _load_old(ref, rel, name, tmp):
    src = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{ref}:{rel}"], check=True, capture_output=True, text=True).stdout
    path = Path(tmp) / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(mod, tc):
    cfg = mod.resolve_loss_config(tc)
    return None if cfg["loss_type"] in mod.LATENT_LOSS_TYPES else mod.create_loss_fn(cfg)


def _outputs(fn):
    if fn is None:
        return None
    torch.manual_seed(0)
    y = torch.randn(16, 40)
    fn.train()
    outs = []
    for step in range(30):
        pred = y + (1.0 - step / 40) * torch.randn_like(y)
        mu, logvar = torch.randn(16, 4), torch.randn(16, 4) * 0.1
        outs.append(fn(pred, y, mu=mu, logvar=logvar))
    fn.eval()
    outs.append(fn(y + 0.3 * torch.randn_like(y), y, mu=mu, logvar=logvar))
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-ref", required=True, help="git commit to compare against")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    import models.train.loss as new_loss
    from models.train.lightning_module import CrossModalLightningModule
    from models.registry import load_config

    with tempfile.TemporaryDirectory() as tmp:
        old_loss = _load_old(args.old_ref, "models/train/loss.py", "old_loss", tmp)
        old_lm = _load_old(args.old_ref, "models/train/lightning_module.py", "old_lightning", tmp)
        cases = []
        for path in sorted(glob.glob("models/configs/*.yml")):
            full = load_config(None, path=path)
            if not full.get("learned", True):
                continue
            tc = copy.deepcopy(full["default"].get("trainer", {}))
            cases.append((os.path.basename(path), tc))
            for lt in (full.get("search_space", {}).get("loss_type") or {}).get("values", []):
                if lt != tc.get("loss_type", "composite"):
                    cases.append((f"{os.path.basename(path)} [loss_type={lt}]", {**tc, "loss_type": lt}))
        fails = 0
        for name, tc in cases:
            try:
                o = _build(old_loss, tc)
            except Exception as e:  # compare errors too
                o = e
            try:
                n = _build(new_loss, tc)
            except Exception as e:
                n = e
            if isinstance(o, Exception) or isinstance(n, Exception):
                ok = type(o) is type(n)
                print(("same-error " if ok else "FAIL error ") + name, repr(o)[:80], repr(n)[:80])
                fails += not ok
                continue
            same = type(o).__name__ == type(n).__name__ and ((o is None and n is None) or torch.equal(_outputs(o), _outputs(n)))
            oh = dict(old_lm.CrossModalLightningModule(nn.Linear(2, 2), None, lr=tc.get("lr", 1e-4), loss_cfg=tc).hparams)
            nh = dict(CrossModalLightningModule(nn.Linear(2, 2), None, lr=tc.get("lr", 1e-4), loss_cfg=tc).hparams)
            ok = same and oh == nh
            fails += not ok
            if not ok:
                print("FAIL", name, same, oh, nh)
    print(f"loss regression vs {args.old_ref}: {len(cases)} cases, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
