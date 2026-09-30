"""E1.1 checks (spec v2): fixed per-term loss scales and monitor-only loss terms. CPU only, random tensors.

Run (kraken_env, from anywhere in the repo):
    python scripts/experiments/linear_backbone/composite_loss/checks/check_e1_loss.py
Exit code 1 on any failure.
"""
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.train.loss import (  # noqa: E402
    compute_correye_loss, compute_neidist_loss, compute_var_match_loss, create_loss_fn, resolve_loss_config,
)

fails = 0
MON = ["varmatch", "correye", "neidist"]


def check(label, ok, extra=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), label, extra)


def comp(**kw):
    return create_loss_fn({"loss_type": "composite", **kw})


def run(fn, steps=10, seed=0):
    """Loss values and gradient summaries over a few steps of shrinking prediction noise."""
    torch.manual_seed(seed)
    y = torch.randn(16, 40)
    outs = []
    fn.train()
    for s in range(steps):
        pred = (y + (1 - s / 12) * torch.randn_like(y)).requires_grad_(True)
        o = fn(pred, y)
        g, = torch.autograd.grad(o, pred)
        outs += [o.detach(), g.norm(), g.sum()]
    return torch.stack(outs)


def scaling_checks():
    s_c, s_n, w = 7.25, 0.8, 0.5
    fn = comp(loss_normalize="none", loss_terms=[{"name": "mse", "weight": 1.0},
              {"name": "correye", "weight": w, "kwargs": {"scale": s_c}},
              {"name": "neidist", "weight": 0.25, "kwargs": {"scale": s_n}}])
    torch.manual_seed(3)
    y = torch.randn(16, 40)
    p = (y + 0.5 * torch.randn_like(y)).requires_grad_(True)
    out = fn(p, y)
    g1, = torch.autograd.grad(out, p)
    p2 = p.detach().clone().requires_grad_(True)
    manual = torch.stack([F.mse_loss(p2, y), torch.tensor(w) * (compute_correye_loss(p2, y) / s_c),
                          torch.tensor(0.25) * (compute_neidist_loss(p2, y) / s_n)]).sum()
    g2, = torch.autograd.grad(manual, p2)
    check("weight*raw/scale exact (value)", torch.equal(out.detach(), manual.detach()))
    check("weight*raw/scale exact (gradient)", torch.equal(g1, g2))
    check("raw terms logged unscaled", torch.equal(fn.last_raw_terms["correye"], compute_correye_loss(p.detach(), y)))
    base = [{"name": "mse", "weight": 1.0}, {"name": "neidist", "weight": 0.5}]
    one = [{"name": "mse", "weight": 1.0}, {"name": "neidist", "weight": 0.5, "kwargs": {"scale": 1.0}}]
    for norm in ("none", "ema"):
        check(f"scale 1 bit-identical to no scale ({norm})",
              torch.equal(run(comp(loss_normalize=norm, loss_terms=base)), run(comp(loss_normalize=norm, loss_terms=one))))


def config_checks():
    r = resolve_loss_config({"loss_type": "composite", "loss_normalize": "auto", "loss_terms": [
        {"name": "mse", "weight": 1.0}, {"name": "correye", "weight": 0.5, "kwargs": {"scale": 3.0}}]})
    check("auto + fixed scale -> none", r["loss_normalize"] == "none")
    r = resolve_loss_config({"loss_type": "composite", "loss_normalize": "auto",
                             "loss_terms": [{"name": "mse", "weight": 1.0}, {"name": "correye", "weight": 0.0}],
                             "loss_weight_correye": 0.5, "loss_kwarg_correye__scale": 4.0})
    check("Tune keys loss_weight_ + loss_kwarg_<term>__scale",
          r["loss_normalize"] == "none" and r["loss_terms"][-1] == {"name": "correye", "weight": 0.5, "kwargs": {"scale": 4.0}})
    check("signature unaffected by scale", r["loss_signature"] == "mse+0.5*correye")
    for label, cfg in [
        ("ema + scale", {"loss_normalize": "ema", "loss_terms": [{"name": "mse"}, {"name": "correye", "weight": 0.5, "kwargs": {"scale": 3.0}}]}),
        ("scale 0", {"loss_normalize": "none", "loss_terms": [{"name": "mse"}, {"name": "correye", "weight": 0.5, "kwargs": {"scale": 0.0}}]}),
        ("monitor kld", {"loss_terms": ["mse"], "loss_monitor_terms": ["kld"]}),
        ("monitor unknown", {"loss_terms": ["mse"], "loss_monitor_terms": ["foo"]}),
    ]:
        try:
            resolve_loss_config({"loss_type": "composite", **cfg})
            check(f"rejects {label}", False)
        except ValueError:
            check(f"rejects {label}", True)
    try:
        comp(loss_terms=["mse"], loss_monitor_terms=["demeaned_mse"])
        check("monitor demeaned_mse without base raises", False)
    except ValueError:
        check("monitor demeaned_mse without base raises", True)
    r = resolve_loss_config({"loss_type": "composite", "loss_terms": [{"name": "mse"}, {"name": "correye", "weight": 0.5}],
                             "loss_monitor_terms": ["correye", "neidist", "varmatch"]})
    check("active term removed from monitors", r["loss_monitor_terms"] == ["neidist", "varmatch"])
    check("monitors ignored for latent loss types",
          resolve_loss_config({"loss_type": "latent_mse", "loss_monitor_terms": ["correye"]})["loss_monitor_terms"] is None)
    check("signature unaffected by monitors", r["loss_signature"] == "mse+0.5*correye")


def monitor_checks():
    for terms, norm in [(["mse"], "auto"), ([{"name": "mse"}, {"name": "neidist", "weight": 0.5}], "ema"),
                        ([{"name": "mse"}, {"name": "correye", "weight": 0.5, "kwargs": {"scale": 5.0}}], "none")]:
        a = run(comp(loss_normalize=norm, loss_terms=terms))
        b = run(comp(loss_normalize=norm, loss_terms=terms, loss_monitor_terms=MON))
        check(f"monitors leave training bit-identical ({norm}, {len(terms)} active)", torch.equal(a, b))
    fm = comp(loss_terms=["mse"], loss_monitor_terms=MON)
    torch.manual_seed(5)
    y = torch.randn(16, 40)
    p = y + 0.3 * torch.randn_like(y)
    fm.train()
    fm(p, y)
    exp = {"varmatch": compute_var_match_loss(p, y), "correye": compute_correye_loss(p, y), "neidist": compute_neidist_loss(p, y)}
    check("monitor values equal the term functions", all(torch.equal(fm.last_monitor_terms[k], v) for k, v in exp.items()))


def lightning_checks():
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset
    from models.train.lightning_module import CrossModalLightningModule, structured_loss_metric_names

    class FakeBase:
        target_modality = "FC"
        fc_train_avg = np.zeros(20, dtype=np.float32)

    class DS(Dataset):
        def __init__(self):
            g = torch.Generator().manual_seed(0)
            self.x, self.y = torch.randn(64, 10, generator=g), torch.randn(64, 20, generator=g)

        def __len__(self):
            return 64

        def __getitem__(self, i):
            return {"x": self.x[i], "y": self.y[i]}

    cfg = {"loss_type": "composite", "loss_terms": ["mse"], "loss_monitor_terms": MON}
    m = CrossModalLightningModule(nn.Linear(10, 20), FakeBase(), lr=1e-3, loss_cfg=cfg)
    tr = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_progress_bar=False,
                    enable_checkpointing=False, enable_model_summary=False)
    tr.fit(m, DataLoader(DS(), batch_size=16), DataLoader(DS(), batch_size=16))
    wanted = (structured_loss_metric_names(cfg, phases=("train",), kinds=("raw",))
              + structured_loss_metric_names(cfg, phases=("val",), kinds=("raw", "weighted", "ref")))
    missing = [x for x in wanted if x not in tr.callback_metrics]
    check("Tune names include monitors and all are logged", not missing and "val_loss_raw_neidist" in wanted, f"missing={missing}")
    check("hparams carry loss_monitor_terms; signature 'mse'",
          m.hparams.get("loss_monitor_terms") == MON and m.hparams["loss_signature"] == "mse")
    m2 = CrossModalLightningModule(nn.Linear(10, 20), FakeBase(), lr=1e-3, loss_cfg={"loss_type": "composite", "loss_terms": ["mse"]})
    check("no loss_monitor_terms hparam when unset", "loss_monitor_terms" not in m2.hparams)


if __name__ == "__main__":
    scaling_checks()
    config_checks()
    monitor_checks()
    lightning_checks()
    print("E1.1 failures:", fails)
    sys.exit(1 if fails else 0)
