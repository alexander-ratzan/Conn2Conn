"""CPU checks for the composite-loss protocol code (scripts/results_utils/loss_grid.py, protocol.py, report.py) on
synthetic inputs: fake Stage 1 logs / trial folders, a tiny model, synthetic run outputs. Real-data training
(`loss_grid.run_one`) is covered by a GPU smoke run, not here.

Run (kraken_env):  python scripts/experiments/composite_loss/checks/check_protocol.py      Exit 1 on failure.
"""
import copy
import filecmp
import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from scripts.results_utils import loss_grid as lg  # noqa: E402
from models.train.loss import resolve_loss_config, compute_correye_loss, compute_correye_dm_loss, compute_neidist_loss, compute_var_match_loss  # noqa: E402

fails = 0


def check(label, ok, extra=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), label, extra)


def stage1_checks(tmp):
    cfg = lg.load_instance("linear_backbone")
    job = lg._launcher_job_name(cfg["stage1"]["launcher"])
    logs, ckpt = tmp / "logs", tmp / "ckpt"
    logs.mkdir()
    rng = np.random.RandomState(0)
    for seed in range(5):
        tune_id = str(1000 + seed)
        # an older failed attempt for seed 0 must be ignored
        if seed == 0:
            (logs / f"{job}_1_0.out").write_text("Seed=0\nTune run: X_tune_999\n")
        (logs / f"{job}_2_{seed}.out").write_text(f"Seed={seed}  (24 trials)\nTune run: {cfg['model']}_tune_{tune_id}  num_samples=24\nTune finished: 24 trial(s).\n")
        for t in range(24):
            d = ckpt / f"{cfg['model']}_tune_{tune_id}" / f"_tune_trainable_ab12cd{seed:02d}_{t}_l1_reg=0.0000,lr=0.0010_2026-09-30_19-00-00"
            d.mkdir(parents=True)
            params = {"n_components_pca_source": [64, 128, 256][t % 3], "zscore_pca_scores": bool(t % 2),
                      "l2_reg": float(10 ** rng.uniform(-7, -3)), "l1_reg": [0.0, 1e-7][t % 2], "lr": float(10 ** rng.uniform(-4, -2.5)),
                      "max_epochs": [100, 150, 250][t % 3]}
            (d / "params.json").write_text(json.dumps(params))
            val = 0.05 + 0.06 * rng.rand()
            rows = [{"training_iteration": e + 1, "val_demeaned_r": val * (e + 1) / 5, "train_loss_raw_correye": 3.0 - e * 0.1,
                     "val_loss_raw_neidist": -1.0 * e} for e in range(5)]
            (d / "result.json").write_text("\n".join(json.dumps(r) for r in rows))
    runs = lg.find_stage1_runs(cfg, log_dir=logs)
    check("find_stage1_runs: 5 seeds, failed attempt ignored", sorted(runs) == [0, 1, 2, 3, 4] and runs[0]["ray_tune_id"] == "1000")
    trials, epochs = lg.collect_stage1_trials(cfg["model"], runs, ckpt_root=ckpt)
    check("collect_stage1_trials: 120 trials, 600 trial-epochs, monitor columns kept",
          len(trials) == 120 and len(epochs) == 600 and "train_loss_raw_correye" in epochs and "val_loss_raw_neidist" in epochs)
    summary, stop = lg.stage1_summary(trials, min_best_val=0.09)
    check("stage1_summary: one row per seed, best = max val", len(summary) == 5 and
          all(abs(r.best_val - trials[trials.seed == r.seed].val_demeaned_r.max()) < 1e-12 for r in summary.itertuples()))
    _, stop_hi = lg.stage1_summary(trials, min_best_val=1.0)
    check("stop condition trips when best val is below threshold", len(stop_hi) == 1 and "below 1.0" in stop_hi[0])
    search = yaml.safe_load((REPO_ROOT / cfg["stage1"]["config"]).read_text())["search_space"]
    consensus, detail = lg.consensus_config(trials, search)
    best = trials.loc[trials.groupby("seed")["val_demeaned_r"].idxmax()]
    gm = math.exp(np.median(np.log(best["p.lr"])))
    check("consensus: geometric median for loguniform lr", abs(consensus["lr"] - gm) < 1e-12)
    counts = best["p.n_components_pca_source"].value_counts()
    check("consensus: categorical majority", consensus["n_components_pca_source"] in counts[counts == counts.max()].index)
    check("consensus: every search key decided", set(consensus) == set(search))
    tr2 = pd.DataFrame({"seed": range(5), "val_demeaned_r": [0.1, 0.2, 0.3, 0.4, 0.5], "p.max_epochs": [50, 50, 200, 250, 250]})
    c2, _ = lg.consensus_config(tr2, {"max_epochs": {"type": "choice", "values": [50, 100, 150, 200, 250], "consensus": "median"}})
    c3, _ = lg.consensus_config(tr2, {"max_epochs": {"type": "choice", "values": [50, 100, 150, 200, 250]}})
    check("consensus: `consensus: median` choice -> median snapped to the grid (200), majority otherwise (250)",
          c2["max_epochs"] == 200 and c3["max_epochs"] == 250)
    ok, stats = lg.consensus_accepted({s: v for s, v in zip(range(5), [0.1] * 5)}, {s: 0.1 for s in range(5)})
    check("consensus_accepted: equal -> accepted", ok)
    ok2, _ = lg.consensus_accepted({s: 0.05 for s in range(5)}, {0: 0.10, 1: 0.11, 2: 0.10, 3: 0.11, 4: 0.10})
    check("consensus_accepted: large miss -> rejected", not ok2)
    split = lg.split_consensus(consensus, cfg["stage1"]["config"])
    check("split_consensus: lr / max_epochs -> trainer, the rest -> model",
          set(split["trainer"]) == {"lr", "max_epochs"} and "l2_reg" in split["model"])


def loss_config_checks():
    cfg = lg.load_instance("linear_backbone")
    scales = {"varmatch": 0.3, "correye": 50.0, "correye_dm": 7.0, "neidist": 20.0}
    ok_all = True
    for c in cfg["grid"]["combos"]:
        tr = lg.combo_loss_trainer(c, scales, 64)
        r = resolve_loss_config(tr)
        active = [t for t in lg.TERMS if c.get(t, 0) > 0]
        ok = (r["loss_signature"] == lg.expected_signature(c) and r["loss_normalize"] == "none"
              and (r["loss_monitor_terms"] or []) == [t for t in lg.TERMS if t not in active]
              and all(s["kwargs"]["scale"] == scales[s["name"]] for s in r["loss_terms"] if isinstance(s, dict) and s["name"] != "mse")
              and tr["batch_size"] == 64 and tr["log_every"] == 1)
        ok_all &= ok
        if not ok:
            print("   bad combo", c["id"], r)
    n = len(cfg["grid"]["combos"])
    check(f"all {n} grid combos resolve (signature, normalize none, scales, monitors = inactive terms)", ok_all and n == 29)
    try:
        lg.combo_loss_trainer({"id": "bad", "correye": 0.5, "correye_dm": 0.5}, scales, 64)
        check("correye + correye_dm in one combination raises", False)
    except ValueError:
        check("correye + correye_dm in one combination raises", True)
    check("no grid combination activates both correye variants",
          all(not (c.get("correye", 0) > 0 and c.get("correye_dm", 0) > 0) for c in cfg["grid"]["combos"]))


class _DS(torch.utils.data.Dataset):
    def __init__(self, n=200, d_in=10, d_out=20, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x, self.y = torch.randn(n, d_in, generator=g), torch.randn(n, d_out, generator=g)
        self.base = types.SimpleNamespace(target_modality="FC", fc_train_avg=np.linspace(-1, 1, d_out).astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class _Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_mid = nn.Parameter(torch.randn(10, 20) * 0.1)

    def forward(self, x):
        return x @ self.W_mid


def scale_checks():
    torch.manual_seed(0)
    model, ds = _Lin(), _DS()
    s = lg.measure_term_scales(model, ds, 64)
    with torch.no_grad():
        p = ds.x @ model.W_mid
    manual = np.mean([abs(float(compute_correye_loss(p[i:i + 64], ds.y[i:i + 64]))) for i in range(0, 192, 64)])
    check("measure_term_scales: full batches only (3 of 200/64), matches manual |correye|",
          s["correye"]["n_batches"] == 3 and abs(s["correye"]["abs_mean"] - manual) < 1e-6)
    mean = torch.as_tensor(ds.base.fc_train_avg)
    manual_dm = np.mean([abs(float(compute_correye_dm_loss(p[i:i + 64], ds.y[i:i + 64], mean))) for i in range(0, 192, 64)])
    check("measure_term_scales: correye_dm uses the training target mean", abs(s["correye_dm"]["abs_mean"] - manual_dm) < 1e-6)
    by_seed = {0: s, 1: copy.deepcopy(s)}
    by_seed[1]["correye"]["abs_mean"] *= 2
    r = lg.reference_scales(by_seed)
    exp = (s["correye"]["abs_mean"] * 1.5) / s["mse"]["abs_mean"]
    check("reference_scales: c_t = mean(s_t) / mean(s_mse); spread recorded", abs(r["c"]["correye"] - exp) < 1e-9 and r["spread_cv"]["correye"] > 0)


def grad_cosine_checks():
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    from models.train.lightning_module import CrossModalLightningModule

    class FakeBase:
        target_modality = "FC"
        fc_train_avg = np.zeros(20, dtype=np.float32)

    cb = lg.TermGradCosine("W_mid", every=1)
    m = CrossModalLightningModule(_Lin(), FakeBase(), lr=1e-2, loss_cfg={"loss_type": "composite", "loss_terms": ["mse"]})
    tr = pl.Trainer(max_epochs=3, accelerator="cpu", logger=False, enable_progress_bar=False, enable_checkpointing=False,
                    enable_model_summary=False, callbacks=[cb])
    tr.fit(m, DataLoader(_DS(), batch_size=64), DataLoader(_DS(seed=1), batch_size=64))
    rows = pd.DataFrame(cb.rows)
    cos_cols = [c for c in rows if c.startswith("cos_")]
    check("TermGradCosine: a row per epoch, 6 pairwise cosines in [-1, 1]",
          len(rows) == 3 and len(cos_cols) == 6 and rows[cos_cols].abs().le(1 + 1e-6).all().all())
    b = cb._batch
    W = m.model.W_mid
    p = b["x"] @ W
    ga, = torch.autograd.grad(torch.nn.functional.mse_loss(p, b["y"]), W, retain_graph=True)
    gb, = torch.autograd.grad(compute_neidist_loss(p, b["y"]), W)
    manual = float(torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0))
    check("TermGradCosine: final-epoch mse/neidist cosine matches a manual computation", abs(rows.iloc[-1]["cos_mse_neidist"] - manual) < 1e-5)
    mean = np.linspace(-1, 1, 20).astype(np.float32)
    cb2 = lg.TermGradCosine("no_such_param", every=1, target_mean=mean)
    m2 = CrossModalLightningModule(_Lin(), FakeBase(), lr=1e-2, loss_cfg={"loss_type": "composite", "loss_terms": ["mse"]})
    pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_progress_bar=False, enable_checkpointing=False,
               enable_model_summary=False, callbacks=[cb2]).fit(m2, DataLoader(_DS(), batch_size=64), DataLoader(_DS(seed=1), batch_size=64))
    rows2 = pd.DataFrame(cb2.rows)
    check("TermGradCosine with target mean: 5 terms, 10 cosines incl. correye_dm",
          len([c for c in rows2 if c.startswith("cos_")]) == 10 and "gnorm_correye_dm" in rows2)
    check("TermGradCosine: missing/frozen param falls back to a trainable weight (recorded)",
          set(rows2["grad_param"]) == {"W_mid"})


def report_checks(tmp):
    import importlib.util
    exp_dir = tmp / "exp"
    inst = exp_dir / "linear_backbone"
    inst.mkdir(parents=True)
    shutil.copy(lg.instance_dir("linear_backbone") / "config.yml", inst / "config.yml")
    orig = lg.EXPERIMENT_DIR
    lg.EXPERIMENT_DIR = exp_dir
    try:
        lg.update_state("linear_backbone", reference_scales={"c": {"varmatch": 0.3, "correye": 50.0, "correye_dm": 7.0, "neidist": 20.0}})
        cfg = lg.load_instance("linear_backbone")
        rng = np.random.RandomState(1)
        for stage, combos in (("consensus", [{"id": "mse_only", "varmatch": 0, "correye": 0, "correye_dm": 0, "neidist": 0}]), ("stage2", cfg["grid"]["combos"])):
            out = inst / "runs" / stage
            out.mkdir(parents=True)
            for c in combos:
                for seed in range(5):
                    rec = {"model": cfg["model"], "instance": "linear_backbone", "grid_version": "v3", "stage": stage, "combo_id": c["id"],
                           "seed": seed, "batch_size": 64, "loss_signature": lg.expected_signature(c), "val_demeaned_r_last": 0.1, "epochs": 6,
                           **{f"{sp}_{m}": float(rng.rand()) for sp in ("train", "val", "test") for m in lg.METRICS}}
                    (out / f"{c['id']}__seed{seed}.json").write_text(json.dumps(rec))
                    ep = pd.DataFrame({"combo_id": c["id"], "seed": seed, "epoch": range(6), "val_demeaned_r": rng.rand(6),
                                       **{f"{sp}_loss_raw_{t}": rng.rand(6) for sp in ("train", "val") for t in lg.ALL_TERMS}})
                    if c["id"] in ("mse_only", "all_0.5", "alldm_0.5"):
                        for a in range(6):
                            ep.loc[a, "cos_mse_neidist"] = np.cos(a)
                    ep.to_csv(out / f"{c['id']}__seed{seed}__epochs.csv", index=False)
        stray = dict(rec, combo_id="vm_ce_0.5", stage="stage2")  # a v2-only combination left in runs/
        (inst / "runs" / "stage2" / "vm_ce_0.5__seed0.json").write_text(json.dumps(stray))
        spec = importlib.util.spec_from_file_location("report", lg.REPO_ROOT / "scripts/experiments/composite_loss/report.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        report.lg.EXPERIMENT_DIR = exp_dir
        rc = report.build("linear_backbone")
        expected = ["tables/seed_records.csv", "tables/combo_summary.csv", "tables/epoch_history.csv.gz", "figures/tradeoff_scatter.png",
                    "figures/tradeoff_interactive.html", "figures/term_trajectories.png", "figures/loss_composition.png",
                    "figures/val_trajectories.png", "figures/dose_response.png", "figures/grad_cosine.png"]
        missing = [e for e in expected if not (inst / e).exists()]
        check("report.build writes every table and figure", rc == 0 and not missing, f"missing={missing}")
        summ = pd.read_csv(inst / "tables" / "combo_summary.csv")
        check("combo_summary: 29 combinations × 5 seeds, grid order", len(summ) == 29 and (summ["n_seeds"] == 5).all()
              and summ["combo_id"].tolist() == [c["id"] for c in cfg["grid"]["combos"]])
        snap = tmp / "snap"
        shutil.copytree(inst / "tables", snap / "tables")
        shutil.copytree(inst / "figures", snap / "figures")
        report.build("linear_backbone")
        same = all(filecmp.cmp(snap / e, inst / e, shallow=False) for e in expected if not e.endswith(".gz"))
        check("report is deterministic (tables, PNGs and HTML byte-identical on rebuild)", same)
        htm = (inst / "figures" / "tradeoff_interactive.html").read_text()
        check("interactive HTML: 29 clickable points, no external scripts", htm.count('class="pt"') == 29 and "<script src" not in htm)
    finally:
        lg.EXPERIMENT_DIR = orig


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        stage1_checks(tmp)
        loss_config_checks()
        scale_checks()
        grad_cosine_checks()
        report_checks(tmp)
    print("protocol check failures:", fails)
    sys.exit(1 if fails else 0)
