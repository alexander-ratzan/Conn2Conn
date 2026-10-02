"""Closed-form reference levels per direction (spec v2 E3.0, Phase D calibration): val / test demeaned r of the null
(CrossModalPCA) and the linear closed form (CrossModal_PCA_PLS, its YAML defaults) on seeds 0-4, SC -> FC and FC -> SC.
Used to transfer the SC -> FC Stage 1 stop thresholds to FC -> SC. CPU only (sbatch).

    python scripts/experiments/composite_loss/checks/direction_baselines.py   # -> checks/direction_baselines.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.hcp_dataset import HCP_Base  # noqa: E402
from models.eval.metrics import compute_demeaned_pearson_r  # noqa: E402
from models.registry import build_model, get_default_config  # noqa: E402

out = {}
for src, tgt in (("SC", "FC"), ("FC", "SC")):
    for seed in range(5):
        base = HCP_Base(parcellation="Glasser", hemi="both", shuffle_seed=seed, source=src, target=tgt,
                        data_load_mode="precomputed")
        X = {"SC": base.sc_upper_triangles, "FC": base.fc_upper_triangles}
        part = base.trainvaltest_partition_indices
        mean = torch.as_tensor(X[tgt][part["train"]].mean(0), dtype=torch.float32)
        for name in ("CrossModalPCA", "CrossModal_PCA_PLS"):
            kw = get_default_config(name).get("model", {})
            model = build_model(base, name, dict(kw))
            row = out.setdefault(f"{src}->{tgt}", {}).setdefault(name, {"val": [], "test": [], "kwargs": {k: v for k, v in kw.items() if k != "name"}})
            for split in ("val", "test"):
                with torch.no_grad():
                    p = model(torch.as_tensor(X[src][part[split]], dtype=torch.float32)).cpu()
                r = compute_demeaned_pearson_r(p, torch.as_tensor(X[tgt][part[split]], dtype=torch.float32), mean)
                row[split].append(float(r.mean()) if hasattr(r, "mean") else float(r))
        print(src, tgt, seed, {n: round(v["val"][-1], 4) for n, v in out[f"{src}->{tgt}"].items()}, flush=True)
for d, models in out.items():
    for v in models.values():
        v["val_mean"], v["test_mean"] = float(np.mean(v["val"])), float(np.mean(v["test"]))
path = Path(__file__).with_name("direction_baselines.json")
path.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({d: {n: (round(v["val_mean"], 4), round(v["test_mean"], 4)) for n, v in m.items()} for d, m in out.items()}))
