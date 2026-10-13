"""Spec v3 E1: condition and frozen-config construction for the covariate family ablation."""
import json
import os
from copy import deepcopy

import yaml

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", ".."))
DEMO_SOURCES = ("age", "sex", "race_eth")
E2_REF_SOURCES = ["fs_all", "age", "sex", "race_eth"]


def load_config():
    with open(os.path.join(EXP_DIR, "config.yml")) as f:
        return yaml.safe_load(f)


def condition_sources(cfg):
    """{condition: cov_sources} for none, full, drop_<family>, only_<family>, e2_ref."""
    families = cfg["families"]
    full = [s for fam in families.values() for s in fam]
    out = {"none": [], "full": full}
    for fam, srcs in families.items():
        out[f"drop_{fam}"] = [s for s in full if s not in srcs]
    for fam, srcs in families.items():
        out[f"only_{fam}"] = list(srcs)
    out["e2_ref"] = list(E2_REF_SOURCES)
    return out


def grid_conditions(cfg):
    return [c for c in condition_sources(cfg) if c != "e2_ref"]


def frozen_config(cfg, direction, seed):
    """Benchmark default config with the seed's E2.2 best-trial model / trainer config merged on top."""
    with open(os.path.join(REPO_ROOT, cfg["frozen"]["default"])) as f:
        default = yaml.safe_load(f)["default"]
    with open(os.path.join(REPO_ROOT, cfg["frozen"]["records"].format(direction=direction))) as f:
        records = json.load(f)
    rec = [r for r in records if r["model"] == cfg["model"] and int(r["seed"]) == int(seed)]
    if len(rec) != 1:
        raise ValueError(f"expected one E2.2 record for {cfg['model']} {direction} seed {seed}, got {len(rec)}")
    rec = rec[0]
    out = {"model": deepcopy(default["model"]), "trainer": deepcopy(default["trainer"])}
    for section in ("model", "trainer"):
        out[section].update(deepcopy(rec["config"].get(section, {})))
    meta = {"e2_log": rec.get("log"), "e2_ray_tune_id": rec.get("ray_tune_id"),
            "e2_test": rec.get("test"), "e2_val": rec.get("val")}
    return out, meta


def condition_config(frozen, cov_sources, cov_dims, min_dim=2):
    """Frozen config with cov_sources set; the frozen fs_all projector is split over FS families by column count."""
    cfg = deepcopy(frozen)
    model = cfg["model"]
    frozen_proj = model.get("cov_projectors", {})
    fs_spec = frozen_proj["fs_all"]
    n_fs = cov_dims["fs_all"]
    proj = {}
    for src in cov_sources:
        if src in frozen_proj:
            proj[src] = deepcopy(frozen_proj[src])
        elif src.startswith("fs_"):
            proj[src] = {"type": fs_spec.get("type", "linear"),
                         "out_dim": max(int(min_dim), int(round(fs_spec["out_dim"] * cov_dims[src] / n_fs)))}
        else:
            raise ValueError(f"no projector rule for covariate source {src!r}")
    model["cov_sources"] = list(cov_sources)
    model["cov_projectors"] = proj
    return cfg
