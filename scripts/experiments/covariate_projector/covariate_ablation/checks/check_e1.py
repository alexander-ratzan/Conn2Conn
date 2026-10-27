"""Spec v3 E1.0 acceptance checks (CPU):

A. FS families split the fs_all columns exactly (disjoint, union = fs_all; 63 / 68 / 68) and are column slices of the
   train-split z-scored fs_all (train mean 0, std 1); HCP_Partition serves them.
B. cov_sources: [] gives z_cov == 0 and the backbone-only prediction; no covariate parameters.
C. The E2.2 covariate set (fs_all + demographics) builds the same parameters and the same outputs as the
   CovProjector at git HEAD (the code before E1), from the same torch seed.
D. Conditions: 10 grid conditions + e2_ref; full = demo + 3 FS families; FS projector dims follow column counts.

  python scripts/experiments/covariate_projector/covariate_ablation/checks/check_e1.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

CHECK_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(CHECK_DIR)
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EXP_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from conditions import condition_config, condition_sources, frozen_config, grid_conditions, load_config  # noqa: E402
from data.hcp_dataset import FS_FAMILY_SOURCES, HCP_Base, HCP_Partition  # noqa: E402
from models.architectures.crossmodal_pca_pls import CrossModal_PCA_PLS_CovProjector  # noqa: E402

ALL_COV = ["fs_all", "fs_vol", "fs_area", "fs_thick", "age", "sex", "race_eth"]
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def head_covprojector():
    src = subprocess.run(["git", "-C", REPO_ROOT, "show", "HEAD:models/architectures/crossmodal_pca_pls.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "crossmodal_pca_pls_head.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("crossmodal_pca_pls_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CrossModal_PCA_PLS_CovProjector


def model_kwargs(cfg_model):
    kw = {k: v for k, v in cfg_model.items() if k not in ("name", "l1_reg", "l2_reg")}
    kw["l1_l2_tuple"] = (cfg_model.get("l1_reg", 0.0), cfg_model.get("l2_reg", 0.0))
    kw["device"] = torch.device("cpu")
    return kw


def batch_of(part, n=16):
    items = [part[i] for i in range(n)]
    x = torch.stack([it["x"] for it in items]) if torch.is_tensor(items[0]["x"]) else None
    cov = {k: torch.stack([it["cov"][k] for it in items]) for k in items[0]["cov"]}
    return x, cov


def main():
    cfg = load_config()
    base = HCP_Base(parcellation="Glasser", hemi="both", shuffle_seed=0, source="FC", target="SC",
                    cov_sources=ALL_COV, data_load_mode="precomputed")

    # A
    cols = base.fs_family_columns
    idx = sorted(i for fam in FS_FAMILY_SOURCES for i in cols[fam])
    check("A1 families disjoint + union = fs_all", idx == list(range(len(base.fs_feature_columns))),
          f"(n_fs_all={len(base.fs_feature_columns)})")
    sizes = {fam: len(cols[fam]) for fam in FS_FAMILY_SOURCES}
    check("A2 family sizes 63/68/68", sizes == {"fs_vol": 63, "fs_area": 68, "fs_thick": 68}, str(sizes))
    slices_ok = all(np.array_equal(base.fs_family_z[f], base.fs_features_z[:, cols[f]]) for f in FS_FAMILY_SOURCES)
    check("A3 families = slices of z-scored fs_all", slices_ok)
    tr = base.trainvaltest_partition_indices["train"]
    stats = {f: (float(np.abs(base.fs_family_z[f][tr].mean(0)).max()),
                 float(np.abs(base.fs_family_z[f][tr].std(0) - 1).max())) for f in FS_FAMILY_SOURCES}
    # constant columns get std 1 in the scaler -> their z is 0 with std 0; allow those
    const = {f: int((base.fs_family_z[f][tr].std(0) == 0).sum()) for f in FS_FAMILY_SOURCES}
    check("A4 train-split z-score", all(m < 1e-4 for m, _ in stats.values()), f"{stats} const_cols={const}")
    part = HCP_Partition(base, "train")
    item = part[0]
    check("A5 partition serves families", all(f in item["cov"] for f in FS_FAMILY_SOURCES),
          str({k: tuple(v.shape) for k, v in item["cov"].items()}))

    frozen, _ = frozen_config(cfg, "fc2sc", 0)
    x, cov = batch_of(part)

    # B
    none_cfg = condition_config(frozen, [], base.cov_dims)["model"]
    torch.manual_seed(0)
    m0 = CrossModal_PCA_PLS_CovProjector(base, **model_kwargs(none_cfg)).eval()
    with torch.no_grad():
        src = {m0.source_modalities[0]: x}
        z_base = m0._predict_target_latent_from_sources(m0._encode_sources(src))
        out = m0(x, cov={})
        ref = m0._decode_target_latent(z_base)
    check("B1 none: output = backbone decode", torch.equal(out, ref))
    check("B2 none: no covariate params", len(list(m0.cov_encoders.parameters())) == 0 and m0.cov_fusion_net is None)
    check("B3 none: reg loss finite", bool(torch.isfinite(m0.get_reg_loss())))

    # C
    ref_cfg = condition_config(frozen, condition_sources(cfg)["e2_ref"], base.cov_dims)["model"]
    check("C0 e2_ref projectors = frozen E2.2 projectors", ref_cfg["cov_projectors"] == frozen["model"]["cov_projectors"])
    Old = head_covprojector()
    torch.manual_seed(0)
    m_old = Old(base, **model_kwargs(ref_cfg)).eval()
    torch.manual_seed(0)
    m_new = CrossModal_PCA_PLS_CovProjector(base, **model_kwargs(ref_cfg)).eval()
    sd_o, sd_n = m_old.state_dict(), m_new.state_dict()
    same_params = sd_o.keys() == sd_n.keys() and all(torch.equal(sd_o[k], sd_n[k]) for k in sd_o)
    check("C1 e2_ref params identical to HEAD", same_params)
    with torch.no_grad():
        o_old, o_new = m_old(x, cov=cov), m_new(x, cov=cov)
    check("C2 e2_ref outputs identical to HEAD", torch.equal(o_old, o_new))

    # D
    srcs = condition_sources(cfg)
    grid = grid_conditions(cfg)
    check("D1 10 grid conditions + e2_ref", len(grid) == 10 and "e2_ref" in srcs and "e2_ref" not in grid, str(grid))
    check("D2 full = demo + FS families", sorted(srcs["full"]) == sorted(["age", "sex", "race_eth", *FS_FAMILY_SOURCES]))
    full_proj = condition_config(frozen, srcs["full"], base.cov_dims)["model"]["cov_projectors"]
    fs_dims = {f: full_proj[f]["out_dim"] for f in FS_FAMILY_SOURCES}
    check("D3 FS projector dims ~ fs_all out_dim", abs(sum(fs_dims.values()) - frozen["model"]["cov_projectors"]["fs_all"]["out_dim"]) <= 2,
          f"{fs_dims} vs fs_all {frozen['model']['cov_projectors']['fs_all']['out_dim']}")
    for cond, s in srcs.items():
        torch.manual_seed(0)
        m = CrossModal_PCA_PLS_CovProjector(base, **model_kwargs(condition_config(frozen, s, base.cov_dims)["model"])).eval()
        with torch.no_grad():
            o = m(x, cov={k: cov[k] for k in s})
        if not torch.isfinite(o).all():
            check(f"D4 {cond} forward finite", False)
    check("D4 every condition builds and runs", not any(f.startswith("D4") for f in failures))

    print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nALL CHECKS PASS", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
