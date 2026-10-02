"""Generate the E2.2 MSE-only benchmark configs (spec v2 E2.2) into models/configs/benchmark/mse/.

Each benchmark config = a model's source config (models/configs/<source>.yml) with:
  - the `data:` block removed (direction comes from the launcher's --source / --target, which always win in main.py),
  - the loss pinned to MSE (composite with a single `mse` term; loss-weight / EMA / loss-type keys removed from the
    search), except models that train on a native objective (MaskedMLPPretrainer: latent_mse, labelled),
  - per-model fixed keys and narrowed search keys from the 2026-09-30 audit and E1 findings (SPECS below).
FC -> SC variants (`<Model>_fc2sc.yml`) are added only where a model setting must differ by direction (E2.0 audit).

    python scripts/experiments/model_benchmark/build_configs.py          # (re)write all configs
    python scripts/experiments/model_benchmark/build_configs.py --check  # exit 1 if any file differs from SPECS
"""
import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
SRC = REPO_ROOT / "models" / "configs"
OUT = SRC / "benchmark" / "mse"
LOSS_KEY = re.compile(r"^loss_(weight|kwarg|scale|normalize|type|terms)|^loss_type$")
MSE_TRAINER = {"loss_type": "composite", "loss_terms": [{"name": "mse", "weight": 1.0}], "loss_normalize": "auto"}

# name -> {source, why, fixed (model/trainer keys), search (replace / add keys), drop (remove search keys),
#          search_alg, trials, native_loss}
SPECS = {
    "CrossModalPCA": dict(
        source="CrossModalPCA", why="null reference; full grid over num_components",
        grid=True),
    "CrossModal_PLS_SVD": dict(
        source="CrossModal_PLS_SVD", why="closed-form; full grid over n_components",
        grid=True),
    "CrossModal_PCA_PLS": dict(
        source="CrossModal_PCA_PLS", why="closed-form; full 150-cell grid (cheap)",
        grid=True),
    "CrossModal_ConditionalGaussian": dict(
        source="CrossModal_ConditionalGaussian",
        why="closed-form ridge-form conditional mean in PCA space; fit_domain fixed to pca because the model rejects "
            "raw_edges with any shrinkage estimator (168 of the pilot's trials errored); 4 keys -> 32",
        fixed={"model": {"fit_domain": "pca"}}),
    "CrossModal_PCA_PLS_learnable": dict(
        source="CrossModal_PCA_PLS_learnable", why="model search space minus loss keys (12 keys -> 64)",
        trials=64),
    "CrossModal_linear_backbone": dict(
        source="CrossModal_linear_backbone", why="model search space minus loss keys (6 keys -> 48)",
        trials=48),
    "CrossModal_PCA_PLS_CovProjector": dict(
        source="CrossModal_PCA_PLS_CovProjector_SC_fs_all_demo",
        why="all covariates (fs_all + age, sex, race/eth), as E1.9; learn_mid added to the search because E1.9 found "
            "the frozen backbone (model default) was the limit (9 keys -> 64)",
        search={"learn_mid": {"type": "choice", "values": [False, True]}},
        trials=64),
    "Sarwar2020MLP": dict(
        source="Sarwar2020MLP",
        why="audit narrowing (3,592 past trials): 3-5 layers, 512-1024 hidden, leaky_relu, low dropout; plain MSE "
            "(its correlation term showed no effect); l2 newly applied (v2:C!1)",
        fixed={"model": {"activation_mode": "leaky_relu", "output_tanh": True, "l1_reg": 0.0},
               "trainer": {"max_epochs": 300}},
        search={"num_hidden_layers": {"type": "choice", "values": [3, 5]},
                "hidden_dim": {"type": "choice", "values": [512, 1024]},
                "dropout": {"type": "uniform", "lower": 0.1, "upper": 0.35},
                "lr": {"type": "loguniform", "lower": 2.0e-5, "upper": 3.0e-4},
                "l2_reg": {"type": "loguniform", "lower": 1.0e-7, "upper": 1.0e-3}},
        only_search=True, trials=16),
    "Chen2024GCN": dict(
        source="Chen2024GCN",
        why="audit narrowing: identity nodes, 2 layers, 500 epochs; conv_dim 128/256 >> 32/64",
        fixed={"model": {"node_feature_type": "identity", "layer_num": 2}, "trainer": {"max_epochs": 500}},
        search={"conv_dim": {"type": "choice", "values": [128, 256]},
                "dnn_dim": {"type": "choice", "values": [32, 64]},
                "lr": {"type": "loguniform", "lower": 5.0e-4, "upper": 5.0e-3},
                "l2_reg": {"type": "loguniform", "lower": 1.0e-5, "upper": 1.0e-3}},
        only_search=True, trials=12),
    "NodalGNN": dict(
        source="NodalGNN",
        why="audit narrowing: 2 layers, decoder 32, default dropouts, 500 epochs; hidden 32/96 best",
        fixed={"model": {"layer_num": 2, "decoder_dim": 32}, "trainer": {"max_epochs": 500}},
        search={"hidden_dim": {"type": "choice", "values": [32, 96]},
                "lr": {"type": "loguniform", "lower": 2.0e-4, "upper": 5.0e-3},
                "l2_reg": {"type": "loguniform", "lower": 5.0e-6, "upper": 1.0e-3}},
        only_search=True, trials=10),
    "NodalMLP": dict(
        source="NodalMLP",
        why="E0 importance (lr, embedding_dim, batch_size dominate): MLP decoder [256, 128], pca encoder, 500 epochs",
        fixed={"model": {"encoder_type": "pca", "decoder_dims": [256, 128], "decoder_symmetry": "asymmetric",
                         "sc_row_norm": "brain_scale"},
               "trainer": {"max_epochs": 500}},
        search={"lr": {"type": "loguniform", "lower": 1.0e-3, "upper": 5.0e-3},
                "embedding_dim": {"type": "choice", "values": [32, 64]},
                "batch_size": {"type": "choice", "values": [8, 32]}},
        only_search=True, trials=12),
    "MaskedMLPPretrainer": dict(
        source="MaskedMLPPretrainer_nonlinear",
        why="nonlinear variant (user 2026-10-02), narrowed around the April mask-grid optimum (val 0.113 at SC 0 / "
            "FC 0.05, k = 128); dropout and l2 searched against overfitting. Native objective latent_mse (labelled).",
        native_loss=True,
        fixed={"model": {"n_components_pca": 128, "num_hidden_layers": 2, "nonlinear": True, "readout_type": "mlp",
                         "readout_hidden_dim": 128},
               "trainer": {"loss_type": "latent_mse"}},
        search={"sc_mask_ratio": {"type": "choice", "values": [0.0, 0.1, 0.2]},
                "fc_mask_ratio": {"type": "choice", "values": [0.05, 0.1, 0.2]},
                "hidden_dim": {"type": "choice", "values": [128, 256]},
                "dropout": {"type": "choice", "values": [0.1, 0.2, 0.3]},
                "l2_reg": {"type": "loguniform", "lower": 1.0e-6, "upper": 1.0e-3},
                "lr": {"type": "loguniform", "lower": 3.0e-4, "upper": 3.0e-3},
                "max_epochs": {"type": "choice", "values": [200, 400]}},
        only_search=True, trials=56),
}
# FC -> SC deltas (E2.0 audit): applied on top of the SC -> FC benchmark config.
FC2SC = {
    "Sarwar2020MLP": dict(why="SC targets reach 3.59 (log1p), so the tanh output bound must be off",
                          fixed={"model": {"output_tanh": False}}),
}


def budget(n_free):
    return max(16, min(64, 8 * n_free))


def build_one(name, spec, fc2sc=None):
    cfg = yaml.safe_load((SRC / f"{spec['source']}.yml").read_text())
    out = {"learned": cfg.get("learned", True), "default": copy.deepcopy(cfg["default"])}
    out["default"].pop("data", None)
    model = out["default"].setdefault("model", {})
    trainer = out["default"].setdefault("trainer", {})
    if out["learned"] and not spec.get("native_loss"):
        for k in [k for k in trainer if k.startswith("loss_")]:
            trainer.pop(k)
        trainer.update(copy.deepcopy(MSE_TRAINER))
    for sec, kv in (spec.get("fixed") or {}).items():
        out["default"][sec].update(copy.deepcopy(kv))
    if fc2sc:
        for sec, kv in (fc2sc.get("fixed") or {}).items():
            out["default"][sec].update(copy.deepcopy(kv))
    search = {} if spec.get("only_search") else {k: v for k, v in (cfg.get("search_space") or {}).items()
                                                   if not LOSS_KEY.match(k)}
    search.update(copy.deepcopy(spec.get("search") or {}))
    fixed_keys = {k for sec in (spec.get("fixed") or {}).values() for k in sec}
    fixed_keys |= {k for sec in ((fc2sc or {}).get("fixed") or {}).values() for k in sec}
    search = {k: v for k, v in search.items() if k not in fixed_keys}
    if spec.get("grid"):
        search = {k: ({"type": "grid", "values": v["values"]} if v.get("type") in ("choice", "grid") else v)
                  for k, v in search.items()}
        trials, alg = 1, "random"
    else:
        trials, alg = spec.get("trials") or budget(len(search)), "optuna"
    out["search_space"] = search
    out["benchmark"] = {"campaign": "e2_mse", "source_config": f"models/configs/{spec['source']}.yml",
                        "search_alg": alg, "num_samples": trials,
                        "objective": "native latent_mse" if spec.get("native_loss") else ("mse" if out["learned"] else "closed-form"),
                        "direction": "fc2sc" if fc2sc else "both"}
    head = (f"# E2.2 MSE-only benchmark config (spec v2 E2.2) for {name}. GENERATED by\n"
            f"# scripts/experiments/model_benchmark/build_configs.py; edit SPECS there, not this file.\n"
            f"# Source: models/configs/{spec['source']}.yml. Direction: set by the launcher (--source/--target), "
            f"no data block.\n# Why: {spec['why']}\n")
    if fc2sc:
        head += f"# FC -> SC delta: {fc2sc['why']}\n"
    return head + yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {f"{n}.yml": build_one(n, s) for n, s in SPECS.items()}
    targets.update({f"{n}_fc2sc.yml": build_one(n, SPECS[n], d) for n, d in FC2SC.items()})
    bad = []
    for fname, text in targets.items():
        path = OUT / fname
        if args.check:
            if not path.exists() or path.read_text() != text:
                bad.append(fname)
        else:
            path.write_text(text)
    if args.check:
        print("out of date:" if bad else "all configs match SPECS", bad or "")
        return 1 if bad else 0
    for fname in targets:
        y = yaml.safe_load((OUT / fname).read_text())
        b = y["benchmark"]
        print(f"{fname:46s} keys={len(y['search_space']):2d} trials={b['num_samples']:3d} alg={b['search_alg']:7s} objective={b['objective']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
