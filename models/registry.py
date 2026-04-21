"""Model registry and configuration helpers.

Loads model YAML configs, resolves search spaces, and builds architecture instances.
"""

import os
from copy import deepcopy
import yaml

TRAINER_KEYS = {"lr", "loss_type", "loss_alpha", "loss_beta", "loss_corr_target", "loss_corr_weight", "loss_terms", "loss_normalize", "loss_scale_ema_decay", "loss_scale_warmup_steps", "max_epochs", "batch_size", "log_every"}
DATA_KEYS = {"parcellation", "hemi", "source", "target", "shuffle_seed", "HCP_dir", "sc_metric_type", "sc_apply_log1p", "volume_feature_type", "centroid_feature_type", "data_load_mode", "precompute_cache_root", "write_manual_cache"}
FLAT_METADATA_KEYS = {"cov_sources_str", "cov_dims", "cov_projectors_tag", "cov_fusion_tag"}

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _config_path(model_name: str) -> str:
    return os.path.join(_CONFIGS_DIR, f"{model_name}.yml")


def load_config(model_name: str, path: str = None) -> dict:
    """Load full config from YAML."""
    p = path or _config_path(model_name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        out = yaml.safe_load(f)
    out.setdefault("default", {})
    out.setdefault("search_space", {})
    out.setdefault("learned", True)
    return out


def get_default_config(model_name: str, path: str = None) -> dict:
    """Return a deep copy of the default section for the given model."""
    cfg = load_config(model_name, path=path)
    return deepcopy(cfg.get("default", {}))


def _normalize_source_list(source_spec):
    if isinstance(source_spec, str):
        return [part.strip() for part in source_spec.split("+") if part.strip()]
    if isinstance(source_spec, (list, tuple)):
        return [str(part).strip() for part in source_spec if str(part).strip()]
    return []


def _resolve_source_dims(value, source_modalities):
    """
    Coerce scalar/dict PCA settings to match selected source modalities.
    """
    if value is None or not source_modalities:
        return value

    if len(source_modalities) == 1:
        modality = source_modalities[0]
        if isinstance(value, dict):
            if modality not in value:
                raise ValueError(
                    f"Resolved source '{modality}' is missing from PCA config keys {sorted(value)}."
                )
            return value[modality]
        return value

    if isinstance(value, dict):
        missing = [modality for modality in source_modalities if modality not in value]
        if missing:
            raise ValueError(
                f"Multi-source setting {source_modalities} is missing PCA dims for {missing}."
            )
        return {modality: value[modality] for modality in source_modalities}

    return {modality: value for modality in source_modalities}


def resolve_source_dependent_config(config: dict) -> dict:
    """
    Normalize source-dependent PCA settings for nested or flat config dicts.
    """
    resolved = deepcopy(config or {})

    if "model" in resolved or "data" in resolved:
        source_spec = resolved.get("data", {}).get("source")
        source_modalities = _normalize_source_list(source_spec)
        model_cfg = resolved.setdefault("model", {})
        if "n_components_pca" in model_cfg:
            model_cfg["n_components_pca"] = _resolve_source_dims(
                model_cfg["n_components_pca"], source_modalities
            )
        if "n_components_pca_source" in model_cfg:
            model_cfg["n_components_pca_source"] = _resolve_source_dims(
                model_cfg["n_components_pca_source"], source_modalities
            )
        return resolved

    source_spec = resolved.get("source")
    source_modalities = _normalize_source_list(source_spec)
    if "n_components_pca" in resolved:
        resolved["n_components_pca"] = _resolve_source_dims(
            resolved["n_components_pca"], source_modalities
        )
    if "n_components_pca_source" in resolved:
        resolved["n_components_pca_source"] = _resolve_source_dims(
            resolved["n_components_pca_source"], source_modalities
        )
    return resolved


def search_space_to_tune(search_space: dict):
    """Convert declarative YAML search space to Ray Tune sampling objects."""
    try:
        from ray import tune
    except ImportError:
        return {}

    out = {}
    for key, spec in (search_space or {}).items():
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t == "choice":
            out[key] = tune.choice(spec.get("values", []))
        elif t == "loguniform":
            out[key] = tune.loguniform(float(spec["lower"]), float(spec["upper"]))
        elif t == "uniform":
            out[key] = tune.uniform(float(spec["lower"]), float(spec["upper"]))
    return out


def get_search_space(model_name: str, path: str = None) -> dict:
    """Load config and return Ray Tune param_space from the search_space section."""
    cfg = load_config(model_name, path=path)
    return search_space_to_tune(cfg.get("search_space"))


def _model_class(name):
    if name == "Sarwar2020MLP":
        from models.architectures.sarwar2020_mlp import Sarwar2020MLP
        return Sarwar2020MLP
    if name == "Chen2024GCN":
        from models.architectures.graph_based.chen2024_gnn import Chen2024GCN
        return Chen2024GCN
    if name == "NodalGNN":
        from models.architectures.graph_based.nodal_gnn import NodalGNN
        return NodalGNN
    if name == "NodalMLP":
        from models.architectures.graph_based.nodal_mlp import NodalMLP
        return NodalMLP
    if name == "LatentAttnMasked":
        from models.architectures.latent_attention.latent_attn_masked import LatentAttnMasked
        return LatentAttnMasked
    if name == "MaskedLatentPretrainer":
        from models.architectures.latent_attention.masked_latent_pretrainer import MaskedLatentPretrainer
        return MaskedLatentPretrainer
    if name == "CrossModal_ConditionalGaussian":
        from models.architectures.latent_attention.conditional_gaussian import CrossModal_ConditionalGaussian
        return CrossModal_ConditionalGaussian
    if name == "Krakencoder_precomputed":
        from models.architectures.krakencoder_precomputed import KrakencoderPrecomputed
        return KrakencoderPrecomputed
    if name == "CrossModalVAE":
        from models.architectures.crossmodal_vae import CrossModalVAE
        return CrossModalVAE
    if name in {
        "CrossModalPCA",
        "CrossModal_PLS_SVD",
        "CrossModal_PCA_PLS",
        "CrossModal_PCA_PLS_learnable",
        "CrossModal_PCA_PLS_CovProjector",
    }:
        from models.architectures import crossmodal_pca_pls
        return getattr(crossmodal_pca_pls, name)
    raise ValueError(f"Unknown model name: {name}")


def build_model(base, model_name: str = None, model_kwargs: dict = None):
    """
    Build a model instance. model_kwargs must not include 'name' or 'base'.
    If model_name is None, it is taken from model_kwargs.pop('name', None).
    """
    model_kwargs = model_kwargs or {}
    name = model_name or model_kwargs.pop("name", None)
    if not name:
        raise ValueError("model_name or model_kwargs['name'] required")

    kwargs = {k: v for k, v in model_kwargs.items() if k not in ({"name"} | FLAT_METADATA_KEYS)}
    for k in ("l1_l2_tuple", "hidden_dims", "fs_hidden_dims"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])
    if "l1_reg" in kwargs or "l2_reg" in kwargs:
        l1 = float(kwargs.pop("l1_reg", 0.0))
        l2 = float(kwargs.pop("l2_reg", 0.0))
        kwargs.setdefault("l1_l2_tuple", (l1, l2))
    if kwargs.get("device") is None:
        kwargs["device"] = None
    if name == "Krakencoder_precomputed":
        kwargs.pop("device", None)

    return _model_class(name)(base, **kwargs)
