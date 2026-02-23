"""
Config helpers: load model config from YAML, resolve search space to Ray Tune API, build model.
Config files live in models/configs/<model_name>.yml.
"""
import os
from copy import deepcopy
import yaml

# Keys that belong to trainer config (used when splitting flat Tune config).
TRAINER_KEYS = {"lr", "loss_type", "loss_alpha", "loss_beta", "max_epochs", "batch_size", "log_every"}

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _config_path(model_name: str) -> str:
    return os.path.join(_CONFIGS_DIR, f"{model_name}.yml")


def load_config(model_name: str, path: str = None) -> dict:
    """
    Load full config (default + search_space + learned) from YAML.
    path: if set, load from this fle; else models/configs/<model_name>.yml.
    """
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
    """Return a deep copy of the default section (model + trainer) for the given model."""
    cfg = load_config(model_name, path=path)
    return deepcopy(cfg.get("default", {}))


def search_space_to_tune(search_space: dict):
    """
    Convert declarative search_space from YAML to Ray Tune sampling API.
    search_space: { param_name: { type: choice|loguniform|uniform, values: [...] or lower/upper } }
    Returns dict of param_name -> tune.* object (or empty dict if ray not available).
    """
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
    """Load config and return Ray Tune param_space (tune.choice etc.) from search_space section."""
    cfg = load_config(model_name, path=path)
    return search_space_to_tune(cfg.get("search_space"))


def build_model(base, model_name: str = None, model_kwargs: dict = None):
    """
    Build a model instance. model_kwargs must not include 'name' or 'base'.
    If model_name is None, it is taken from model_kwargs.pop('name', None).
    """
    from models import models as models_module

    model_kwargs = model_kwargs or {}
    name = model_name or model_kwargs.pop("name", None)
    if not name:
        raise ValueError("model_name or model_kwargs['name'] required")
    kwargs = {k: v for k, v in model_kwargs.items() if k != "name"}
    # YAML may give lists for tuples (e.g. l1_l2_tuple, hidden_dims)
    for k in ("l1_l2_tuple", "hidden_dims"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])
    if kwargs.get("device") is None:
        kwargs["device"] = None
    cls = getattr(models_module, name)
    return cls(base, **kwargs)