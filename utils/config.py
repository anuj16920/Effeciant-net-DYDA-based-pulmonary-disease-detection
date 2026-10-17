"""YAML configuration loading and validation.

The config is kept as a plain nested ``dict`` so it can be stored verbatim inside
checkpoints. :func:`load_config` validates every section against an explicit
schema (unknown keys, missing keys, wrong types and out-of-range values all raise
:class:`ConfigError` with the offending dotted key) and records the project root
so relative paths can be resolved with :func:`resolve_path`.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

MODEL_IDS = ("backbone", "cbam", "dyda", "hybrid")
ABLATION_IDS = {"backbone": "E1", "cbam": "E2", "dyda": "E3", "hybrid": "E4"}
FEATURE_STAGES = ("final", "stride16")
LABEL_FROM = ("folder", "filename")

PROJECT_ROOT_KEY = "_project_root"
CONFIG_PATH_KEY = "_config_path"


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or has invalid values."""


# --------------------------------------------------------------------------- schema helpers
Number = (int, float)


def _req(cfg: Dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(cfg, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(cfg).__name__}")
    if key not in cfg:
        raise ConfigError(f"Missing required config key '{where}.{key}'".lstrip("."))
    return cfg[key]


def _check_keys(cfg: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where}': {sorted(unknown)}. Allowed: {sorted(allowed)}")


def _typed(cfg: Dict[str, Any], key: str, types: Any, where: str,
           check: Optional[Callable[[Any], bool]] = None, msg: str = "", nullable: bool = False) -> Any:
    val = _req(cfg, key, where)
    full = f"{where}.{key}" if where else key
    if val is None:
        if nullable:
            return val
        raise ConfigError(f"'{full}' must not be null")
    # bool is a subclass of int; reject it where a number is expected.
    if isinstance(val, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        raise ConfigError(f"'{full}' must be {types}, got bool")
    if not isinstance(val, types):
        raise ConfigError(f"'{full}' must be of type {types}, got {type(val).__name__} ({val!r})")
    if check is not None and not check(val):
        raise ConfigError(f"'{full}' has invalid value {val!r}: {msg}")
    return val


def _pos(x: Any) -> bool:
    return x > 0


def _nonneg(x: Any) -> bool:
    return x >= 0


def _prob(x: Any) -> bool:
    return 0.0 <= x <= 1.0


# --------------------------------------------------------------------------- section validators
def _validate_experiment(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "experiment", "")
    _check_keys(sec, {"name", "model"}, "experiment")
    _typed(sec, "name", str, "experiment", lambda s: len(s.strip()) > 0, "must be non-empty")
    _typed(sec, "model", str, "experiment", lambda s: s in MODEL_IDS, f"must be one of {MODEL_IDS}")


def _validate_paths(cfg: Dict[str, Any]) -> None:
    keys = {"raw_dir", "processed_dir", "manifest_dir", "manifest", "splits", "data_report",
            "outputs_dir", "checkpoints_dir"}
    sec = _req(cfg, "paths", "")
    _check_keys(sec, keys, "paths")
    for k in keys:
        _typed(sec, k, str, "paths", lambda s: len(s) > 0, "must be non-empty")


def _validate_classes(cfg: Dict[str, Any]) -> None:
    classes = _typed(cfg, "classes", list, "", lambda c: len(c) >= 2, "need at least 2 classes")
    if not all(isinstance(c, str) and c for c in classes):
        raise ConfigError("'classes' must be a list of non-empty strings")
    if len(set(classes)) != len(classes):
        raise ConfigError(f"'classes' contains duplicates: {classes}")


def _validate_sources(cfg: Dict[str, Any]) -> None:
    sources = _typed(cfg, "sources", dict, "", lambda d: len(d) > 0, "need at least one source")
    classes = set(cfg["classes"])
    allowed = {"enabled", "root", "label_from", "label_map", "filename_label_regex",
               "patient_id_regex", "pneumonia_subtypes"}
    for name, src in sources.items():
        where = f"sources.{name}"
        if not isinstance(src, dict):
            raise ConfigError(f"'{where}' must be a mapping")
        _check_keys(src, allowed, where)
        _typed(src, "enabled", bool, where)
        _typed(src, "root", str, where, lambda s: len(s) > 0, "must be non-empty")
        mode = _typed(src, "label_from", str, where, lambda s: s in LABEL_FROM, f"must be one of {LABEL_FROM}")
        lmap = _typed(src, "label_map", dict, where, lambda d: len(d) > 0, "must not be empty")
        for raw, target in lmap.items():
            if target is not None and target not in classes:
                raise ConfigError(
                    f"'{where}.label_map' maps '{raw}' -> '{target}', which is not in classes {sorted(classes)}. "
                    "Use null to exclude a raw label explicitly.")
        rx = _typed(src, "filename_label_regex", str, where, nullable=True)
        if mode == "filename" and not rx:
            raise ConfigError(f"'{where}.filename_label_regex' is required when label_from=filename")
        _typed(src, "patient_id_regex", str, where, nullable=True)
        if "pneumonia_subtypes" in src:
            sub = _typed(src, "pneumonia_subtypes", dict, where)
            for k, v in sub.items():
                if not isinstance(v, bool):
                    raise ConfigError(f"'{where}.pneumonia_subtypes.{k}' must be true/false")


def _validate_manifest(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "manifest", "")
    _check_keys(sec, {"image_extensions", "exclude_dirs", "phash_hamming_threshold",
                      "drop_label_conflicts", "min_image_side", "hash_workers"}, "manifest")
    exts = _typed(sec, "image_extensions", list, "manifest", lambda l: len(l) > 0, "must not be empty")
    if not all(isinstance(e, str) and e.startswith(".") for e in exts):
        raise ConfigError("'manifest.image_extensions' entries must be strings starting with '.'")
    _typed(sec, "exclude_dirs", list, "manifest")
    _typed(sec, "phash_hamming_threshold", int, "manifest", lambda x: 0 <= x <= 32, "must be in [0, 32]")
    _typed(sec, "drop_label_conflicts", bool, "manifest")
    _typed(sec, "min_image_side", int, "manifest", _pos, "must be > 0")
    _typed(sec, "hash_workers", int, "manifest", _pos, "must be > 0")


def _validate_preprocessing(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "preprocessing", "")
    _check_keys(sec, {"img_size", "clahe", "mean", "std", "cache"}, "preprocessing")
    _typed(sec, "cache", bool, "preprocessing")
    _typed(sec, "img_size", int, "preprocessing", lambda x: x >= 32, "must be >= 32")
    cl = _typed(sec, "clahe", dict, "preprocessing")
    _check_keys(cl, {"enabled", "clip_limit", "tile_grid"}, "preprocessing.clahe")
    _typed(cl, "enabled", bool, "preprocessing.clahe")
    _typed(cl, "clip_limit", Number, "preprocessing.clahe", _pos, "must be > 0")
    _typed(cl, "tile_grid", list, "preprocessing.clahe",
           lambda l: len(l) == 2 and all(isinstance(v, int) and v > 0 for v in l), "must be [int>0, int>0]")
    for k in ("mean", "std"):
        _typed(sec, k, list, "preprocessing",
               lambda l: len(l) == 3 and all(isinstance(v, Number) for v in l), "must be 3 numbers")
    if any(v <= 0 for v in sec["std"]):
        raise ConfigError("'preprocessing.std' values must be > 0")


def _validate_augmentation(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "augmentation", "")
    w = "augmentation"
    _check_keys(sec, {"enabled", "p", "rotation_deg", "scale", "translate", "brightness",
                      "contrast", "horizontal_flip"}, w)
    _typed(sec, "enabled", bool, w)
    _typed(sec, "p", Number, w, _prob, "must be in [0, 1]")
    _typed(sec, "rotation_deg", Number, w, lambda x: 0 <= x <= 45, "must be in [0, 45]")
    _typed(sec, "scale", list, w, lambda l: len(l) == 2 and 0 < l[0] <= l[1], "must be [lo, hi] with 0 < lo <= hi")
    _typed(sec, "translate", Number, w, lambda x: 0 <= x < 0.5, "must be in [0, 0.5)")
    _typed(sec, "brightness", Number, w, lambda x: 0 <= x < 1, "must be in [0, 1)")
    _typed(sec, "contrast", Number, w, lambda x: 0 <= x < 1, "must be in [0, 1)")
    _typed(sec, "horizontal_flip", bool, w)


def _validate_model(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "model", "")
    w = "model"
    _check_keys(sec, {"backbone", "pretrained", "feature_stage", "freeze_backbone", "dropout",
                      "dyda", "cbam", "swin"}, w)
    _typed(sec, "backbone", str, w, lambda s: s == "efficientnet_b3",
           "the spec mandates efficientnet_b3 (no substitutions)")
    _typed(sec, "pretrained", bool, w)
    _typed(sec, "feature_stage", str, w, lambda s: s in FEATURE_STAGES, f"must be one of {FEATURE_STAGES}")
    _typed(sec, "freeze_backbone", bool, w)
    _typed(sec, "dropout", Number, w, lambda x: 0 <= x < 1, "must be in [0, 1)")
    for att in ("dyda", "cbam"):
        a = _typed(sec, att, dict, w)
        keys = {"reduction", "spatial_kernel"} | ({"gate_hidden"} if att == "dyda" else set())
        _check_keys(a, keys, f"{w}.{att}")
        _typed(a, "reduction", int, f"{w}.{att}", _pos, "must be > 0")
        _typed(a, "spatial_kernel", int, f"{w}.{att}", lambda k: k > 0 and k % 2 == 1, "must be a positive odd int")
        if att == "dyda":
            _typed(a, "gate_hidden", int, f"{w}.{att}", _pos, "must be > 0")
    s = _typed(sec, "swin", dict, w)
    ws = f"{w}.swin"
    _check_keys(s, {"embed_dim", "depth", "num_heads", "window_size", "shift_size", "upsample_size",
                    "mlp_ratio", "drop_path", "attn_drop", "proj_drop"}, ws)
    dim = _typed(s, "embed_dim", int, ws, _pos, "must be > 0")
    _typed(s, "depth", int, ws, lambda x: x >= 2, "must be >= 2 so both W-MSA and SW-MSA blocks exist")
    heads = _typed(s, "num_heads", int, ws, _pos, "must be > 0")
    if dim % heads:
        raise ConfigError(f"'{ws}.embed_dim' ({dim}) must be divisible by num_heads ({heads})")
    win = _typed(s, "window_size", int, ws, lambda x: x >= 2, "must be >= 2")
    _typed(s, "shift_size", int, ws, lambda x: 0 < x < win, f"must satisfy 0 < shift_size < window_size ({win})")
    up = _typed(s, "upsample_size", int, ws, _pos, "must be > 0", nullable=True)
    if up is not None and (up % win or up // win < 2):
        raise ConfigError(f"'{ws}.upsample_size' ({up}) must be a multiple of window_size ({win}) "
                          "giving at least 2x2 windows, otherwise shifted windows are a no-op")
    _typed(s, "mlp_ratio", Number, ws, _pos, "must be > 0")
    for k in ("drop_path", "attn_drop", "proj_drop"):
        _typed(s, k, Number, ws, lambda x: 0 <= x < 1, "must be in [0, 1)")


def _validate_training(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "training", "")
    w = "training"
    _check_keys(sec, {"seed", "deterministic", "test_fraction", "folds", "batch_size", "epochs", "lr",
                      "min_lr", "weight_decay", "num_workers", "amp", "grad_clip", "class_weights",
                      "early_stopping", "device"}, w)
    _typed(sec, "seed", int, w, _nonneg, "must be >= 0")
    _typed(sec, "deterministic", bool, w)
    _typed(sec, "test_fraction", Number, w, lambda x: 0 < x < 0.5, "must be in (0, 0.5)")
    _typed(sec, "folds", int, w, lambda x: x >= 2, "must be >= 2")
    _typed(sec, "batch_size", int, w, _pos, "must be > 0")
    _typed(sec, "epochs", int, w, _pos, "must be > 0")
    lr = _typed(sec, "lr", Number, w, _pos, "must be > 0")
    _typed(sec, "min_lr", Number, w, lambda x: 0 <= x <= lr, "must be in [0, lr]")
    _typed(sec, "weight_decay", Number, w, _nonneg, "must be >= 0")
    _typed(sec, "num_workers", int, w, _nonneg, "must be >= 0")
    _typed(sec, "amp", bool, w)
    _typed(sec, "grad_clip", Number, w, _pos, "must be > 0", nullable=True)
    _typed(sec, "class_weights", bool, w)
    es = _typed(sec, "early_stopping", dict, w)
    _check_keys(es, {"patience", "min_delta"}, f"{w}.early_stopping")
    _typed(es, "patience", int, f"{w}.early_stopping", _pos, "must be > 0")
    _typed(es, "min_delta", Number, f"{w}.early_stopping", _nonneg, "must be >= 0")
    _typed(sec, "device", str, w, lambda s: s in ("auto", "cpu", "cuda"), "must be auto|cpu|cuda")


def _validate_evaluation(cfg: Dict[str, Any]) -> None:
    sec = _req(cfg, "evaluation", "")
    _check_keys(sec, {"batch_size", "gradcam"}, "evaluation")
    _typed(sec, "batch_size", int, "evaluation", _pos, "must be > 0")
    g = _typed(sec, "gradcam", dict, "evaluation")
    _check_keys(g, {"target_layer", "max_examples", "alpha"}, "evaluation.gradcam")
    _typed(g, "target_layer", str, "evaluation.gradcam")
    _typed(g, "max_examples", int, "evaluation.gradcam", _pos, "must be > 0")
    _typed(g, "alpha", Number, "evaluation.gradcam", _prob, "must be in [0, 1]")


TOP_LEVEL = {"experiment", "paths", "classes", "sources", "manifest", "preprocessing", "augmentation",
             "model", "training", "evaluation"}


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a config dict in place and return it. Raises :class:`ConfigError`."""
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(cfg).__name__}")
    _check_keys({k: v for k, v in cfg.items() if not k.startswith("_")}, TOP_LEVEL, "<root>")
    _validate_experiment(cfg)
    _validate_paths(cfg)
    _validate_classes(cfg)
    _validate_sources(cfg)
    _validate_manifest(cfg)
    _validate_preprocessing(cfg)
    _validate_augmentation(cfg)
    _validate_model(cfg)
    _validate_training(cfg)
    _validate_evaluation(cfg)
    return cfg


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load, optionally override, and validate a YAML config.

    Args:
        path: path to the YAML file.
        overrides: flat mapping of dotted keys to values, e.g. ``{"experiment.model": "dyda"}``.

    Returns:
        The validated config dict, with ``_project_root`` and ``_config_path`` added.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
    if cfg is None:
        raise ConfigError(f"Config file is empty: {path}")
    for dotted, value in (overrides or {}).items():
        set_by_path(cfg, dotted, value)
    validate_config(cfg)
    # configs/config.yaml -> project root is the parent of configs/
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg[PROJECT_ROOT_KEY] = str(root)
    cfg[CONFIG_PATH_KEY] = str(path)
    return cfg


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` for ``dotted='a.b'``; intermediate keys must exist."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise ConfigError(f"Override key '{dotted}' does not exist in the config")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"Override key '{dotted}' does not exist in the config")
    node[parts[-1]] = value


def project_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get(PROJECT_ROOT_KEY, Path.cwd()))


def resolve_path(cfg: Dict[str, Any], p: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else (project_root(cfg) / p).resolve()


def get_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` to an absolute path."""
    return resolve_path(cfg, cfg["paths"][key])


def experiment_name(cfg: Dict[str, Any]) -> str:
    return cfg["experiment"]["name"]


def with_model(cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Deep copy of ``cfg`` with ``experiment.model`` and ``experiment.name`` set to ``model_id``."""
    if model_id not in MODEL_IDS:
        raise ConfigError(f"Unknown model '{model_id}'. Choose from {MODEL_IDS}")
    new = copy.deepcopy(cfg)
    new["experiment"]["model"] = model_id
    new["experiment"]["name"] = model_id
    return new


def public_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Config without private bookkeeping keys (for saving into checkpoints / outputs)."""
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}
