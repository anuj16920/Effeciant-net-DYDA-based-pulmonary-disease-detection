"""Checkpoint save / load / resume.

A checkpoint is a dict with: ``model_state``, ``model_config`` (variant + model section +
num_classes + img_size), ``class_map``, ``preprocessing``, ``config`` (full public config),
``fold``, ``seed``, ``epoch``, ``metric`` (name/value, validation only), ``versions`` and,
for resumable ``last`` checkpoints, optimizer / scheduler / scaler / RNG / trainer state.

Only load checkpoints you created yourself: ``torch.load`` unpickles arbitrary objects.
"""
from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from utils.common import library_versions
from utils.config import public_config

logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT = 1


def rng_state(loader_generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        st["loader"] = loader_generator.get_state()
    return st


def restore_rng_state(st: Dict[str, Any], loader_generator: Optional[torch.Generator] = None) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if loader_generator is not None and "loader" in st:
        loader_generator.set_state(st["loader"])


def make_checkpoint(model: nn.Module, cfg: Dict[str, Any], epoch: int, metric_name: str,
                    metric_value: Optional[float], fold: Optional[int] = None,
                    extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    variant = getattr(model, "variant", cfg["experiment"]["model"])
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": {"variant": variant, "model": copy.deepcopy(cfg["model"]),
                         "num_classes": len(cfg["classes"]), "img_size": cfg["preprocessing"]["img_size"]},
        "class_map": {i: c for i, c in enumerate(cfg["classes"])},
        "preprocessing": copy.deepcopy(cfg["preprocessing"]),
        "config": public_config(cfg),
        "fold": fold,
        "seed": cfg["training"]["seed"],
        "epoch": epoch,
        "metric": {"name": metric_name, "value": metric_value, "split": "validation"},
        "versions": library_versions(),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def save_checkpoint(ckpt: Dict[str, Any], path: Path) -> Path:
    """Atomic save (write to temp file, then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model_state", "model_config", "class_map", "preprocessing"):
        if key not in ckpt:
            raise ValueError(f"{path} is not a DyDA checkpoint (missing '{key}')")
    return ckpt


def config_from_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Config stored in the checkpoint, with the model built offline (weights come from the checkpoint)."""
    cfg = copy.deepcopy(ckpt["config"])
    cfg["model"] = copy.deepcopy(ckpt["model_config"]["model"])
    cfg["model"]["pretrained"] = False
    cfg["preprocessing"] = copy.deepcopy(ckpt["preprocessing"])
    cfg["classes"] = [ckpt["class_map"][i] for i in sorted(ckpt["class_map"])]
    cfg["experiment"]["model"] = ckpt["model_config"]["variant"]
    return cfg


def model_from_checkpoint(path_or_ckpt: Any, device: torch.device | str = "cpu"
                          ) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Rebuild the model from a checkpoint. Returns ``(model.eval(), cfg, ckpt)``."""
    from models.hybrid_model import build_model

    ckpt = path_or_ckpt if isinstance(path_or_ckpt, dict) else load_checkpoint(path_or_ckpt)
    cfg = config_from_checkpoint(ckpt)
    model = build_model(cfg, ckpt["model_config"]["variant"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), cfg, ckpt
