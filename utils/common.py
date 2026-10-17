"""Shared utilities: logging setup, seeding, device selection, library versions, JSON IO."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# timm imports `transformers` (if installed) which warns loudly when its own torch
# minimum is not met. That warning is irrelevant here (transformers is unused).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402

DISCLAIMER = (
    "RESEARCH/EDUCATION PROTOTYPE ONLY. This model is NOT a validated diagnostic system and "
    "must not be used for clinical decisions. Predictions and Grad-CAM maps have not been "
    "clinically validated."
)


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure root logging once (console + optional file)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in DataLoader workers
    """DataLoader ``worker_init_fn`` deriving per-worker seeds from the torch seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def select_device(pref: str = "auto") -> torch.device:
    """Return the torch device for ``pref`` in {auto, cpu, cuda}; cuda falls back to cpu with a warning."""
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        logging.getLogger(__name__).warning("CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def library_versions() -> Dict[str, str]:
    """Versions of the libraries that affect numerical results (stored in checkpoints)."""
    import cv2
    import sklearn
    import timm

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "timm": timm.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _nan_to_none(o: Any) -> Any:
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    return o


def save_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as pretty JSON (NaN -> null), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_nan_to_none(obj), fh, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
