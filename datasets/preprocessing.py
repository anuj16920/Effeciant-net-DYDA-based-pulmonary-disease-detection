"""Deterministic image preprocessing shared by train / val / test / inference.

Pipeline (identical everywhere)::

    load (PNG/JPEG/..., gray/RGB/RGBA, 8/16-bit) -> CLAHE on luminance -> 3-channel uint8
      -> resize to img_size x img_size                          [= :func:`prepare_uint8`]
      -> (training only: augmentation on the uint8 image)
      -> float /255 -> normalize(mean, std) -> CHW tensor       [= :func:`to_tensor`]

:func:`preprocess_image` = :func:`prepare_uint8` + :func:`to_tensor` (no augmentation) and is
what evaluation and inference use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
import torch

ImageInput = Union[str, Path, np.ndarray]


class ImageReadError(IOError):
    """Raised when an image file is missing or cannot be decoded."""


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image as uint8, either ``(H, W)`` grayscale or ``(H, W, 3)`` RGB.

    Uses ``np.fromfile`` + ``cv2.imdecode`` so Windows paths with spaces / non-ASCII work.
    16-bit images are min-max scaled to 8-bit; alpha channels are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageReadError(f"Image not found: {path}")
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"Cannot read {path}: {exc}") from exc
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None
    if img is None:
        raise ImageReadError(f"Cannot decode image (corrupt or unsupported format): {path}")
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = np.zeros_like(img) if hi <= lo else (img - lo) / (hi - lo) * 255.0
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim not in (2, 3):
        raise ImageReadError(f"Unexpected image shape {img.shape}: {path}")
    return img


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    """CLAHE on the luminance channel (gray: directly; RGB: L channel of LAB)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def to_three_channel(img: np.ndarray) -> np.ndarray:
    return np.repeat(img[:, :, None], 3, axis=2) if img.ndim == 2 else img


def prepare_uint8(image: ImageInput, pcfg: Dict[str, Any]) -> np.ndarray:
    """Load (if path) -> CLAHE -> 3-channel -> resize. Returns ``(S, S, 3)`` uint8 RGB."""
    img = load_image(image) if isinstance(image, (str, Path)) else np.asarray(image)
    if img.dtype != np.uint8:
        raise ValueError(f"array input must be uint8, got {img.dtype}")
    if pcfg["clahe"]["enabled"]:
        img = apply_clahe(img, pcfg["clahe"]["clip_limit"], pcfg["clahe"]["tile_grid"])
    img = to_three_channel(img)
    size = int(pcfg["img_size"])
    if img.shape[:2] != (size, size):
        interp = cv2.INTER_AREA if min(img.shape[:2]) >= size else cv2.INTER_LINEAR
        img = cv2.resize(img, (size, size), interpolation=interp)
    return np.ascontiguousarray(img)


def to_tensor(img: np.ndarray, pcfg: Dict[str, Any]) -> torch.Tensor:
    """``(S, S, 3)`` uint8 -> normalized float32 tensor ``(3, S, S)``."""
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8, f"bad image {img.shape} {img.dtype}"
    arr = img.astype(np.float32) / 255.0
    arr = (arr - np.asarray(pcfg["mean"], np.float32)) / np.asarray(pcfg["std"], np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def denormalize(t: torch.Tensor, pcfg: Dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`to_tensor` -> ``(S, S, 3)`` uint8 (for visualisation)."""
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.asarray(pcfg["std"], np.float32) + np.asarray(pcfg["mean"], np.float32)
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def preprocess_image(image: ImageInput, pcfg: Dict[str, Any]) -> torch.Tensor:
    """Full deterministic preprocessing (no augmentation) -> ``(3, S, S)`` tensor."""
    return to_tensor(prepare_uint8(image, pcfg), pcfg)


def preprocessing_signature(pcfg: Dict[str, Any]) -> str:
    """Short hash of the parameters that affect :func:`prepare_uint8` output (cache key)."""
    key = {"img_size": pcfg["img_size"], "clahe": pcfg["clahe"], "cv2": cv2.__version__, "v": 1}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
