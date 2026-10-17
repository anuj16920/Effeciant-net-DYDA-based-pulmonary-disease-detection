"""Training-only, mild augmentation applied to the preprocessed uint8 image.

Mild rotation, scale and translation (single affine warp), then mild brightness/contrast.
Horizontal flips are off by default because cardiac/mediastinal laterality is
anatomically meaningful on chest X-rays.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


class TrainAugmentation:
    """Callable ``(S,S,3) uint8 -> (S,S,3) uint8``. Never used for val/test/inference."""

    def __init__(self, acfg: Dict[str, Any]) -> None:
        self.cfg = dict(acfg)

    def __call__(self, img: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        c = self.cfg
        if not c["enabled"]:
            return img
        rng = rng if rng is not None else np.random.default_rng(np.random.randint(0, 2**31 - 1))
        if rng.random() >= c["p"]:
            return img
        h, w = img.shape[:2]
        angle = rng.uniform(-c["rotation_deg"], c["rotation_deg"])
        scale = rng.uniform(c["scale"][0], c["scale"][1])
        tx, ty = rng.uniform(-c["translate"], c["translate"], size=2) * (w, h)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        m[:, 2] += (tx, ty)
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if c["horizontal_flip"] and rng.random() < 0.5:
            out = out[:, ::-1]
        alpha = 1.0 + rng.uniform(-c["contrast"], c["contrast"])
        beta = 255.0 * rng.uniform(-c["brightness"], c["brightness"])
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)
