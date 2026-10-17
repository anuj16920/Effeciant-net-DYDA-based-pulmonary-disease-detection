"""Predict on a single chest X-ray or a directory of images.

Usage::

    python -m inference.predict --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth \
        --image "path\\to\\xray.png" --gradcam
    python -m inference.predict --config configs/config.yaml --checkpoint ... --dir "path\\to\\folder" [--recursive]

Single image -> JSON (``outputs/predictions/<stem>.json`` unless ``--output``);
directory -> CSV (``outputs/predictions/predictions.csv`` unless ``--output``).
Preprocessing is taken from the checkpoint, so it is identical to training/evaluation.

RESEARCH USE ONLY -- not a diagnostic device.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from datasets.preprocessing import ImageReadError, preprocess_image
from explainability.gradcam import GRADCAM_CAUTION, explain_image
from training.checkpointing import model_from_checkpoint
from utils.common import DISCLAIMER, save_json, select_device, setup_logging
from utils.config import get_path, load_config

logger = logging.getLogger(__name__)


class Predictor:
    """Loads a checkpoint once and predicts class probabilities (softmax) for image files."""

    def __init__(self, checkpoint: Path, device: str | torch.device = "auto") -> None:
        self.device = select_device(device) if isinstance(device, str) else device
        self.model, self.cfg, self.ckpt = model_from_checkpoint(checkpoint, self.device)
        self.checkpoint = Path(checkpoint)
        self.classes: List[str] = self.cfg["classes"]
        self.pcfg = self.ckpt["preprocessing"]

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, S, S)`` -> probabilities ``(B, K)``."""
        return torch.softmax(self.model(x.to(self.device)), dim=1).cpu()

    def predict(self, image_path: Path) -> Dict[str, Any]:
        probs = self.predict_tensor(preprocess_image(image_path, self.pcfg)[None])[0]
        k = int(probs.argmax())
        return {"image": str(image_path), "predicted_class": self.classes[k], "confidence": float(probs[k]),
                "probabilities": {c: float(p) for c, p in zip(self.classes, probs.tolist())},
                "model_variant": self.ckpt["model_config"]["variant"], "checkpoint": str(self.checkpoint),
                "disclaimer": DISCLAIMER}

    def gradcam(self, image_path: Path, out_dir: Path, target_class: Optional[str] = None,
                alpha: float = 0.4) -> Dict[str, Any]:
        tc = None if target_class is None else self.classes.index(target_class)
        res = explain_image(self.model, image_path, self.pcfg, self.classes, out_dir, self.device,
                            target_class=tc, alpha=alpha)
        return {"gradcam_dir": str(out_dir), "gradcam_target_class": res["cam_target_class"],
                "gradcam_caution": GRADCAM_CAUTION}


def list_images(folder: Path, exts: Sequence[str], recursive: bool = False) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in {e.lower() for e in exts})


def predict_directory(pred: Predictor, folder: Path, exts: Sequence[str], recursive: bool = False,
                      gradcam_root: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for p in list_images(folder, exts, recursive):
        try:
            r = pred.predict(p)
            row = {"image": r["image"], "predicted_class": r["predicted_class"], "confidence": r["confidence"],
                   **{f"prob_{c}": v for c, v in r["probabilities"].items()}, "error": ""}
            if gradcam_root is not None:
                row["gradcam_dir"] = pred.gradcam(p, gradcam_root / p.stem)["gradcam_dir"]
        except ImageReadError as exc:
            logger.warning("Skipping %s: %s", p, exc)
            row = {"image": str(p), "predicted_class": "", "confidence": float("nan"), "error": str(exc)}
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DyDA chest X-ray inference (RESEARCH USE ONLY)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image path")
    src.add_argument("--dir", help="directory of images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--gradcam", action="store_true", help="also export Grad-CAM heatmap/overlay")
    ap.add_argument("--target-class", default=None, help="Grad-CAM target class (default: predicted)")
    ap.add_argument("--output", default=None, help="output JSON (single) or CSV (directory)")
    ap.add_argument("--device", default=None, help="auto|cpu|cuda (default: from config)")
    args = ap.parse_args(argv)
    setup_logging()
    logger.warning(DISCLAIMER)
    cfg = load_config(args.config)
    pred = Predictor(Path(args.checkpoint), args.device or cfg["training"]["device"])
    if args.target_class is not None and args.target_class not in pred.classes:
        ap.error(f"--target-class must be one of {pred.classes}")
    out_root = get_path(cfg, "outputs_dir") / "predictions"
    alpha = cfg["evaluation"]["gradcam"]["alpha"]
    if args.image:
        img = Path(args.image)
        res = pred.predict(img)
        if args.gradcam:
            res.update(pred.gradcam(img, out_root / "gradcam" / img.stem, args.target_class, alpha))
        out = Path(args.output) if args.output else out_root / f"{img.stem}.json"
        save_json(res, out)
        print(json.dumps(res, indent=2))
        logger.info("Saved %s", out)
    else:
        folder = Path(args.dir)
        if not folder.is_dir():
            ap.error(f"--dir {folder} is not a directory")
        df = predict_directory(pred, folder, cfg["manifest"]["image_extensions"], args.recursive,
                               out_root / "gradcam" if args.gradcam else None)
        out = Path(args.output) if args.output else out_root / "predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Predicted %d image(s) -> %s", len(df), out)
    logger.warning(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
