"""Grad-CAM on the EfficientNet-B3 feature map (Selvaraju et al., 2017).

Grad-CAM shows which regions of the chosen feature map the predicted (or selected) class
score is most *sensitive* to. It is a model-sensitivity visualisation, not a lesion
localisation method and not clinical evidence. With ``feature_stage=final`` the map is
only 7x7 at 224 input, so heatmaps are coarse.

CLI (batch export of correct / incorrect examples from a split)::

    python -m explainability.gradcam --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth \
        --split test --max-examples 16
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from datasets.preprocessing import prepare_uint8, to_tensor
from utils.common import DISCLAIMER, save_json, select_device, setup_logging

logger = logging.getLogger(__name__)

GRADCAM_CAUTION = ("Grad-CAM highlights regions the model output is sensitive to. It is NOT a validated "
                   "localisation of pathology and NOT clinical proof. " + DISCLAIMER)


def resolve_target_layer(model: nn.Module, name: str = "auto") -> nn.Module:
    """``auto`` -> the backbone layer producing the feature map; otherwise a dotted module name."""
    if name in (None, "", "auto"):
        return model.gradcam_target_layer
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"Grad-CAM target layer '{name}' not found in model") from exc


class GradCAM:
    """Grad-CAM with automatic hook removal (use as a context manager or call :meth:`remove`).

    ``cam(x)`` with ``x`` of shape ``(1, 3, H, W)`` returns ``(heatmap (H, W) in [0, 1], probs (K,), class)``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._acts = out
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._grads = grad

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        assert x.dim() == 4 and x.shape[0] == 1, f"Grad-CAM expects a single image (1,3,H,W), got {tuple(x.shape)}"
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)  # keeps the graph alive even if the backbone is frozen
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            cls = int(probs.argmax()) if target_class is None else int(target_class)
            if not 0 <= cls < logits.shape[1]:
                raise ValueError(f"target_class {cls} out of range 0..{logits.shape[1] - 1}")
            logits[0, cls].backward()
        if self._acts is None or self._grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        acts, grads = self._acts.detach()[0], self._grads.detach()[0]   # (C, h, w)
        weights = grads.mean(dim=(1, 2))                                   # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(0))           # (h, w)
        cam = nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                        align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else np.zeros(tuple(x.shape[-2:]), np.float32)
        self._acts = self._grads = None
        self.model.zero_grad(set_to_none=True)
        return cam.astype(np.float32), probs, cls


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    """``(H, W)`` in [0, 1] -> ``(H, W, 3)`` uint8 RGB jet heatmap."""
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    return cv2.addWeighted(heatmap_rgb(cam), alpha, image_rgb, 1 - alpha, 0)


def annotate(img: np.ndarray, lines: List[str]) -> np.ndarray:
    """Add a black caption strip above ``img`` with the given text lines."""
    scale = max(img.shape[1] / 448.0, 0.35)
    lh = int(22 * scale) + 4
    strip = np.zeros((lh * len(lines) + 6, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(strip, t, (5, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return np.vstack([strip, img])


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"could not encode {path}")
    enc.tofile(str(path))


def explain_image(model: nn.Module, image_path: Path, pcfg: Dict[str, Any], classes: List[str], out_dir: Path,
                  device: torch.device, target_layer: str = "auto", target_class: Optional[int] = None,
                  true_label: Optional[str] = None, alpha: float = 0.4) -> Dict[str, Any]:
    """Grad-CAM for one image; writes original / heatmap / overlay PNGs + metadata JSON into ``out_dir``."""
    rgb = prepare_uint8(image_path, pcfg)      # the exact preprocessed image the model sees (before normalisation)
    x = to_tensor(rgb, pcfg)[None].to(device)
    with GradCAM(model, resolve_target_layer(model, target_layer)) as gc:
        cam, probs, cls = gc(x, target_class)
    pred = int(probs.argmax())
    lines = [f"pred: {classes[pred]} ({probs[pred]:.1%})", f"true: {true_label or 'n/a'}",
             f"CAM target: {classes[cls]}  | research use only"]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_png(out_dir / "original.png", annotate(rgb, lines))
    _write_png(out_dir / "heatmap.png", annotate(heatmap_rgb(cam), lines))
    _write_png(out_dir / "overlay.png", annotate(overlay(rgb, cam, alpha), lines))
    meta = {"image": str(image_path), "predicted_class": classes[pred], "confidence": float(probs[pred]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}, "true_label": true_label,
            "cam_target_class": classes[cls], "target_layer": target_layer,
            "correct": None if true_label is None else classes[pred] == true_label, "caution": GRADCAM_CAUTION}
    save_json(meta, out_dir / "gradcam.json")
    return {**meta, "cam": cam}


def export_examples(model: nn.Module, cfg: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_root: Path,
                    device: torch.device, max_examples: int = 16, target_layer: str = "auto",
                    alpha: float = 0.4) -> Dict[str, int]:
    """Export Grad-CAM for up to ``max_examples`` correct and incorrect predictions.

    ``rows``: dicts with ``image_path`` (absolute), ``label`` (true class name), optional ``name``.
    """
    from utils.config import resolve_path

    classes = cfg["classes"]
    counts = {"correct": 0, "incorrect": 0}
    index = []
    for i, r in enumerate(rows):
        if min(counts.values()) >= max_examples:
            break
        path = resolve_path(cfg, r["image_path"])
        tmp = out_root / "_tmp"
        res = explain_image(model, path, cfg["preprocessing"], classes, tmp, device, target_layer,
                            true_label=r["label"], alpha=alpha)
        bucket = "correct" if res["correct"] else "incorrect"
        if counts[bucket] >= max_examples:
            continue
        dst = out_root / bucket / f"{counts[bucket]:03d}_{Path(path).stem}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(dst / f.name)
        counts[bucket] += 1
        index.append({k: v for k, v in res.items() if k != "cam"} | {"folder": str(dst)})
    tmp = out_root / "_tmp"
    if tmp.is_dir():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    save_json({"counts": counts, "caution": GRADCAM_CAUTION, "examples": index}, out_root / "index.json")
    logger.info("Grad-CAM exported to %s: %s", out_root, counts)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    from datasets.dataset import load_manifest
    from datasets.splits import load_splits
    from training.checkpointing import model_from_checkpoint
    from utils.config import get_path, load_config

    ap = argparse.ArgumentParser(description="Export Grad-CAM for correct/incorrect predictions on a split")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", help="'test' or 'val_foldK' (use the checkpoint's own fold)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--target-layer", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    device = select_device(cfg["training"]["device"])
    model, ck_cfg, ck = model_from_checkpoint(args.checkpoint, device)
    cfg["preprocessing"] = ck["preprocessing"]
    manifest = load_manifest(cfg)
    splits = load_splits(get_path(cfg, "splits"), manifest)
    if args.split == "test":
        idx = splits["test"]
    elif args.split.startswith("val_fold"):
        k = int(args.split[len("val_fold"):])
        if ck.get("fold") is not None and ck["fold"] != k:
            raise ValueError(f"checkpoint was trained on fold {ck['fold']}; val_fold{k} would be training data")
        idx = splits["folds"][k]["val"]
    else:
        raise ValueError("--split must be 'test' or 'val_foldK'")
    g = cfg["evaluation"]["gradcam"]
    rows = manifest.iloc[idx][["image_path", "label"]].to_dict("records")
    out = get_path(cfg, "outputs_dir") / "gradcam" / ck["config"]["experiment"]["name"] / args.split
    export_examples(model, cfg, rows, out, device, args.max_examples or g["max_examples"],
                    args.target_layer or g["target_layer"], g["alpha"])
    logger.info(GRADCAM_CAUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
