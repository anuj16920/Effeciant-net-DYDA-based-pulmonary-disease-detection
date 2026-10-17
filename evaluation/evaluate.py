"""Evaluate checkpoints on the held-out test split and/or the CV validation folds.

Usage::

    # one checkpoint on the held-out TEST split (final reporting only)
    python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth
    # every fold checkpoint of an experiment: fold k on its own val fold, and all folds on TEST (mean +- std)
    python -m evaluation.evaluate --config configs/config.yaml --experiment hybrid --folds

Outputs go to ``outputs/<experiment>/evaluation/``: metrics JSON, per-source metrics,
predictions CSV and plots (confusion matrix, ROC, per-class bars). A checkpoint is never
evaluated on validation data it was trained on.
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from datasets.dataset import XRayDataset, load_manifest, subset
from datasets.splits import load_splits
from evaluation.metrics import aggregate, compute_metrics, flatten_scalars
from evaluation.plots import plot_confusion_matrix, plot_history, plot_per_class_bars, plot_roc_curves
from training.checkpointing import load_checkpoint, model_from_checkpoint
from training.cross_validation import make_loader, predictions_frame
from training.trainer import predict_loader
from utils.common import DISCLAIMER, load_json, save_json, select_device, setup_logging
from utils.config import get_path, load_config
from utils.synthetic import data_tag

logger = logging.getLogger(__name__)


def per_source_metrics(out: Dict[str, Any], classes: List[str]) -> Dict[str, Any]:
    """Metrics restricted to each source dataset (exposes source-shortcut effects)."""
    src = np.array([m["source_dataset"] for m in out["meta"]])
    res = {}
    for s in sorted(set(src)):
        idx = src == s
        m = compute_metrics(out["y_true"][idx], out["y_pred"][idx], out["probs"][idx], classes)
        res[s] = {k: m[k] for k in ("n_samples", "accuracy", "balanced_accuracy", "macro_f1", "per_class",
                                    "confusion_matrix", "classes_absent_in_truth")}
        res[s]["roc_auc_ovr"] = m.get("roc_auc_ovr")
    return res


def evaluate_checkpoint(cfg: Dict[str, Any], ckpt_path: Path, indices: Sequence[int], split_name: str,
                        out_dir: Path, manifest: Optional[pd.DataFrame] = None,
                        device: Optional[torch.device] = None, plots: bool = True) -> Dict[str, Any]:
    """Evaluate one checkpoint on the manifest rows ``indices``; write JSON/CSV/plots to ``out_dir``."""
    device = device or select_device(cfg["training"]["device"])
    manifest = load_manifest(cfg) if manifest is None else manifest
    model, ck_cfg, ck = model_from_checkpoint(ckpt_path, device)
    if list(ck_cfg["classes"]) != list(cfg["classes"]):
        raise ValueError(f"checkpoint classes {ck_cfg['classes']} != config classes {cfg['classes']}")
    eval_cfg = copy.deepcopy(cfg)
    if ck["preprocessing"] != cfg["preprocessing"]:
        logger.warning("Checkpoint preprocessing differs from config; using the checkpoint's preprocessing.")
    eval_cfg["preprocessing"] = ck["preprocessing"]
    ds = XRayDataset(subset(manifest, indices), eval_cfg, train=False, return_meta=True)
    out = predict_loader(model, make_loader(ds, eval_cfg, False, device), device,
                         use_amp=cfg["training"]["amp"])
    classes = cfg["classes"]
    metrics = compute_metrics(out["y_true"], out["y_pred"], out["probs"], classes)
    tag = data_tag(cfg)
    metrics.update({"split": split_name, "checkpoint": str(ckpt_path), "variant": ck["model_config"]["variant"],
                    "checkpoint_fold": ck.get("fold"), "checkpoint_val_metric": ck.get("metric"),
                    "per_source": per_source_metrics(out, classes),
                    "synthetic_data": bool(tag), "disclaimer": DISCLAIMER})
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "metrics.json")
    predictions_frame(out, classes).to_csv(out_dir / "predictions.csv", index=False)
    rows = [{"source_dataset": s, **{k: v[k] for k in ("n_samples", "accuracy", "balanced_accuracy", "macro_f1")}}
            for s, v in metrics["per_source"].items()]
    pd.DataFrame(rows).to_csv(out_dir / "per_source_metrics.csv", index=False)
    if plots:
        name = f"{ck['model_config']['variant']} / {split_name}"
        plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / "confusion_matrix.png",
                              title=f"Confusion matrix ({name})", tag=tag)
        plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / "confusion_matrix_counts.png",
                              normalize=False, title=f"Confusion matrix ({name})", tag=tag)
        plot_roc_curves(out["y_true"], out["probs"], classes, out_dir / "roc_curves.png",
                        metrics.get("roc_auc_ovr"), tag=tag)
        plot_per_class_bars(metrics, out_dir / "per_class_metrics.png", tag=tag)
    logger.info("%s on %s: n=%d acc=%.4f bal_acc=%.4f macroF1=%.4f %s", ck["model_config"]["variant"], split_name,
                metrics["n_samples"], metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"], tag)
    del model
    return metrics


def evaluate_experiment_folds(cfg: Dict[str, Any], experiment: str, with_test: bool = True,
                              device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Fold k's best.pth on val fold k (validation estimate) and, optionally, every fold on TEST."""
    manifest = load_manifest(cfg)
    splits = load_splits(get_path(cfg, "splits"), manifest)
    base = get_path(cfg, "outputs_dir") / experiment / "evaluation"
    ck_root = get_path(cfg, "checkpoints_dir") / experiment
    val_rows, test_rows = [], []
    for fd in splits["folds"]:
        ck = ck_root / f"fold_{fd['fold']}" / "best.pth"
        if not ck.is_file():
            logger.warning("Missing %s; fold skipped", ck)
            continue
        m = evaluate_checkpoint(cfg, ck, fd["val"], f"val_fold{fd['fold']}", base / f"fold_{fd['fold']}" / "val",
                                manifest, device)
        val_rows.append({"fold": fd["fold"], **flatten_scalars(m)})
        hist = get_path(cfg, "outputs_dir") / experiment / f"fold_{fd['fold']}" / "history.json"
        if hist.is_file():
            plot_history(load_json(hist), base / f"fold_{fd['fold']}" / "training_curves.png", tag=data_tag(cfg))
        if with_test:
            m = evaluate_checkpoint(cfg, ck, splits["test"], "test", base / f"fold_{fd['fold']}" / "test", manifest,
                                    device)
            test_rows.append({"fold": fd["fold"], **flatten_scalars(m)})
    if not val_rows:
        raise FileNotFoundError(f"No fold checkpoints under {ck_root}; run training.cross_validation first.")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "fold"} for r in rows]  # noqa: E731
    summary = {"experiment": experiment, "validation_per_fold": val_rows, "validation_aggregate": aggregate(strip(val_rows)),
               "synthetic_data": bool(data_tag(cfg))}
    if test_rows:
        summary.update({"test_per_fold_model": test_rows, "test_aggregate": aggregate(strip(test_rows)),
                        "note": "test_aggregate = the 5 fold models each evaluated on the same held-out test set."})
    save_json(summary, base / "folds_summary.json")
    pd.DataFrame(val_rows).to_csv(base / "validation_per_fold.csv", index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_csv(base / "test_per_fold_model.csv", index=False)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate DyDA checkpoints (held-out test / validation folds)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="checkpoint to evaluate on the held-out TEST split")
    ap.add_argument("--experiment", default=None, help="experiment name (for --folds and output folder)")
    ap.add_argument("--folds", action="store_true", help="evaluate every fold checkpoint (val fold + test)")
    ap.add_argument("--no-test", action="store_true", help="with --folds: skip the test split")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging()
    if not args.checkpoint and not args.folds:
        ap.error("give --checkpoint and/or --folds")
    if args.checkpoint:
        ck = load_checkpoint(args.checkpoint)
        exp = args.experiment or ck["config"]["experiment"]["name"]
        manifest = load_manifest(cfg)
        splits = load_splits(get_path(cfg, "splits"), manifest)
        evaluate_checkpoint(cfg, Path(args.checkpoint), splits["test"], "test",
                            get_path(cfg, "outputs_dir") / exp / "evaluation" / "test", manifest)
    if args.folds:
        evaluate_experiment_folds(cfg, args.experiment or cfg["experiment"]["name"], with_test=not args.no_test)
    logger.info(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
