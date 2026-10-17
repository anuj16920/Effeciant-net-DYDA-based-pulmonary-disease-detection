"""K-fold cross-validation on the saved grouped splits.

Usage::

    python -m training.cross_validation --config configs/config.yaml --model hybrid [--resume] [--folds 0,1]

Per fold (``k``):
    checkpoints/<experiment>/fold_k/{best.pth,last.pth}
    outputs/<experiment>/fold_k/{history.csv,history.json,val_metrics.json,val_predictions.csv}
Aggregate:
    outputs/<experiment>/cv_summary.{json,csv}   (per-fold rows + mean +- std)
    checkpoints/<experiment>/best_model.pth       (fold with the best *validation* macro-F1)

The held-out test split is never touched here.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.dataset import XRayDataset, load_manifest, subset
from datasets.splits import load_splits
from evaluation.metrics import aggregate, compute_metrics, flatten_scalars
from models.hybrid_model import build_model
from utils.common import save_json, seed_everything, seed_worker, select_device, setup_logging
from utils.config import ABLATION_IDS, MODEL_IDS, get_path, load_config, with_model

from .checkpointing import model_from_checkpoint
from .trainer import Trainer, class_weights_from_labels, predict_loader

logger = logging.getLogger(__name__)


def make_loader(ds: XRayDataset, cfg: Dict[str, Any], train: bool, device: torch.device,
                generator: Optional[torch.Generator] = None, batch_size: Optional[int] = None) -> DataLoader:
    t = cfg["training"]
    bs = batch_size or (t["batch_size"] if train else cfg["evaluation"]["batch_size"])
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=t["num_workers"],
                      pin_memory=device.type == "cuda", drop_last=train and len(ds) > bs,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=t["num_workers"] > 0)


def predictions_frame(out: Dict[str, Any], classes: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(out["meta"]) if out["meta"] else pd.DataFrame(index=range(len(out["y_true"])))
    df["y_true"] = [classes[i] for i in out["y_true"]]
    df["y_pred"] = [classes[i] for i in out["y_pred"]]
    df["correct"] = out["y_true"] == out["y_pred"]
    for k, c in enumerate(classes):
        df[f"prob_{c}"] = out["probs"][:, k]
    return df


def run_fold(cfg: Dict[str, Any], manifest: pd.DataFrame, splits: Dict[str, Any], fold: int,
             resume: bool = False, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Train one fold and evaluate its best checkpoint on that fold's validation set."""
    exp = cfg["experiment"]["name"]
    device = device or select_device(cfg["training"]["device"])
    seed = cfg["training"]["seed"]
    seed_everything(seed, cfg["training"]["deterministic"])  # same seed for every model -> identical data order
    fd = splits["folds"][fold]
    tr_df, va_df = subset(manifest, fd["train"]), subset(manifest, fd["val"])
    tr_ds = XRayDataset(tr_df, cfg, train=True)
    va_ds = XRayDataset(va_df, cfg, train=False, return_meta=True)
    gen = torch.Generator().manual_seed(seed)
    tr_dl = make_loader(tr_ds, cfg, True, device, gen)
    va_dl = make_loader(va_ds, cfg, False, device)
    cw = class_weights_from_labels(tr_ds.labels, len(cfg["classes"])) if cfg["training"]["class_weights"] else None

    out_dir = get_path(cfg, "outputs_dir") / exp / f"fold_{fold}"
    ckpt_dir = get_path(cfg, "checkpoints_dir") / exp / f"fold_{fold}"
    model = build_model(cfg)
    trainer = Trainer(model, cfg, device, out_dir, ckpt_dir, fold=fold, class_weights=cw, loader_generator=gen)
    t0 = time.time()
    summary = trainer.fit(tr_dl, va_dl, resume=resume)
    summary["train_seconds"] = round(time.time() - t0, 1)
    if cw is not None:
        summary["class_weights"] = cw.tolist()

    best, _, _ = model_from_checkpoint(trainer.best_path, device)
    out = predict_loader(best, va_dl, device, trainer.use_amp)
    metrics = compute_metrics(out["y_true"], out["y_pred"], out["probs"], cfg["classes"])
    metrics.update({"fold": fold, "split": "validation", "n_train": len(tr_ds), "n_val": len(va_ds)})
    save_json(metrics, out_dir / "val_metrics.json")
    predictions_frame(out, cfg["classes"]).to_csv(out_dir / "val_predictions.csv", index=False)
    summary.update({"metrics": metrics, "best_checkpoint": str(trainer.best_path)})
    del best, model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_cv(cfg: Dict[str, Any], folds: Optional[Sequence[int]] = None, resume: bool = False) -> Dict[str, Any]:
    """Run (a subset of) the CV folds for ``cfg['experiment']['model']`` and aggregate."""
    exp, variant = cfg["experiment"]["name"], cfg["experiment"]["model"]
    manifest = load_manifest(cfg)
    splits = load_splits(get_path(cfg, "splits"), manifest)
    folds = list(range(splits["n_folds"])) if folds is None else list(folds)
    bad = [f for f in folds if not 0 <= f < splits["n_folds"]]
    if bad:
        raise ValueError(f"fold(s) {bad} out of range 0..{splits['n_folds'] - 1}")
    device = select_device(cfg["training"]["device"])
    logger.info("CV %s (%s) on %s, folds %s", variant, ABLATION_IDS[variant], device, folds)
    results = [run_fold(cfg, manifest, splits, f, resume, device) for f in folds]

    rows = [{"fold": r["fold"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
             **flatten_scalars(r["metrics"])} for r in results]
    agg = aggregate([{k: v for k, v in row.items() if k not in ("fold", "best_epoch", "epochs_run")} for row in rows])
    out_dir = get_path(cfg, "outputs_dir") / exp
    table = pd.DataFrame(rows)
    mean_row = {"fold": "mean", **{k: v["mean"] for k, v in agg.items()}}
    std_row = {"fold": "std", **{k: v["std"] for k, v in agg.items()}}
    pd.concat([table, pd.DataFrame([mean_row, std_row])], ignore_index=True).to_csv(out_dir / "cv_summary.csv",
                                                                                   index=False)
    best = max(results, key=lambda r: r["metrics"]["macro_f1"])
    best_dst = get_path(cfg, "checkpoints_dir") / exp / "best_model.pth"
    shutil.copy2(best["best_checkpoint"], best_dst)
    summary = {
        "experiment": exp, "model": variant, "ablation_id": ABLATION_IDS[variant], "device": str(device),
        "folds": folds, "per_fold": rows, "aggregate": agg,
        "best_fold_by_val_macro_f1": best["fold"], "best_model_checkpoint": str(best_dst),
        "note": "Fold metrics are VALIDATION estimates (folds also drive early stopping/checkpoint selection); "
                "the held-out test split is evaluated separately by evaluation.evaluate.",
        "fold_details": [{k: v for k, v in r.items() if k != "metrics"} for r in results],
    }
    save_json(summary, out_dir / "cv_summary.json")
    f1 = agg["macro_f1"]
    logger.info("CV %s macro-F1 (validation) = %.4f +- %.4f over %d fold(s)", variant, f1["mean"] or float("nan"),
                f1["std"] if f1["std"] is not None else float("nan"), f1["n"])
    return summary


def parse_folds(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    if s.isdigit() and "," not in s:
        return list(range(int(s)))  # "--folds 2" -> first two folds
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Grouped K-fold cross-validation for E1-E4")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", choices=MODEL_IDS, default=None, help="overrides experiment.model")
    ap.add_argument("--resume", action="store_true", help="resume each fold from its last.pth")
    ap.add_argument("--folds", default=None, help="'N' = first N folds, or comma list like '0,2'")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.model:
        cfg = with_model(cfg, args.model)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    setup_logging(log_file=get_path(cfg, "outputs_dir") / cfg["experiment"]["name"] / "cv.log")
    run_cv(cfg, parse_folds(args.folds), args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
