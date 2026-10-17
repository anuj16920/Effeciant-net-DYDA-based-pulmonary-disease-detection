"""Single-fold training entry point (debugging / quick runs).

Usage::

    python -m training.train --config configs/config.yaml --model dyda --fold 0 --epochs 2
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from datasets.dataset import load_manifest
from datasets.splits import load_splits
from utils.common import save_json, setup_logging
from utils.config import MODEL_IDS, get_path, load_config, with_model

from .cross_validation import run_fold


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a single CV fold")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", choices=MODEL_IDS, default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.model:
        cfg = with_model(cfg, args.model)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    exp_dir = get_path(cfg, "outputs_dir") / cfg["experiment"]["name"]
    setup_logging(log_file=exp_dir / f"train_fold{args.fold}.log")
    manifest = load_manifest(cfg)
    splits = load_splits(get_path(cfg, "splits"), manifest)
    res = run_fold(cfg, manifest, splits, args.fold, resume=args.resume)
    save_json({k: v for k, v in res.items()}, exp_dir / f"fold_{args.fold}" / "train_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
