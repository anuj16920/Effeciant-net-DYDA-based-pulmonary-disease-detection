"""Directory inference wrapper -> CSV.

Usage::

    python -m inference.batch_predict --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth \
        --dir "path\\to\\folder" [--recursive] [--gradcam] [--output preds.csv]

RESEARCH USE ONLY -- not a diagnostic device.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import predict


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch inference over a directory (RESEARCH USE ONLY)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    fwd = ["--config", args.config, "--checkpoint", args.checkpoint, "--dir", args.dir]
    fwd += ["--recursive"] if args.recursive else []
    fwd += ["--gradcam"] if args.gradcam else []
    fwd += ["--output", args.output] if args.output else []
    fwd += ["--device", args.device] if args.device else []
    return predict.main(fwd)


if __name__ == "__main__":
    raise SystemExit(main())
