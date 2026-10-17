"""Thin CLI dispatcher.

    python main.py prepare  --config configs/config.yaml
    python main.py train    --config configs/config.yaml --model hybrid [--resume] [--folds 5]
    python main.py train-fold --config configs/config.yaml --model dyda --fold 0
    python main.py ablation --config configs/config.yaml [--with-test]
    python main.py evaluate --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth
    python main.py gradcam  --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth --split test
    python main.py predict  --config configs/config.yaml --checkpoint ... --image path\\to\\xray.png [--gradcam]
    python main.py batch-predict --config configs/config.yaml --checkpoint ... --dir path\\to\\folder
    python main.py synthetic --out outputs/_synthetic_smoke      (SYNTHETIC smoke-test workspace)
    python main.py test     [-- extra pytest args]

Every sub-command forwards the remaining arguments to the module's own ``main``.
"""
from __future__ import annotations

import importlib
import sys
from typing import Optional, Sequence

COMMANDS = {
    "prepare": "datasets.prepare_manifest",
    "train": "training.cross_validation",
    "train-fold": "training.train",
    "ablation": "evaluation.ablation",
    "evaluate": "evaluation.evaluate",
    "gradcam": "explainability.gradcam",
    "predict": "inference.predict",
    "batch-predict": "inference.batch_predict",
    "synthetic": "utils.synthetic",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "test":
        import pytest

        return int(pytest.main(["-q", *[a for a in rest if a != "--"]]))
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Choose from: {', '.join([*COMMANDS, 'test'])}", file=sys.stderr)
        return 2
    return int(importlib.import_module(COMMANDS[cmd]).main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
