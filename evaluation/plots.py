"""Matplotlib plots (headless ``Agg`` backend). Every figure takes an optional ``tag`` that
is appended to the title, used to stamp ``[SYNTHETIC DATA]`` on smoke-test outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _title(t: str, tag: str) -> str:
    return f"{t} {tag}".strip()


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: Sequence[Sequence[int]], classes: List[str], path: Path, normalize: bool = True,
                          title: str = "Confusion matrix", tag: str = "") -> Path:
    cm = np.asarray(cm, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        shown = cm / cm.sum(1, keepdims=True) if normalize else cm
    shown = np.nan_to_num(shown)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{shown[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > (0.5 if normalize else shown.max() / 2) else "black")
    ax.set_title(_title(title + (" (row-normalised)" if normalize else ""), tag), fontsize=10)
    return _save(fig, path)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: List[str], path: Path,
                    auc: Optional[Dict[str, Any]] = None, tag: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for k, name in enumerate(classes):
        pos = (np.asarray(y_true) == k).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue  # OvR ROC undefined for this class
        fpr, tpr, _ = roc_curve(pos, y_prob[:, k])
        val = (auc or {}).get("per_class", {}).get(name)
        ax.plot(fpr, tpr, color=PALETTE[k % len(PALETTE)], label=f"{name}" + (f" (AUC {val:.3f})" if val else ""))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(_title("One-vs-rest ROC", tag), fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_per_class_bars(metrics: Dict[str, Any], path: Path, tag: str = "") -> Path:
    classes = list(metrics["per_class"])
    keys = ["precision", "recall", "f1", "specificity"]
    x = np.arange(len(classes))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(keys):
        vals = [metrics["per_class"][c][k] or 0.0 for c in classes]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w, label=k, color=PALETTE[i])
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title(_title("Per-class metrics", tag), fontsize=10)
    return _save(fig, path)


def plot_history(history: List[Dict[str, Any]], path: Path, tag: str = "") -> Path:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.plot(ep, [h["train_loss"] for h in history], label="train", color=PALETTE[0])
    a1.plot(ep, [h["val_loss"] for h in history], label="val", color=PALETTE[1])
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.legend()
    a2.plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1", color=PALETTE[2])
    a2.plot(ep, [h["val_acc"] for h in history], label="val acc", color=PALETTE[3])
    a2.set_xlabel("epoch")
    a2.set_ylim(0, 1.02)
    a2.legend()
    fig.suptitle(_title("Training curves", tag), fontsize=10)
    return _save(fig, path)


def plot_ablation(rows: List[Dict[str, Any]], metrics: List[str], path: Path, tag: str = "",
                  title: str = "Ablation (mean +- std over folds)") -> Path:
    """``rows``: one per model with keys ``label`` and ``<metric>_mean`` / ``<metric>_std``."""
    x = np.arange(len(metrics))
    w = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(1.8 * len(metrics) + 3, 4))
    for i, r in enumerate(rows):
        means = [r.get(f"{m}_mean") or 0.0 for m in metrics]
        stds = [r.get(f"{m}_std") or 0.0 for m in metrics]
        ax.bar(x + (i - (len(rows) - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=r["label"],
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend(fontsize=8, ncol=len(rows), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(_title(title, tag), fontsize=10)
    return _save(fig, path)
