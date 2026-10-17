"""Classification metrics for the 4-class task.

All functions take integer class indices ``y_true``/``y_pred`` of shape ``(N,)`` and,
where needed, probabilities ``y_prob`` of shape ``(N, K)``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

logger = logging.getLogger(__name__)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Macro F1 over all ``num_classes`` labels (absent classes contribute 0)."""
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)),
                                                  average="macro", zero_division=0)
    return float(f1)


def sensitivity_specificity(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-class one-vs-rest sensitivity (TPR) and specificity (TNR) from a confusion matrix.

    ``cm[i, j]`` = number of samples with true class ``i`` predicted as ``j``. Undefined
    values (zero denominator) are NaN.
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    return {"sensitivity": sens, "specificity": spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ovr_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """One-vs-rest ROC-AUC per class plus macro average over the classes where it is defined.

    A class is skipped (value None, reason logged) if it has no positives or no negatives.
    """
    per_class: Dict[str, Optional[float]] = {}
    skipped: Dict[str, str] = {}
    for k, name in enumerate(class_names):
        pos = int((y_true == k).sum())
        neg = int(len(y_true) - pos)
        if pos == 0 or neg == 0:
            reason = f"no {'positive' if pos == 0 else 'negative'} samples"
            skipped[name] = reason
            per_class[name] = None
            logger.info("ROC-AUC for class '%s' skipped: %s", name, reason)
            continue
        per_class[name] = float(roc_auc_score((y_true == k).astype(int), y_prob[:, k]))
    valid = [v for v in per_class.values() if v is not None]
    return {"per_class": per_class, "macro": float(np.mean(valid)) if valid else None,
            "n_classes_used": len(valid), "skipped": skipped}


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[np.ndarray],
                    class_names: List[str]) -> Dict[str, Any]:
    """Full metric set: accuracy, balanced accuracy, macro P/R/F1, per-class P/R/F1/support,
    sensitivity/specificity, confusion matrix and OvR ROC-AUC (when valid)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(class_names)
    labels = list(range(k))
    if len(y_true) == 0:
        raise ValueError("compute_metrics called with no samples")
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ss = sensitivity_specificity(cm)
    present = np.unique(y_true)
    # balanced accuracy = mean recall over classes present in y_true
    bal = float(balanced_accuracy_score(y_true, y_pred)) if len(present) > 1 else float(np.mean(y_true == y_pred))
    out: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": bal,
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i]),
                   "sensitivity": _nan_none(ss["sensitivity"][i]), "specificity": _nan_none(ss["specificity"][i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "classes_absent_in_truth": [class_names[i] for i in labels if i not in present],
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        assert y_prob.shape == (len(y_true), k), f"y_prob shape {y_prob.shape} != ({len(y_true)}, {k})"
        out["roc_auc_ovr"] = ovr_roc_auc(y_true, y_prob, class_names)
    if out["classes_absent_in_truth"]:
        out["note"] = ("Macro P/R/F1 average over all configured classes; classes absent from the ground truth "
                       "contribute 0 and pull the macro average down.")
    return out


def _nan_none(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]


def flatten_scalars(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Scalar metrics (incl. macro ROC-AUC and per-class F1/sens/spec) for tables."""
    row: Dict[str, Optional[float]] = {k: m[k] for k in SCALAR_KEYS}
    row["macro_roc_auc"] = (m.get("roc_auc_ovr") or {}).get("macro")
    for name, pc in m["per_class"].items():
        row[f"f1_{name}"] = pc["f1"]
        row[f"sens_{name}"] = pc["sensitivity"]
        row[f"spec_{name}"] = pc["specificity"]
    return row


def aggregate(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean and sample std (ddof=1) per key across folds, ignoring None values."""
    keys = list(rows[0].keys()) if rows else []
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        out[k] = {"mean": float(vals.mean()) if len(vals) else None,
                  "std": float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else None),
                  "n": int(len(vals))}
    return out
