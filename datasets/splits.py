"""Grouped, stratified splitting: held-out test first, then K-fold CV on the remainder.

* Test split: the first fold of ``StratifiedGroupKFold(n_splits=round(1/test_fraction))``
  (so the realised test fraction is approximately, not exactly, ``test_fraction``).
* CV: ``StratifiedGroupKFold(n_splits=folds)`` on the development (non-test) rows.

Zero group overlap between test/dev and between every fold's train/val is asserted.
Indices refer to rows of the manifest; a fingerprint of the manifest is stored so stale
splits are detected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils.common import load_json, save_json


class SplitLeakageError(AssertionError):
    """Raised when a group appears on both sides of a split."""


def manifest_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in df[["image_path", "sha256", "label", "group_id"]].itertuples(index=False):
        h.update("|".join(map(str, row)).encode())
        h.update(b"\n")
    return h.hexdigest()


def assert_no_group_overlap(groups: pd.Series, a: np.ndarray, b: np.ndarray, what: str) -> None:
    overlap = set(groups.iloc[a]) & set(groups.iloc[b])
    if overlap:
        raise SplitLeakageError(f"{what}: {len(overlap)} group(s) on both sides, e.g. {sorted(overlap)[:5]}")
    if set(a.tolist()) & set(b.tolist()):
        raise SplitLeakageError(f"{what}: overlapping row indices")


def _dist(df: pd.DataFrame, idx: np.ndarray, classes: List[str]) -> Dict[str, int]:
    vc = df["label"].iloc[idx].value_counts()
    return {c: int(vc.get(c, 0)) for c in classes}


def make_splits(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create held-out test + K folds. Raises ValueError if there are too few groups."""
    classes = cfg["classes"]
    tcfg = cfg["training"]
    seed, k = tcfg["seed"], tcfg["folds"]
    unknown = set(df["label"]) - set(classes)
    if unknown:
        raise ValueError(f"manifest has labels not in config classes: {sorted(unknown)}")
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = df["group_id"].reset_index(drop=True)
    n_groups = groups.nunique()
    n_test = max(2, int(round(1.0 / tcfg["test_fraction"])))
    if n_groups < n_test + k:
        raise ValueError(f"only {n_groups} groups; need at least {n_test + k} for a held-out test split "
                         f"(1/{n_test}) plus {k}-fold CV")
    idx = np.arange(len(df))
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    dev, test = next(outer.split(idx, y, groups))
    assert_no_group_overlap(groups, dev, test, "dev vs test")

    inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    seen_val: List[int] = []
    for f, (tr, va) in enumerate(inner.split(dev, y[dev], groups.iloc[dev])):
        tr_idx, va_idx = np.sort(dev[tr]), np.sort(dev[va])
        assert_no_group_overlap(groups, tr_idx, va_idx, f"fold {f} train vs val")
        assert_no_group_overlap(groups, tr_idx, test, f"fold {f} train vs test")
        assert_no_group_overlap(groups, va_idx, test, f"fold {f} val vs test")
        seen_val.extend(va_idx.tolist())
        folds.append({"fold": f, "train": tr_idx.tolist(), "val": va_idx.tolist()})
    if sorted(seen_val) != sorted(dev.tolist()):
        raise SplitLeakageError("validation folds do not partition the development set")

    dist = {"test": _dist(df, test, classes), "dev": _dist(df, dev, classes)}
    for fd in folds:
        dist[f"fold{fd['fold']}_train"] = _dist(df, np.array(fd["train"]), classes)
        dist[f"fold{fd['fold']}_val"] = _dist(df, np.array(fd["val"]), classes)
    return {
        "manifest_fingerprint": manifest_fingerprint(df),
        "seed": seed,
        "test_fraction_requested": tcfg["test_fraction"],
        "test_fraction_actual": round(len(test) / len(df), 4),
        "n_folds": k,
        "test": np.sort(test).tolist(),
        "dev": np.sort(dev).tolist(),
        "folds": folds,
        "class_distribution": dist,
    }


def save_splits(splits: Dict[str, Any], path: Path) -> Path:
    return save_json(splits, path)


def load_splits(path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    """Load splits and verify they belong to this manifest (and still have no leakage)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Splits file not found: {path}. Run `python -m datasets.prepare_manifest` first.")
    splits = load_json(path)
    if splits["manifest_fingerprint"] != manifest_fingerprint(df):
        raise ValueError(f"{path} was built for a different manifest. Re-run datasets.prepare_manifest.")
    groups = df["group_id"].reset_index(drop=True)
    test = np.array(splits["test"], dtype=int)
    for fd in splits["folds"]:
        tr, va = np.array(fd["train"], dtype=int), np.array(fd["val"], dtype=int)
        assert_no_group_overlap(groups, tr, va, f"fold {fd['fold']} train vs val")
        assert_no_group_overlap(groups, np.concatenate([tr, va]), test, f"fold {fd['fold']} vs test")
    return splits
