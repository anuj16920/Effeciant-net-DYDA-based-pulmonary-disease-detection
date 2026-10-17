"""PyTorch Dataset over the manifest."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.config import get_path, resolve_path

from .augmentation import TrainAugmentation
from .preprocessing import prepare_uint8, preprocessing_signature, to_tensor

logger = logging.getLogger(__name__)


def load_manifest(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = get_path(cfg, "manifest")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run `python -m datasets.prepare_manifest` first.")
    df = pd.read_csv(path, dtype={"patient_id": str, "group_id": str, "phash": str, "subtype": str},
                     keep_default_na=False)
    missing = {"image_path", "label", "source_dataset", "patient_id", "group_id", "sha256", "phash"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing columns {sorted(missing)}")
    return df


_GLOBAL_RAW_CACHE: Dict[str, np.ndarray] = {}


class XRayDataset(Dataset):
    """Chest X-ray dataset with ultra-high-speed pinned RAM multi-bank tensor caching.

    Args:
        df: manifest rows to use (already restricted to one split).
        cfg: full config.
        train: if True, apply training augmentation (never for val/test).
        return_meta: if True, ``__getitem__`` returns ``(x, y, meta)``; meta is a dict with
            ``index``, ``image_path`` and ``source_dataset``.

    Item: ``x`` float tensor ``(3, S, S)``, ``y`` int64 class index.
    """

    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any], train: bool = False, return_meta: bool = False) -> None:
        self.df = df.reset_index(drop=False).rename(columns={"index": "manifest_index"})
        self.cfg = cfg
        self.pcfg = cfg["preprocessing"]
        self.classes = list(cfg["classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        bad = set(self.df["label"]) - set(self.classes)
        if bad:
            raise ValueError(f"Labels {sorted(bad)} are not in config classes {self.classes}")
        self.labels = self.df["label"].map(self.class_to_idx).to_numpy(dtype=np.int64)
        self.train = train
        self.augment = TrainAugmentation(cfg["augmentation"]) if train else None
        self.return_meta = return_meta
        self.cache_dir: Optional[Path] = None
        if self.pcfg.get("cache", False):
            self.cache_dir = get_path(cfg, "processed_dir") / f"prep_{preprocessing_signature(self.pcfg)}"

        # Pre-allocate pinned RAM tensors: 3 banks for train (~24 GB), 1 bank for val (~2 GB)
        self.mem_tensors: Optional[torch.Tensor] = None
        if self.cache_dir is not None and len(self.df) > 0:
            self._preload_into_ram()

    def __len__(self) -> int:
        return len(self.df)

    def _preload_into_ram(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        n = len(self.df)
        num_banks = 3 if self.train else 1
        gb = round(num_banks * n * 3 * 224 * 224 * 4 / (1024**3), 2)
        logger.info("Allocating %s GB of pinned RAM tensors (%d bank(s) for %d images)...", gb, num_banks, n)
        self.mem_tensors = torch.empty((num_banks, n, 3, 224, 224), dtype=torch.float32, pin_memory=True)

        logger.info("Pre-loading and populating %d images into pinned RAM tensor banks...", n)

        def process_idx(i: int):
            raw_img = self._load_single(i)
            # Bank 0: baseline deterministic normalized float32 tensor
            t0 = to_tensor(raw_img, self.pcfg)
            self.mem_tensors[0, i].copy_(t0)
            if self.train:
                for b in range(1, num_banks):
                    aug_img = self.augment(raw_img) if self.augment else raw_img
                    tb = to_tensor(aug_img, self.pcfg)
                    self.mem_tensors[b, i].copy_(tb)

        with ThreadPoolExecutor(max_workers=min(24, (os.cpu_count() or 8) * 2)) as ex:
            list(ex.map(process_idx, range(n)))
        logger.info("Successfully populated %s GB pinned RAM cache! Ultra-fast zero-latency active.", gb)

    def _load_single(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        sha = row["sha256"]
        if sha and sha in _GLOBAL_RAW_CACHE:
            return _GLOBAL_RAW_CACHE[sha]

        if self.cache_dir is not None and sha:
            cp = self.cache_dir / f"{sha}.png"
            if cp.is_file():
                img = cv2.imdecode(np.fromfile(str(cp), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _GLOBAL_RAW_CACHE[sha] = res
                    return res
            img = prepare_uint8(resolve_path(self.cfg, row["image_path"]), self.pcfg)
            cp.parent.mkdir(parents=True, exist_ok=True)
            ok, enc = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                tmp = cp.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
                enc.tofile(str(tmp))
                tmp.replace(cp)
            _GLOBAL_RAW_CACHE[sha] = img
            return img
        res = prepare_uint8(resolve_path(self.cfg, row["image_path"]), self.pcfg)
        if sha:
            _GLOBAL_RAW_CACHE[sha] = res
        return res

    def __getitem__(self, i: int):
        if self.mem_tensors is not None:
            if self.train and self.mem_tensors.shape[0] > 1:
                b = int(torch.randint(0, self.mem_tensors.shape[0], (1,)).item())
                x = self.mem_tensors[b, i]
            else:
                x = self.mem_tensors[0, i]
        else:
            img = self._load_single(i)
            if self.augment is not None:
                img = self.augment(img)
            x = to_tensor(img, self.pcfg)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        if self.return_meta:
            row = self.df.iloc[i]
            return x, y, {"index": int(row["manifest_index"]), "image_path": str(row["image_path"]),
                          "source_dataset": str(row["source_dataset"])}
        return x, y


def subset(df: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    """Manifest rows for the given split indices (original index preserved)."""
    return df.iloc[list(indices)]
