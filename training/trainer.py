"""Train/validation loops with AdamW, cosine annealing, early stopping on val macro-F1,
optional train-only class weights, optional AMP (CUDA only), gradient clipping,
per-epoch CSV/JSON logs, and resumable checkpoints."""
from __future__ import annotations

import contextlib
import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import macro_f1
from utils.common import save_json

from .checkpointing import load_checkpoint, make_checkpoint, restore_rng_state, rng_state, save_checkpoint

logger = logging.getLogger(__name__)

MONITOR = "val_macro_f1"


def autocast_ctx(device: torch.device, enabled: bool):
    """float16 autocast on CUDA only; a no-op context otherwise (torch 2.1 rejects fp16 CPU autocast)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights ``N / (K * n_c)`` from TRAINING labels only.

    Classes absent from the training fold get weight 0 (with a warning)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    w = np.zeros(num_classes)
    nz = counts > 0
    w[nz] = counts.sum() / (nz.sum() * counts[nz])
    if (~nz).any():
        logger.warning("Classes %s absent from training fold; class weight set to 0", np.where(~nz)[0].tolist())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False
                   ) -> Dict[str, Any]:
    """Run inference over a loader. Returns logits, probs, y_true, y_pred and meta (if provided)."""
    model.eval()
    logits_all, y_all, metas = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with autocast_ctx(device, use_amp):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        y_all.append(y)
        if len(batch) > 2:
            meta = batch[2]
            metas.extend({k: (v[i].item() if torch.is_tensor(v) else v[i]) for k, v in meta.items()}
                         for i in range(len(y)))
    logits = torch.cat(logits_all) if logits_all else torch.empty(0)
    probs = torch.softmax(logits, dim=1) if len(logits) else logits
    return {"logits": logits.numpy(), "probs": probs.numpy(), "y_true": torch.cat(y_all).numpy() if y_all else
            np.empty(0, int), "y_pred": probs.argmax(1).numpy() if len(probs) else np.empty(0, int), "meta": metas}


class Trainer:
    """Fits one model on one fold.

    Writes into ``out_dir``: ``history.csv`` / ``history.json``; into ``ckpt_dir``:
    ``best.pth`` (best validation macro-F1, model weights + metadata) and ``last.pth``
    (full resumable state).
    """

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], device: torch.device, out_dir: Path,
                 ckpt_dir: Path, fold: Optional[int] = None, class_weights: Optional[torch.Tensor] = None,
                 loader_generator: Optional[torch.Generator] = None) -> None:
        t = cfg["training"]
        self.cfg, self.t = cfg, t
        self.model = model.to(device)
        self.device = device
        self.fold = fold
        self.out_dir, self.ckpt_dir = Path(out_dir), Path(ckpt_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = len(cfg["classes"])
        self.use_amp = bool(t["amp"]) and device.type == "cuda"
        if t["amp"] and device.type != "cuda":
            logger.info("AMP requested but device is %s; running in float32.", device.type)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        self.class_weights = class_weights
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t["epochs"], eta_min=t["min_lr"])
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.loader_generator = loader_generator
        self.best_metric = -float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.history: List[Dict[str, Any]] = []
        self.start_epoch = 1
        self.finished = False

    # ------------------------------------------------------------------ paths
    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pth"

    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.pth"

    # ------------------------------------------------------------------ state
    def _trainer_state(self) -> Dict[str, Any]:
        return {"optimizer_state": self.optimizer.state_dict(), "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(), "rng_state": rng_state(self.loader_generator),
                "trainer": {"best_metric": self.best_metric, "best_epoch": self.best_epoch,
                            "bad_epochs": self.bad_epochs, "history": self.history, "finished": self.finished}}

    def resume(self) -> bool:
        """Restore from ``last.pth`` if present. Returns True if state was restored."""
        if not self.last_path.is_file():
            logger.info("No checkpoint at %s; starting from scratch.", self.last_path)
            return False
        ck = load_checkpoint(self.last_path, map_location="cpu")
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scheduler.load_state_dict(ck["scheduler_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        tr = ck["trainer"]
        self.best_metric, self.best_epoch = tr["best_metric"], tr["best_epoch"]
        self.bad_epochs, self.history, self.finished = tr["bad_epochs"], tr["history"], tr["finished"]
        self.start_epoch = ck["epoch"] + 1
        restore_rng_state(ck["rng_state"], self.loader_generator)
        logger.info("Resumed fold %s from epoch %d (best %s=%.4f @ epoch %d)", self.fold, ck["epoch"], MONITOR,
                    self.best_metric, self.best_epoch)
        return True

    # ------------------------------------------------------------------ loops
    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, *_ in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(self.device, self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss {loss.item()} (fold {self.fold})")
            self.scaler.scale(loss).backward()
            if self.t["grad_clip"]:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.t["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            n = y.shape[0]
            total += n
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
        return {"train_loss": loss_sum / max(total, 1), "train_acc": correct / max(total, 1)}

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        out = predict_loader(self.model, loader, self.device, self.use_amp)
        y, logits = torch.from_numpy(out["y_true"]), torch.from_numpy(out["logits"])
        loss = nn.functional.cross_entropy(logits, y, weight=self.class_weights).item() if len(y) else float("nan")
        yt, yp = out["y_true"], out["y_pred"]
        recalls = [np.mean(yp[yt == k] == k) for k in np.unique(yt)]
        return {"val_loss": loss, "val_acc": float(np.mean(yt == yp)),
                "val_balanced_acc": float(np.mean(recalls)) if recalls else float("nan"),
                MONITOR: macro_f1(yt, yp, self.num_classes)}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume: bool = False,
            stop_after_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Train with early stopping. ``stop_after_epochs`` simulates an interruption (tests)."""
        if resume:
            self.resume()
        es = self.t["early_stopping"]
        ran = 0
        for epoch in range(self.start_epoch, self.t["epochs"] + 1):
            if self.finished:
                break
            t0 = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            tr = self._train_epoch(train_loader)
            va = self._validate(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            score = va[MONITOR]
            improved = score > self.best_metric + es["min_delta"]
            if improved:
                self.best_metric, self.best_epoch, self.bad_epochs = score, epoch, 0
                save_checkpoint(make_checkpoint(self.model, self.cfg, epoch, MONITOR, score, self.fold), self.best_path)
            else:
                self.bad_epochs += 1
            row = {"epoch": epoch, "lr": lr, **tr, **va, "improved": improved, "seconds": round(time.time() - t0, 2),
                   "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
                   if self.device.type == "cuda" else None}
            self.history.append(row)
            logger.info("fold %s epoch %d/%d | train_loss %.4f acc %.3f | val_loss %.4f acc %.3f macroF1 %.4f%s",
                        self.fold, epoch, self.t["epochs"], tr["train_loss"], tr["train_acc"], va["val_loss"],
                        va["val_acc"], score, " *" if improved else "")
            if self.bad_epochs >= es["patience"]:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, es["patience"])
                self.finished = True
            if epoch == self.t["epochs"]:
                self.finished = True
            save_checkpoint(make_checkpoint(self.model, self.cfg, epoch, MONITOR, score, self.fold,
                                            extra=self._trainer_state()), self.last_path)
            self._write_history()
            ran += 1
            if stop_after_epochs is not None and ran >= stop_after_epochs:
                break
        self._write_history()
        return {"fold": self.fold, "best_epoch": self.best_epoch, "best_val_macro_f1": self.best_metric,
                "epochs_run": len(self.history), "finished": self.finished,
                "early_stopped": self.finished and len(self.history) < self.t["epochs"]}

    def _write_history(self) -> None:
        save_json(self.history, self.out_dir / "history.json")
        if self.history:
            with (self.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.history[0].keys()))
                w.writeheader()
                w.writerows(self.history)
