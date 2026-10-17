"""Training / CV / checkpoint smoke tests on tiny SYNTHETIC data (CPU, random init).

These verify the machinery only; the numbers produced are meaningless."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from datasets import prepare_manifest as pm
from datasets.dataset import XRayDataset, load_manifest, subset
from datasets.splits import load_splits
from models.hybrid_model import build_model
from training.checkpointing import load_checkpoint, model_from_checkpoint
from training.cross_validation import make_loader, run_cv
from training.trainer import Trainer, class_weights_from_labels
from utils.common import load_json, seed_everything
from utils.config import get_path, with_model
from utils.synthetic import synthetic_config

pytestmark = pytest.mark.training  # opt-in: `pytest --run-training`


def tiny(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False
    c["model"]["swin"].update(embed_dim=32, num_heads=4)
    c["preprocessing"]["img_size"] = 64
    c["training"].update(batch_size=8, epochs=1, num_workers=0, amp=False, device="cpu")
    c["evaluation"]["batch_size"] = 16
    c["manifest"]["hash_workers"] = 2
    return c


@pytest.fixture(scope="module")
def tcfg(tmp_path_factory, base_cfg):
    cfg = synthetic_config(tiny(base_cfg), tmp_path_factory.mktemp("train"), n_per_class=6, seed=3, size=96)
    assert pm.run(cfg) == 0
    return cfg


@pytest.mark.parametrize("variant", ["backbone", "cbam", "dyda", "hybrid"])
def test_cv_smoke_all_models(tcfg, variant):
    cfg = with_model(tcfg, variant)
    summary = run_cv(cfg, folds=[0, 1])
    assert [r["fold"] for r in summary["per_fold"]] == [0, 1]
    for r in summary["per_fold"]:
        assert 0.0 <= r["macro_f1"] <= 1.0 and 0.0 <= r["accuracy"] <= 1.0
    out = get_path(cfg, "outputs_dir") / variant
    for f in (out / "cv_summary.json", out / "cv_summary.csv", out / "fold_0" / "history.csv",
              out / "fold_0" / "val_metrics.json", out / "fold_1" / "val_predictions.csv"):
        assert f.is_file(), f
    best = get_path(cfg, "checkpoints_dir") / variant / "best_model.pth"
    ck = load_checkpoint(best)
    assert ck["model_config"]["variant"] == variant and ck["metric"]["split"] == "validation"
    assert ck["class_map"] == {0: "COVID-19", 1: "Pneumonia", 2: "Tuberculosis", 3: "Normal"}
    assert {"torch", "timm"} <= set(ck["versions"]) and ck["seed"] == cfg["training"]["seed"]
    assert ck["preprocessing"]["img_size"] == 64


def test_checkpoint_roundtrip_identical_outputs(tcfg):
    ckpt = get_path(tcfg, "checkpoints_dir") / "hybrid" / "best_model.pth"
    if not ckpt.is_file():
        run_cv(with_model(tcfg, "hybrid"), folds=[0])
    m1, cfg1, ck = model_from_checkpoint(ckpt)
    m2 = build_model(cfg1, "hybrid").eval()
    m2.load_state_dict(ck["model_state"])
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))


def _trainer(cfg, name, fold=0):
    manifest = load_manifest(cfg)
    splits = load_splits(get_path(cfg, "splits"), manifest)
    fd = splits["folds"][fold]
    seed_everything(cfg["training"]["seed"])
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    tr = make_loader(XRayDataset(subset(manifest, fd["train"]), cfg, train=True), cfg, True, torch.device("cpu"), gen)
    va = make_loader(XRayDataset(subset(manifest, fd["val"]), cfg, train=False), cfg, False, torch.device("cpu"))
    model = build_model(cfg, "dyda")
    root = get_path(cfg, "outputs_dir")
    t = Trainer(model, cfg, torch.device("cpu"), root / name, get_path(cfg, "checkpoints_dir") / name, fold,
                loader_generator=gen)
    return t, tr, va


def test_resume_matches_uninterrupted_run(tcfg):
    cfg = copy.deepcopy(tcfg)
    cfg["training"]["epochs"] = 2
    cfg["training"]["early_stopping"]["patience"] = 5
    full, tr, va = _trainer(cfg, "resume_full")
    res_full = full.fit(tr, va)
    assert res_full["epochs_run"] == 2 and res_full["finished"]

    part, tr, va = _trainer(cfg, "resume_part")
    r1 = part.fit(tr, va, stop_after_epochs=1)
    assert r1["epochs_run"] == 1 and not r1["finished"]
    # fresh objects, as after a crash; resume from last.pth
    resumed, tr, va = _trainer(cfg, "resume_part")
    r2 = resumed.fit(tr, va, resume=True)
    assert r2["epochs_run"] == 2 and r2["finished"]
    assert [h["epoch"] for h in resumed.history] == [1, 2]
    for (k, a), b in zip(full.model.state_dict().items(), resumed.model.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-5), f"param {k} differs after resume"
    assert resumed.scheduler.last_epoch == full.scheduler.last_epoch == 2
    # resuming a finished run is a no-op
    again, tr, va = _trainer(cfg, "resume_part")
    assert again.fit(tr, va, resume=True)["epochs_run"] == 2


def test_early_stopping_and_history(tcfg):
    cfg = copy.deepcopy(tcfg)
    cfg["training"]["epochs"] = 6
    cfg["training"]["early_stopping"]["patience"] = 1
    cfg["training"]["early_stopping"]["min_delta"] = 1.0  # macro-F1 can never improve by > 1
    cfg["augmentation"]["enabled"] = False
    t, tr, va = _trainer(cfg, "early")
    res = t.fit(tr, va)
    assert res["early_stopped"] and res["epochs_run"] < 6
    hist = load_json(t.out_dir / "history.json")
    assert {"train_loss", "val_loss", "val_macro_f1", "lr"} <= set(hist[0])


def test_class_weights_use_training_labels_only():
    w = class_weights_from_labels(np.array([0, 0, 0, 1, 2, 2]), 4)
    assert w[3] == 0 and w[0] < w[2] < w[1]
    assert torch.isclose((w[:3] * torch.tensor([3.0, 1.0, 2.0])).sum(), torch.tensor(6.0))
