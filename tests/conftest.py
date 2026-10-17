"""Shared pytest fixtures. All test data is synthetic and generated on the fly."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@pytest.fixture(scope="session")
def base_cfg():
    """The real project config (validated)."""
    return load_config(CONFIG_PATH)


@pytest.fixture()
def cfg(base_cfg):
    """Fast test config: random init (no weight download), small Swin."""
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    c["model"]["swin"]["embed_dim"] = 64
    c["model"]["swin"]["num_heads"] = 4
    c["training"]["num_workers"] = 0
    c["training"]["amp"] = False
    c["training"]["device"] = "cpu"
    return c


def pytest_addoption(parser):
    parser.addoption("--run-training", action="store_true", default=False,
                     help="also run the tiny synthetic training/CV smoke tests (tests marked 'training')")


def pytest_configure(config):
    config.addinivalue_line("markers", "training: runs actual (tiny, synthetic) training; opt-in via --run-training")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-training"):
        return
    skip = pytest.mark.skip(reason="training smoke test; run with `pytest --run-training` on the training machine")
    for item in items:
        if "training" in item.keywords:
            item.add_marker(skip)
