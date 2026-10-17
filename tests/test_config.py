"""Config loader validation tests."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from utils.config import (ConfigError, get_path, load_config, public_config, validate_config,
                          with_model)

from .conftest import CONFIG_PATH, PROJECT_ROOT


def _raw():
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_project_config_loads_and_resolves_paths(base_cfg):
    assert base_cfg["classes"] == ["COVID-19", "Pneumonia", "Tuberculosis", "Normal"]
    assert get_path(base_cfg, "manifest") == (PROJECT_ROOT / "data" / "manifests" / "manifest.csv").resolve()
    assert "_project_root" not in public_config(base_cfg)


def test_lung_opacity_is_explicitly_excluded(base_cfg):
    assert base_cfg["sources"]["covid_radiography"]["label_map"]["Lung_Opacity"] is None


@pytest.mark.parametrize("dotted,value,fragment", [
    ("experiment.model", "resnet", "experiment.model"),
    ("model.backbone", "resnet50", "efficientnet_b3"),
    ("model.swin.shift_size", 0, "shift_size"),
    ("model.swin.upsample_size", 7, "no-op"),
    ("model.swin.num_heads", 7, "divisible"),
    ("model.swin.depth", 1, "SW-MSA"),
    ("model.dyda.spatial_kernel", 4, "odd"),
    ("training.test_fraction", 0.7, "test_fraction"),
    ("training.lr", -1.0, "training.lr"),
    ("training.folds", True, "bool"),
    ("augmentation.p", 1.5, "augmentation.p"),
])
def test_invalid_values_raise_clear_errors(dotted, value, fragment):
    cfg = _raw()
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        validate_config(cfg)


def test_unknown_and_missing_keys():
    cfg = _raw()
    cfg["training"]["lr_typo"] = 1.0
    with pytest.raises(ConfigError, match="Unknown key"):
        validate_config(cfg)
    cfg = _raw()
    del cfg["preprocessing"]["clahe"]
    with pytest.raises(ConfigError, match="preprocessing.clahe"):
        validate_config(cfg)


def test_label_map_target_must_be_a_class():
    cfg = _raw()
    cfg["sources"]["covid_radiography"]["label_map"]["Lung_Opacity"] = "Pneumonia-ish"
    with pytest.raises(ConfigError, match="not in classes"):
        validate_config(cfg)


def test_missing_file_and_empty_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)


def test_overrides_and_with_model(base_cfg):
    cfg = load_config(CONFIG_PATH, overrides={"experiment.model": "dyda"})
    assert cfg["experiment"]["model"] == "dyda"
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(CONFIG_PATH, overrides={"training.nope": 1})
    e2 = with_model(base_cfg, "cbam")
    assert e2["experiment"] == {"name": "cbam", "model": "cbam"}
    assert base_cfg["experiment"]["model"] == "hybrid"  # original untouched
    with pytest.raises(ConfigError):
        with_model(base_cfg, "vit")


def test_validate_is_idempotent(base_cfg):
    validate_config(copy.deepcopy(base_cfg))
