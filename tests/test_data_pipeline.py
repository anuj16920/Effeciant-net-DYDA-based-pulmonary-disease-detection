"""Data pipeline tests on SYNTHETIC images only (generated into a temp dir)."""
from __future__ import annotations

import copy

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from datasets import prepare_manifest as pm
from datasets.augmentation import TrainAugmentation
from datasets.dataset import XRayDataset, load_manifest, subset
from datasets.preprocessing import (ImageReadError, apply_clahe, load_image, prepare_uint8, preprocess_image,
                                    to_tensor)
from datasets.splits import SplitLeakageError, assert_no_group_overlap, load_splits, make_splits
from utils.common import load_json
from utils.config import get_path, resolve_path
from utils.synthetic import synthetic_config, synthetic_xray


@pytest.fixture(scope="module")
def synth(tmp_path_factory, base_cfg):
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    c["manifest"]["hash_workers"] = 2
    cfg = synthetic_config(c, tmp_path_factory.mktemp("synth"), n_per_class=6, seed=0)
    assert pm.main_with_cfg(cfg) == 0
    return cfg


@pytest.fixture(scope="module")
def manifest(synth):
    return load_manifest(synth)


@pytest.fixture(scope="module")
def report(synth):
    return load_json(get_path(synth, "data_report"))


# ----------------------------------------------------------------------------- manifest
def test_manifest_columns_and_counts(manifest, report):
    for col in pm.MANIFEST_COLUMNS:
        assert col in manifest.columns
    assert len(manifest) == 45 == report["n_images"]
    assert report["class_counts_overall"] == {"COVID-19": 7, "Pneumonia": 13, "Tuberculosis": 6, "Normal": 19}
    assert report["class_counts_per_source"]["montgomery_tb"] == {"COVID-19": 0, "Pneumonia": 0,
                                                                  "Tuberculosis": 3, "Normal": 3}
    assert manifest["sha256"].str.len().eq(64).all()
    assert manifest["phash"].str.fullmatch(r"[0-9a-f]{16}").all()


def test_exclusions_are_reported_not_mapped(manifest, report):
    assert report["excluded_by_label_map"]["covid_radiography"] == {"Lung_Opacity": 1}
    assert report["unmapped_labels"]["covid_radiography"] == {"Other/images": 1}
    assert any("Other" in p for p in report["unmapped_examples"]["covid_radiography"])
    assert len(report["unreadable_images"]) == 1 and "corrupt" in report["unreadable_images"][0]["path"]
    paths = manifest["image_path"].str.lower()
    assert not paths.str.contains("masks|manualmask|lung_opacity|other|corrupt").any()


def test_duplicates_and_patients_share_groups(manifest, report):
    def gid(name):
        return manifest.loc[manifest["image_path"].str.endswith(name), "group_id"].item()

    assert gid("COVID-1.png") == gid("COVID-dup.png")              # exact duplicate
    assert gid("Normal-1.png") == gid("Normal-near.png")           # near duplicate
    assert gid("person1_bacteria_1.jpeg") == gid("person1_bacteria_99.jpeg")  # same patient
    assert report["duplicates"]["exact_duplicate_files"] == 1
    assert report["duplicates"]["near_duplicate_pairs"] == 1
    assert report["n_groups"] == 42
    pids = manifest.set_index(manifest["image_path"].str.split("/").str[-1])["patient_id"]
    assert pids["MCUCXR_0002_1.png"] == "montgomery_tb:MCUCXR_0002"
    assert pids["person3_bacteria_3.jpeg"] == "kaggle_pneumonia:person3"
    assert pids["COVID-2.png"] == ""


def test_subtypes_and_warnings(manifest, report):
    kag = manifest[(manifest["source_dataset"] == "kaggle_pneumonia") & (manifest["label"] == "Pneumonia")]
    assert set(kag["subtype"]) == {"bacteria", "virus"}
    w = " ".join(report["warnings"])
    assert "SOURCE-SHORTCUT RISK: class 'COVID-19'" in w
    assert "only contains classes" in w
    assert report["patient_id_missing"]["covid_radiography"] > 0


def test_label_conflict_groups_are_dropped(tmp_path, base_cfg):
    c = copy.deepcopy(base_cfg)
    c["manifest"]["hash_workers"] = 1
    cfg = synthetic_config(c, tmp_path, n_per_class=3, seed=1)
    cov = resolve_path(cfg, cfg["sources"]["covid_radiography"]["root"])
    # same bytes under two different labels
    (cov / "Normal" / "images" / "conflict.png").write_bytes((cov / "COVID" / "images" / "COVID-2.png").read_bytes())
    df, rep = pm.build_manifest(cfg)
    assert len(rep["label_conflicts"]) == 1 and rep["dropped_label_conflict_images"] == 2
    assert not df["image_path"].str.endswith(("conflict.png", "COVID-2.png")).any()


def test_missing_source_is_reported(tmp_path, base_cfg):
    c = copy.deepcopy(base_cfg)
    cfg = synthetic_config(c, tmp_path, n_per_class=3, seed=2)
    cfg["sources"]["shenzhen_tb"]["root"] = str(tmp_path / "does_not_exist")
    _, rep = pm.build_manifest(cfg)
    assert "shenzhen_tb" in rep["missing_sources"]


# ----------------------------------------------------------------------------- hashing
def test_phash_and_hamming():
    rng = np.random.default_rng(0)
    a = synthetic_xray("Normal", rng)
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    c = synthetic_xray("Normal", rng)
    ha, hb, hc = (int(pm.phash(x), 16) for x in (a, b, c))
    assert bin(ha ^ hb).count("1") <= 4 < bin(ha ^ hc).count("1")
    h = np.array([ha, hb, hc], dtype=np.uint64)
    d = pm.hamming_matrix(h, h)
    assert d[0, 1] == bin(ha ^ hb).count("1") and d[0, 2] == bin(ha ^ hc).count("1") and (np.diag(d) == 0).all()
    assert pm.near_duplicate_pairs([f"{x:016x}" for x in (ha, hb, hc)], 4) == [(0, 1)]


# ----------------------------------------------------------------------------- splits
def test_splits_have_zero_group_overlap(synth, manifest):
    splits = load_splits(get_path(synth, "splits"), manifest)
    groups = manifest["group_id"]
    test = np.array(splits["test"])
    assert len(splits["folds"]) == 5
    all_val = []
    for fd in splits["folds"]:
        tr, va = np.array(fd["train"]), np.array(fd["val"])
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])
        assert not set(groups.iloc[np.concatenate([tr, va])]) & set(groups.iloc[test])
        all_val.extend(va.tolist())
    assert sorted(all_val) == splits["dev"]
    assert sorted(splits["dev"] + splits["test"]) == list(range(len(manifest)))
    assert 0.05 < splits["test_fraction_actual"] < 0.35
    assert set(splits["class_distribution"]["test"]) == set(synth["classes"])


def test_splits_are_deterministic_and_detect_stale_manifest(synth, manifest):
    a, b = make_splits(manifest, synth), make_splits(manifest, synth)
    assert a["test"] == b["test"] and a["folds"] == b["folds"]
    changed = manifest.copy()
    changed.loc[0, "label"] = "Normal" if changed.loc[0, "label"] != "Normal" else "COVID-19"
    with pytest.raises(ValueError, match="different manifest"):
        load_splits(get_path(synth, "splits"), changed)


def test_overlap_assertion_fires():
    groups = pd.Series(["a", "b", "c", "a"])
    with pytest.raises(SplitLeakageError):
        assert_no_group_overlap(groups, np.array([0, 1]), np.array([2, 3]), "x")


def test_too_few_groups_raises(synth, manifest):
    with pytest.raises(ValueError, match="groups"):
        make_splits(manifest.head(5), synth)


# ----------------------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("kind", ["gray", "rgb", "rgba", "u16", "jpg"])
def test_preprocessing_handles_formats(tmp_path, base_cfg, kind):
    rng = np.random.default_rng(0)
    g = synthetic_xray("Normal", rng, size=300)
    if kind == "gray":
        img, ext = g, ".png"
    elif kind == "rgb":
        img, ext = np.dstack([g, g, g]), ".png"
    elif kind == "rgba":
        img, ext = np.dstack([g, g, g, np.full_like(g, 255)]), ".png"
    elif kind == "u16":
        img, ext = g.astype(np.uint16) * 200, ".png"
    else:
        img, ext = g, ".jpg"
    p = tmp_path / f"dir with space/img_{kind}{ext}"
    p.parent.mkdir()
    cv2.imencode(ext, img)[1].tofile(str(p))
    t = preprocess_image(p, base_cfg["preprocessing"])
    assert t.shape == (3, 224, 224) and t.dtype == torch.float32 and torch.isfinite(t).all()
    u8 = prepare_uint8(p, base_cfg["preprocessing"]).astype(int)
    assert u8.shape == (224, 224, 3)
    assert np.abs(u8[..., 0] - u8[..., 1]).max() <= 1 and np.abs(u8[..., 1] - u8[..., 2]).max() <= 1


def test_unreadable_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"xx")
    with pytest.raises(ImageReadError):
        load_image(bad)
    with pytest.raises(ImageReadError):
        load_image(tmp_path / "missing.png")


def test_clahe_changes_contrast_and_is_deterministic(base_cfg):
    rng = np.random.default_rng(0)
    g = (synthetic_xray("Normal", rng) // 4 + 100).astype(np.uint8)  # low contrast
    out = apply_clahe(g, 2.0, (8, 8))
    assert out.std() > g.std()
    p = base_cfg["preprocessing"]
    assert np.array_equal(prepare_uint8(g, p), prepare_uint8(g, p))


# ----------------------------------------------------------------------------- dataset + augmentation
def test_dataset_val_matches_inference_preprocessing(synth, manifest):
    ds = XRayDataset(manifest, synth, train=False, return_meta=True)
    x, y, meta = ds[3]
    ref = preprocess_image(resolve_path(synth, manifest.loc[3, "image_path"]), synth["preprocessing"])
    assert torch.equal(x, ref)
    assert int(y) == synth["classes"].index(manifest.loc[3, "label"])
    assert meta["index"] == 3 and meta["source_dataset"] == manifest.loc[3, "source_dataset"]
    x2, _, _ = ds[3]  # second call goes through the cache
    assert torch.equal(x, x2)


def test_augmentation_only_in_training(synth, manifest):
    df = subset(manifest, [0, 1, 2])
    val = XRayDataset(df, synth, train=False)
    assert val.augment is None and torch.equal(val[0][0], val[0][0])
    c = copy.deepcopy(synth)
    c["augmentation"]["p"] = 1.0
    tr = XRayDataset(df, c, train=True)
    np.random.seed(0)
    a, b = tr[0][0], tr[0][0]
    assert a.shape == (3, 224, 224) and not torch.equal(a, b)


def test_augmentation_mild_and_disable(base_cfg):
    img = prepare_uint8(synthetic_xray("Normal", np.random.default_rng(0)), base_cfg["preprocessing"])
    acfg = dict(base_cfg["augmentation"], p=1.0)
    out = TrainAugmentation(acfg)(img, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 60
    off = TrainAugmentation(dict(acfg, enabled=False))
    assert off(img) is img
    assert to_tensor(out, base_cfg["preprocessing"]).shape == (3, 224, 224)
