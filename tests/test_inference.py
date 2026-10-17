"""Inference + Grad-CAM tests on SYNTHETIC images with randomly initialised checkpoints."""
from __future__ import annotations

import copy
import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from explainability.gradcam import GradCAM, explain_image, export_examples, resolve_target_layer
from inference import batch_predict, predict
from models.hybrid_model import build_model
from training.checkpointing import make_checkpoint, model_from_checkpoint, save_checkpoint
from utils.synthetic import synthetic_xray

VARIANTS = ["backbone", "cbam", "dyda", "hybrid"]


def small(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False
    c["model"]["swin"].update(embed_dim=32, num_heads=4)
    c["preprocessing"]["img_size"] = 96
    c["training"]["device"] = "cpu"
    return c


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    for i, lab in enumerate(["COVID-19", "Pneumonia", "Tuberculosis"]):
        cv2.imencode(".png", synthetic_xray(lab, rng, 200))[1].tofile(str(d / f"img{i}.png"))
    (d / "broken.png").write_bytes(b"nope")
    return d


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, base_cfg):
    d = tmp_path_factory.mktemp("ck")
    cfg = small(base_cfg)
    out = {}
    for v in VARIANTS:
        torch.manual_seed(0)
        c = copy.deepcopy(cfg)
        c["experiment"]["model"] = v
        m = build_model(c, v)
        out[v] = save_checkpoint(make_checkpoint(m, c, epoch=1, metric_name="val_macro_f1", metric_value=0.0,
                                                 fold=0), d / f"{v}.pth")
    return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_predictor_probabilities(ckpts, images, variant):
    pred = predict.Predictor(ckpts[variant], "cpu")
    r = pred.predict(images / "img0.png")
    assert set(r["probabilities"]) == {"COVID-19", "Pneumonia", "Tuberculosis", "Normal"}
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert r["confidence"] == max(r["probabilities"].values())
    assert "NOT a validated diagnostic" in r["disclaimer"]
    assert pred.pcfg["img_size"] == 96  # preprocessing comes from the checkpoint


def test_checkpoint_roundtrip_same_prediction(ckpts, images):
    m, cfg, ck = model_from_checkpoint(ckpts["hybrid"])
    from datasets.preprocessing import preprocess_image

    x = preprocess_image(images / "img1.png", ck["preprocessing"])[None]
    with torch.no_grad():
        p1 = torch.softmax(m(x), 1)
    p2 = predict.Predictor(ckpts["hybrid"], "cpu").predict_tensor(x)
    assert torch.allclose(p1, p2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gradcam_shape_range_and_hook_removal(ckpts, images, variant):
    m, cfg, ck = model_from_checkpoint(ckpts[variant])
    layer = resolve_target_layer(m, "auto")
    from datasets.preprocessing import preprocess_image

    x = preprocess_image(images / "img2.png", ck["preprocessing"])[None]
    with GradCAM(m, layer) as gc:
        cam, probs, cls = gc(x)
        cam3, _, cls3 = gc(x, target_class=3)
    assert cam.shape == (96, 96) and cam.dtype == np.float32
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert probs.sum() == pytest.approx(1.0, abs=1e-5) and cls == int(probs.argmax()) and cls3 == 3
    assert len(layer._forward_hooks) == 0, "Grad-CAM hook not removed"
    assert all(p.grad is None for p in m.parameters())


def test_gradcam_named_layer_and_bad_layer(ckpts):
    m, _, _ = model_from_checkpoint(ckpts["dyda"])
    assert resolve_target_layer(m, "backbone.blocks.5") is m.backbone.blocks[5]
    with pytest.raises(ValueError):
        resolve_target_layer(m, "no.such.layer")


def test_explain_and_export(ckpts, images, tmp_path, base_cfg):
    m, cfg, ck = model_from_checkpoint(ckpts["hybrid"])
    res = explain_image(m, images / "img0.png", ck["preprocessing"], cfg["classes"], tmp_path / "one",
                        torch.device("cpu"), true_label="COVID-19")
    for f in ("original.png", "heatmap.png", "overlay.png", "gradcam.json"):
        assert (tmp_path / "one" / f).is_file()
    meta = json.loads((tmp_path / "one" / "gradcam.json").read_text(encoding="utf-8"))
    assert "NOT clinical proof" in meta["caution"] and meta["true_label"] == "COVID-19"
    assert res["cam"].shape == (96, 96)
    cfg2 = copy.deepcopy(cfg)
    cfg2["_project_root"] = str(tmp_path)
    rows = [{"image_path": str(images / f"img{i}.png"), "label": lab}
            for i, lab in enumerate(["COVID-19", "Pneumonia", "Tuberculosis"])]
    counts = export_examples(m, cfg2, rows, tmp_path / "export", torch.device("cpu"), max_examples=2)
    assert counts["correct"] + counts["incorrect"] == 3
    assert (tmp_path / "export" / "index.json").is_file()
    assert not (tmp_path / "export" / "_tmp").exists()
    for bucket in ("correct", "incorrect"):
        for d in (tmp_path / "export" / bucket).glob("*") if (tmp_path / "export" / bucket).exists() else []:
            assert (d / "overlay.png").is_file()


def test_predict_cli_single_and_dir(ckpts, images, tmp_path, base_cfg, capsys):
    from tests.conftest import CONFIG_PATH

    out_json = tmp_path / "single.json"
    assert predict.main(["--config", str(CONFIG_PATH), "--checkpoint", str(ckpts["dyda"]), "--image",
                         str(images / "img0.png"), "--output", str(out_json), "--device", "cpu"]) == 0
    res = json.loads(out_json.read_text(encoding="utf-8"))
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-5) and "disclaimer" in res
    out_csv = tmp_path / "batch.csv"
    assert batch_predict.main(["--config", str(CONFIG_PATH), "--checkpoint", str(ckpts["dyda"]), "--dir",
                               str(images), "--output", str(out_csv), "--device", "cpu"]) == 0
    df = pd.read_csv(out_csv)
    assert len(df) == 4
    ok = df[df["error"].isna()]
    assert len(ok) == 3
    assert np.allclose(ok[[c for c in df.columns if c.startswith("prob_")]].sum(1), 1.0, atol=1e-5)
    assert df.loc[df["image"].str.endswith("broken.png"), "error"].str.contains("decode").all()


def test_predict_with_gradcam_flag(ckpts, images, tmp_path):
    pred = predict.Predictor(ckpts["cbam"], "cpu")
    info = pred.gradcam(images / "img1.png", tmp_path / "cam", target_class="Normal")
    assert info["gradcam_target_class"] == "Normal"
    assert (tmp_path / "cam" / "overlay.png").is_file()
