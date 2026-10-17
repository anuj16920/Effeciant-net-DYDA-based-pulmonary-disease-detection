"""Metric correctness against hand-computed values, plus ablation statistics helpers."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.ablation import paired_tests
from evaluation.metrics import aggregate, compute_metrics, flatten_scalars, macro_f1, ovr_roc_auc
from evaluation.plots import plot_ablation, plot_confusion_matrix, plot_per_class_bars, plot_roc_curves

CLASSES = ["COVID-19", "Pneumonia", "Tuberculosis", "Normal"]
# Hand-worked example (see comments for the arithmetic).
Y_TRUE = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
Y_PRED = [0, 0, 1, 1, 1, 2, 3, 3, 3, 0]


def test_hand_computed_metrics():
    m = compute_metrics(Y_TRUE, Y_PRED, None, CLASSES)
    # confusion matrix rows = true, cols = predicted
    assert m["confusion_matrix"] == [[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [1, 0, 0, 2]]
    assert m["accuracy"] == pytest.approx(0.7)                                   # 7 / 10
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 1 / 2 + 2 / 3) / 4)  # mean recall
    assert m["macro_precision"] == pytest.approx((2 / 3 + 2 / 3 + 1 + 2 / 3) / 4)    # 0.75
    assert m["macro_recall"] == pytest.approx(m["balanced_accuracy"])
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 2 / 3 + 2 / 3) / 4)       # 0.7
    pc = m["per_class"]
    assert pc["Pneumonia"]["f1"] == pytest.approx(0.8)                           # 2*(2/3*1)/(2/3+1)
    assert pc["COVID-19"]["support"] == 3 and pc["Normal"]["support"] == 3
    # specificity = TN / (TN + FP)
    assert pc["COVID-19"]["specificity"] == pytest.approx(6 / 7)
    assert pc["Pneumonia"]["specificity"] == pytest.approx(7 / 8)
    assert pc["Tuberculosis"]["specificity"] == pytest.approx(1.0)
    assert pc["Normal"]["specificity"] == pytest.approx(6 / 7)
    assert pc["Tuberculosis"]["sensitivity"] == pytest.approx(0.5)
    assert macro_f1(Y_TRUE, Y_PRED, 4) == pytest.approx(0.7)


def test_ovr_auc_hand_computed_and_skips_undefined_classes():
    y = np.array([0, 0, 1, 1])
    p0 = np.array([0.9, 0.6, 0.65, 0.2])
    probs = np.stack([p0, 1 - p0, np.zeros(4), np.zeros(4)], axis=1)
    auc = ovr_roc_auc(y, probs, CLASSES)
    # class 0: positive scores {0.9, 0.6} vs negative {0.65, 0.2}: 3 of 4 pairs ordered -> 0.75
    assert auc["per_class"]["COVID-19"] == pytest.approx(0.75)
    assert auc["per_class"]["Pneumonia"] == pytest.approx(0.75)
    assert auc["per_class"]["Tuberculosis"] is None and auc["per_class"]["Normal"] is None
    assert set(auc["skipped"]) == {"Tuberculosis", "Normal"} and auc["n_classes_used"] == 2
    assert auc["macro"] == pytest.approx(0.75)


def test_absent_class_is_flagged():
    m = compute_metrics([0, 1, 1], [0, 1, 0], None, CLASSES)
    assert m["classes_absent_in_truth"] == ["Tuberculosis", "Normal"] and "note" in m
    assert m["per_class"]["Normal"]["sensitivity"] is None


def test_aggregate_mean_std():
    rows = [{"a": 0.5, "b": None}, {"a": 0.7, "b": 1.0}, {"a": 0.9, "b": None}]
    agg = aggregate(rows)
    assert agg["a"]["mean"] == pytest.approx(0.7) and agg["a"]["std"] == pytest.approx(0.2)  # ddof=1
    assert agg["b"] == {"mean": 1.0, "std": 0.0, "n": 1}


def test_flatten_scalars():
    m = compute_metrics(Y_TRUE, Y_PRED, np.eye(4)[Y_PRED], CLASSES)
    row = flatten_scalars(m)
    assert row["macro_f1"] == pytest.approx(0.7) and row["macro_roc_auc"] is not None
    assert "sens_Tuberculosis" in row and "spec_Normal" in row


def test_paired_tests_low_power():
    a = [0.80, 0.82, 0.79, 0.85, 0.81]
    b = [0.70, 0.71, 0.69, 0.72, 0.70]  # A better on every fold
    t = paired_tests(a, b)
    assert t["mean_diff"] == pytest.approx(0.11)
    assert t["wilcoxon_p"] == pytest.approx(0.0625)  # smallest attainable with n=5 -> never < 0.05
    assert t["ttest_p"] < 0.05
    same = paired_tests(a, a)
    assert same["wilcoxon_p"] == 1.0


def test_plots_write_files(tmp_path):
    m = compute_metrics(Y_TRUE, Y_PRED, np.eye(4)[Y_PRED] * 0.7 + 0.075, CLASSES)
    probs = np.eye(4)[Y_PRED] * 0.7 + 0.075
    for f in (plot_confusion_matrix(m["confusion_matrix"], CLASSES, tmp_path / "cm.png", tag="[SYNTHETIC DATA]"),
              plot_roc_curves(np.array(Y_TRUE), probs, CLASSES, tmp_path / "roc.png", m["roc_auc_ovr"]),
              plot_per_class_bars(m, tmp_path / "bars.png"),
              plot_ablation([{"label": "E1", "macro_f1_mean": 0.5, "macro_f1_std": 0.1}], ["macro_f1"],
                            tmp_path / "abl.png")):
        assert f.is_file() and f.stat().st_size > 1000
