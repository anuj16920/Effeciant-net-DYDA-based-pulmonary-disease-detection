"""E1-E4 ablation on identical splits, preprocessing, epochs, optimizer and evaluation.

Usage::

    python -m evaluation.ablation --config configs/config.yaml [--models backbone,cbam,dyda,hybrid]
                                  [--folds 5] [--epochs N] [--skip-train] [--with-test]

Writes ``outputs/ablation/``: ``ablation_summary.{csv,md,json}``, ``ablation_comparison.png``
and ``paired_tests.csv``.

Statistical caveat: with n=5 folds, paired tests have very low power. The smallest
attainable two-sided exact Wilcoxon signed-rank p-value with n=5 is 0.0625, so a
"significant" result at alpha=0.05 is impossible by construction. Fold scores are also
not independent (overlapping training sets). Treat all comparisons as exploratory.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from datasets.dataset import load_manifest
from datasets.splits import load_splits
from evaluation.evaluate import evaluate_experiment_folds
from evaluation.plots import plot_ablation
from models.hybrid_model import build_model, count_parameters
from training.cross_validation import parse_folds, run_cv
from utils.common import load_json, save_json, setup_logging
from utils.config import ABLATION_IDS, MODEL_IDS, get_path, load_config, with_model
from utils.synthetic import data_tag

logger = logging.getLogger(__name__)

TABLE_METRICS = ["macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "macro_roc_auc"]
POWER_NOTE = ("n={n} folds gives LOW statistical power; the minimum two-sided exact Wilcoxon p-value at n=5 is "
              "0.0625, fold scores are correlated (overlapping training data), and no multiple-comparison "
              "correction is applied. These comparisons are exploratory only and do not establish superiority.")


def paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Paired comparison of per-fold scores ``a`` vs ``b`` (same folds, same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out: Dict[str, Optional[float]] = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None,
                                       "wilcoxon_p": None, "ttest_p": None}
    if len(d) < 2:
        return out
    if np.allclose(d, 0):
        out["wilcoxon_p"] = 1.0
        out["ttest_p"] = 1.0
        return out
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError as exc:  # e.g. too few non-zero differences
        logger.info("Wilcoxon not computed: %s", exc)
    if np.std(d, ddof=1) > 0:
        out["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    return out


def _fmt(m: Optional[float], s: Optional[float]) -> str:
    if m is None:
        return "n/a"
    return f"{m:.4f} ± {s:.4f}" if s is not None else f"{m:.4f}"


def aggregate_ablation(cfg: Dict[str, Any], models: List[str], with_test: bool = False) -> Dict[str, Any]:
    """Collect each model's cv_summary.json into one table, chart and paired-test file."""
    out_dir = get_path(cfg, "outputs_dir") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = data_tag(cfg)
    summaries, rows, chart_rows = {}, [], []
    for m in models:
        p = get_path(cfg, "outputs_dir") / m / "cv_summary.json"
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing; run training for model '{m}' first (or drop --skip-train)")
        summaries[m] = load_json(p)
    fold_sets = {m: tuple(s["folds"]) for m, s in summaries.items()}
    if len(set(fold_sets.values())) != 1:
        raise ValueError(f"models were trained on different fold sets {fold_sets}; comparison would be unfair")
    for m in models:
        agg = summaries[m]["aggregate"]
        n_params = count_parameters(build_model(dict(cfg, model=dict(cfg["model"], pretrained=False)), m))
        row = {"id": ABLATION_IDS[m], "model": m, "params": n_params, "n_folds": len(summaries[m]["folds"])}
        chart = {"label": f"{ABLATION_IDS[m]} {m}"}
        for k in TABLE_METRICS:
            a = agg.get(k, {"mean": None, "std": None})
            row[f"{k}_mean"], row[f"{k}_std"] = a["mean"], a["std"]
            chart[f"{k}_mean"], chart[f"{k}_std"] = a["mean"], a["std"]
        if with_test:
            fs = get_path(cfg, "outputs_dir") / m / "evaluation" / "folds_summary.json"
            if fs.is_file() and "test_aggregate" in load_json(fs):
                ta = load_json(fs)["test_aggregate"]
                for k in ("macro_f1", "balanced_accuracy"):
                    row[f"test_{k}_mean"], row[f"test_{k}_std"] = ta[k]["mean"], ta[k]["std"]
        rows.append(row)
        chart_rows.append(chart)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ablation_summary.csv", index=False)

    # paired tests on per-fold validation macro-F1 and balanced accuracy
    tests = []
    for x, y in itertools.combinations(models, 2):
        for metric in ("macro_f1", "balanced_accuracy"):
            fa = {r["fold"]: r[metric] for r in summaries[x]["per_fold"]}
            fb = {r["fold"]: r[metric] for r in summaries[y]["per_fold"]}
            common = sorted(set(fa) & set(fb))
            t = paired_tests([fa[f] for f in common], [fb[f] for f in common])
            tests.append({"a": f"{ABLATION_IDS[x]} {x}", "b": f"{ABLATION_IDS[y]} {y}", "metric": metric, **t})
    pd.DataFrame(tests).to_csv(out_dir / "paired_tests.csv", index=False)
    n = len(next(iter(fold_sets.values())))
    note = POWER_NOTE.format(n=n)

    md = [f"# Ablation summary {tag}".rstrip(), "",
          "Validation-fold metrics (mean ± sample std over folds). Identical splits, preprocessing, epochs, "
          "optimizer and evaluation for every model.", ""]
    md.append("| ID | Model | Params | " + " | ".join(TABLE_METRICS) + " |")
    md.append("|---|---|---:|" + "---:|" * len(TABLE_METRICS))
    for r in rows:
        md.append(f"| {r['id']} | {r['model']} | {r['params']:,} | " +
                  " | ".join(_fmt(r[f'{k}_mean'], r[f'{k}_std']) for k in TABLE_METRICS) + " |")
    md += ["", "## Paired comparisons (exploratory)", "", f"> {note}", "",
           "| A | B | metric | mean diff (A−B) | Wilcoxon p | paired t p |", "|---|---|---|---:|---:|---:|"]
    for t in tests:
        f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        md.append(f"| {t['a']} | {t['b']} | {t['metric']} | {f(t['mean_diff'])} | {f(t['wilcoxon_p'])} | "
                  f"{f(t['ttest_p'])} |")
    if tag:
        md += ["", "**These numbers come from SYNTHETIC placeholder images and carry no meaning about real "
                   "performance.**"]
    (out_dir / "ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_ablation(chart_rows, ["macro_f1", "balanced_accuracy", "accuracy", "macro_roc_auc"],
                  out_dir / "ablation_comparison.png", tag=tag)
    result = {"models": models, "rows": rows, "paired_tests": tests, "statistical_note": note,
              "synthetic_data": bool(tag), "folds": list(next(iter(fold_sets.values())))}
    save_json(result, out_dir / "ablation_summary.json")
    logger.info("Ablation summary written to %s", out_dir)
    logger.warning(note)
    return result


def run_ablation(cfg: Dict[str, Any], models: Sequence[str] = MODEL_IDS, folds: Optional[List[int]] = None,
                 skip_train: bool = False, with_test: bool = False) -> Dict[str, Any]:
    manifest = load_manifest(cfg)
    load_splits(get_path(cfg, "splits"), manifest)  # all models share this exact split file
    for m in models:
        mcfg = with_model(cfg, m)
        if not skip_train:
            run_cv(mcfg, folds)
        if with_test:
            evaluate_experiment_folds(mcfg, m, with_test=True)
    return aggregate_ablation(cfg, list(models), with_test)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run E1-E4 ablation with identical settings")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--models", default=",".join(MODEL_IDS))
    ap.add_argument("--folds", default=None, help="'N' = first N folds, or comma list")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true", help="only aggregate existing cv_summary.json files")
    ap.add_argument("--with-test", action="store_true", help="also evaluate every fold model on the held-out test")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = set(models) - set(MODEL_IDS)
    if bad:
        ap.error(f"unknown models {sorted(bad)}")
    setup_logging(log_file=get_path(cfg, "outputs_dir") / "ablation" / "ablation.log")
    run_ablation(cfg, models, parse_folds(args.folds), args.skip_train, args.with_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
