# TRAINING INSTRUCTIONS (strict protocol)

Training is done on a **separate training machine**, not on the development laptop. Follow these steps
**in order**. Every "MUST / MUST NOT" is part of the protocol. If you break one, the results are not
comparable and must not be reported as E1–E4 results.

> Research/education prototype. Not a diagnostic system. Do not report any number as clinical performance.

---

## 0. Freeze the code and config

1. Copy the whole `DYDA/` folder **without** `outputs/`, `checkpoints/`, `data/processed/`,
   `data/manifests/` or `.venv/`. These are regenerated on the training machine.
2. Before the first run, record the exact code and config:
   ```powershell
   git init; git add -A; git commit -m "frozen for training run"    # or note a zip checksum
   Get-FileHash configs\config.yaml -Algorithm SHA256
   ```
3. Once training of the ablation has started, **MUST NOT** edit any code or `configs/config.yaml`. Any
   change means re-running **all** of E1–E4 on all folds.

## 1. Environment on the training machine

Python 3.10 is recommended (the code was verified with 3.10.5, torch 2.1.0, timm 1.0.24).

```powershell
cd <path>\DYDA
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121   # match the machine's CUDA
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q                     # all tests MUST pass (training smoke tests are skipped)
pytest -q --run-training      # MUST pass on the training machine before real training (tiny synthetic CPU runs)
```

If you use a newer torch/timm, run both pytest commands first and record the versions. The versions are
also saved inside every checkpoint.

## 2. Data placement (no download scripts; you obtain each dataset under its licence)

Place the four sources as described in `README.md` §5. Alternatively, set `sources.<name>.root` in
`configs/config.yaml` to absolute paths **before freezing the config**.

* **MUST** include all four sources. Without the COVID-19 Radiography Database or the TB sets, a
  4-class model cannot be trained honestly. `prepare_manifest` will warn.
* **MUST NOT** map `Lung_Opacity` or any other unlisted label to a class.
* **MUST NOT** use the original Kaggle train/val/test folders as splits. The pipeline merges and
  re-splits them.
* If the Kaggle archive has a nested `chest_xray/chest_xray` copy, use only **one** copy.

## 3. Manifest, report and splits (run once)

```powershell
python -m datasets.prepare_manifest --config configs/config.yaml
```

Then **STOP and review** `data/manifests/data_report.json` before any training:

* `class_counts_overall`, `class_counts_per_source`: every class must have images.
* `missing_sources`, `unreadable_images`, `unmapped_labels`, `excluded_by_label_map`.
* `duplicates.cross_source_clusters`, `label_conflicts`, `max_group_size`. If `max_group_size` is
  unexpectedly large, lower `manifest.phash_hamming_threshold` and re-run this step. This is the only
  allowed config change, and it is allowed only **before** any training.
* `warnings`: keep them; they go into the report.

Rules:

* `splits.json` is created **once** and **MUST NOT** be regenerated after training starts. Regenerating
  it changes the held-out test set.
* Keep `manifest.csv`, `data_report.json` and `splits.json` together with the results.

## 4. Size the batch on the GPU (one short probe; the probe is not a result)

```powershell
python -m training.train --config configs/config.yaml --model hybrid --fold 0 --epochs 1
```

* Check `outputs/hybrid/fold_0/history.csv`: `seconds` gives the time per epoch, and `peak_gpu_mem_mb`
  gives memory use.
* The largest model is E4 (`hybrid`). Choose **one** `training.batch_size` that fits E4. If you get
  CUDA OOM on a 4 GB GPU, try 8.
* Set `training.batch_size` and `training.epochs` in `configs/config.yaml`, **then** freeze the config
  (step 0).
* **Delete the probe outputs** before the real run so they cannot mix with it:
  `Remove-Item -Recurse outputs\hybrid, checkpoints\hybrid`.

## 5. Settings that MUST be identical for E1, E2, E3 and E4

| Setting | Key | Rule |
|---|---|---|
| Splits | `data/manifests/splits.json` | same file for all models |
| Seed | `training.seed` | same (default 42) |
| Epochs, early stopping | `training.epochs`, `training.early_stopping.*` | same |
| Optimizer / LR / schedule | `training.lr`, `min_lr`, `weight_decay` | same |
| Batch size, AMP, grad clip | `training.batch_size`, `amp`, `grad_clip` | same |
| Class weights | `training.class_weights` | same (computed from the training fold only) |
| Preprocessing / augmentation | `preprocessing.*`, `augmentation.*` | same |
| Backbone | `model.backbone`, `pretrained`, `feature_stage`, `freeze_backbone` | same |

You **MUST NOT** tune hyperparameters for one model only, and **MUST NOT** pick settings by looking at
the test set.

## 6. Run the ablation (the actual training)

Run everything in one go (recommended):

```powershell
python -m evaluation.ablation --config configs/config.yaml --with-test
```

or one model at a time, all 5 folds each, using the same config:

```powershell
python -m training.cross_validation --config configs/config.yaml --model backbone   # E1
python -m training.cross_validation --config configs/config.yaml --model cbam       # E2
python -m training.cross_validation --config configs/config.yaml --model dyda       # E3
python -m training.cross_validation --config configs/config.yaml --model hybrid     # E4
python -m evaluation.ablation --config configs/config.yaml --skip-train --with-test
```

* **If interrupted:** re-run the same command with `--resume`. This continues from each fold's
  `last.pth` and gives exactly the same result. Do **not** restart from scratch with a different seed.
* **MUST** run all 5 folds for every model. **MUST NOT** drop, re-run or cherry-pick folds or seeds
  because their results look bad.
* If a run crashes for a technical reason (OOM, power loss), resume it. If you must change a setting to
  fix it, the change applies to **all** models, and all of them must be re-run.

## 7. Held-out test set: final reporting only

* The test split is used **only** by `--with-test` and `evaluation.evaluate`. Checkpoint selection, early
  stopping and class weights already use training/validation folds only.
* **MUST NOT** change anything (config, epochs, model choice) after looking at test results and then
  re-test. If that happens, say so explicitly in the report.

```powershell
python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth
python -m explainability.gradcam --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth --split test
```

## 8. What to bring back from the training machine

Copy these folders (checkpoints are large, so `last.pth` files may be left out):

* `data/manifests/` (`manifest.csv`, `data_report.json`, `splits.json`);
* `outputs/` (all `cv_summary.*`, `evaluation/`, `ablation/`, `gradcam/`, `*.log`, `history.csv`);
* `checkpoints/<model>/best_model.pth` and `checkpoints/<model>/fold_k/best.pth`;
* `configs/config.yaml` exactly as used, plus the git commit or checksum and
  `pip freeze > outputs/pip_freeze.txt`.

## 9. How to report (honesty rules)

* Report **validation** results as mean ± std over 5 folds (`cv_summary`), and **test** results
  separately (`--with-test` gives mean ± std of the 5 fold models on the same test set).
* Always include the **per-source breakdown** and the `data_report.json` warnings (source-shortcut risk,
  pediatric Kaggle set).
* Paired tests across 5 folds are **exploratory**. Wilcoxon cannot go below p = 0.0625 with n = 5. Do
  **not** claim that one model is "significantly better" or "superior".
* Do not call the model diagnostic or clinically validated. Grad-CAM is sensitivity visualisation, not
  lesion localisation.
* Report everything that was run, including failed and interrupted runs.

## Quick checklist

- [ ] `pytest -q` and `pytest -q --run-training` pass on the training machine
- [ ] All 4 sources present; `data_report.json` reviewed
- [ ] `splits.json` created once, never regenerated
- [ ] Batch size chosen with the E4 probe; probe outputs deleted; config frozen and hashed
- [ ] E1–E4 × 5 folds trained with identical settings (resume only, no restarts with new seeds)
- [ ] Test set evaluated once, after all training
- [ ] Results, logs, config, versions and data report copied back
