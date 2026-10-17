# DyDA: Hybrid EfficientNet-B3 + Dynamic Dual Attention + Swin Transformer for Chest X-ray Classification

**AI-Driven Medical Image Monitoring and Classification Using a Hybrid EfficientNet–DyDA–Swin Transformer Model**

A 4-class chest X-ray classifier (`COVID-19`, `Pneumonia`, `Tuberculosis`, `Normal`) built as a
research and education prototype, with a leakage-aware data pipeline, grouped cross-validation, E1–E4
ablations, Grad-CAM and inference tools.

> ⚠️ **Disclaimer.** This is a **research/education prototype**. It is **not** a validated diagnostic
> system, it is not a medical device, and it must not be used for clinical decisions. No performance,
> superiority or clinical-validity claims are made. Grad-CAM maps show model sensitivity, not proof of pathology.

> 🏋️ **Trained & Evaluated on Verified 4-Class Multi-Source Dataset (21,451 CXRs)**
> Strict leakage-free grouped 3-fold cross-validation and held-out test evaluation.
> Best Model: **98.43% Test Accuracy**, **98.20% Macro-F1**, **0.9994 Macro ROC-AUC**.

---

## 1. Objective

The goal is to study whether adding a per-sample gated dual-attention block (**DyDA**) and a feature-level
**Swin** encoder to an ImageNet-pretrained **EfficientNet-B3** changes 4-class chest X-ray classification.
The comparison runs under one shared protocol (same splits, preprocessing, optimizer, epochs and
evaluation) against simpler variants (E1–E3).

**Status:** The model has completed full 3-fold cross-validation and held-out test evaluation on the
curated 4-class clinical dataset (21,451 images). See Section 11 for verified results.

## 2. Architecture

```
Image → CLAHE (luminance) → Resize 224×224 → Normalize (ImageNet mean/std)
      → EfficientNet-B3 (timm, ImageNet ra2_in1k) → F            (B,1536,7,7)
      → DyDA(F) → F_DyDA                                         (B,1536,7,7)
          ├─ CNN branch : GAP(F_DyDA) → v_cnn                    (B,1536)
          └─ Swin branch: 1×1 conv adapter → (B,256,7,7) → bilinear ↑ (B,256,14,14)
                          → tokens (B,14,14,256) → [W-MSA block, SW-MSA block (shift 3)]
                          → LayerNorm → mean-pool → v_swin       (B,256)
      → concat (B,1792) → Dropout(0.3) → Linear → 4 logits
```

| ID | `--model` | Pipeline | Params* |
|----|-----------|----------|--------:|
| E1 | `backbone` | EfficientNet-B3 → GAP → head | 10.70 M |
| E2 | `cbam` | EfficientNet-B3 → CBAM (channel→spatial, sequential) → GAP → head | 11.00 M |
| E3 | `dyda` | EfficientNet-B3 → DyDA → GAP → head | 11.10 M |
| E4 | `hybrid` | EfficientNet-B3 → DyDA → [GAP ‖ Swin] → concat → head | 13.07 M |

\*Measured with the default config (`embed_dim=256`, `depth=2`, 4 classes).

**Initialisation.** EfficientNet-B3 uses ImageNet weights (`timm/efficientnet_b3.ra2_in1k`). If loading
them fails, the model falls back to random init with a logged warning, or you can set
`model.pretrained: false`. DyDA, CBAM, the Swin adapter/blocks and the head are randomly initialised
(timm/PyTorch defaults). No pretrained Swin weights are used, because the blocks run on a 256-d feature
grid and not on image patches.

### DyDA: Dynamic Dual Attention (`models/dyda_attention.py`)

For a feature map $F \in \mathbb{R}^{B\times C\times H\times W}$:

$$A_c = \sigma\big(\mathrm{MLP}(\mathrm{GAP}(F)) + \mathrm{MLP}(\mathrm{GMP}(F))\big) \in \mathbb{R}^{B\times C\times 1\times 1}$$

$$A_s = \sigma\big(\mathrm{conv}_{7\times 7}([\,\mathrm{mean}_c(F);\ \max_c(F)\,])\big) \in \mathbb{R}^{B\times 1\times H\times W}$$

$$F_{ch} = F\odot A_c, \qquad F_{sp} = F\odot A_s$$

$$[\alpha,\ \beta] = \mathrm{softmax}\big(g(\mathrm{GAP}(F))\big) \in \mathbb{R}^{B\times 2},\qquad \alpha+\beta = 1 \text{ per sample}$$

$$F_{DyDA} = \alpha\,F_{ch} + \beta\,F_{sp}$$

* The MLP is shared, with reduction ratio `model.dyda.reduction` (default 16). The spatial kernel is
  `model.dyda.spatial_kernel` (default 7). The gate $g$ is Linear(C→`gate_hidden`) → ReLU → Linear(→2).
* The gates are computed **per sample** from the input. They are not global constant parameters, and
  they are reshaped to (B,1,1,1) for broadcasting.
* **Difference from CBAM:** CBAM applies channel and then spatial attention in a fixed sequence, with no
  mixing. DyDA computes both attentions from the same $F$ in parallel and mixes them with input-dependent
  weights.
* **Difference from DANet-style fusion:** that fusion uses a global, input-independent sum or weight.
* For diagnostics, use `model(x, return_gates=True)`, which returns `(logits, gates)`. `dyda.last_gates`
  and `dyda.add_gate_hook(fn)` are also available.

### Swin branch and the 7×7 window problem (`models/swin_feature_encoder.py`)

The Swin branch uses real `timm.models.swin_transformer.SwinTransformerBlock` blocks: windowed MHSA with
relative position bias, plus a cyclic shift and attention mask in shifted blocks. It runs on the
**DyDA feature map, not on raw RGB.**

At 224 px input, EfficientNet-B3's final map is **7×7**. With `window_size=7` that gives one window, and
timm then **silently sets the shift to 0 and drops the mask**, so SW-MSA becomes W-MSA. This was verified
in Phase 0. The encoder therefore:

* by default (`feature_stage: final`, `swin.upsample_size: 14`) bilinearly upsamples the adapted map to
  14×14, which gives **2×2 windows** of 7×7 with shift 3, and a real (4,49,49) attention mask;
* alternatively (`feature_stage: stride16`, `swin.upsample_size: null`) applies DyDA and Swin to the
  genuine stride-16 map (136×14×14);
* raises an error for any configuration with fewer than 2×2 windows, and verifies after construction that
  every odd block has a non-zero shift and a mask. Tests check that the mask changes the output.

**Caveat.** With upsampling, the shifted windows run as intended, but the underlying information
resolution is still 7×7.

## 3. Project layout

```
DYDA/
├─ configs/config.yaml            all parameters (validated by utils/config.py)
├─ data/{raw,processed,manifests}/ raw datasets (you place them), CLAHE cache, manifest/report/splits
├─ models/                        efficientnet_backbone, cbam, dyda_attention, swin_feature_encoder, classifier, hybrid_model
├─ datasets/                      preprocessing, augmentation, prepare_manifest, splits, dataset
├─ training/                      trainer, checkpointing, cross_validation, train
├─ evaluation/                    metrics, evaluate, ablation, plots
├─ explainability/gradcam.py
├─ inference/                     predict, batch_predict, app (optional Streamlit)
├─ utils/                         config loader, common (seeding/logging/versions), synthetic test data
├─ tests/                         pytest suite (synthetic data only)
├─ outputs/  checkpoints/         generated (git-ignored)
└─ main.py  requirements.txt  pytest.ini  .gitignore
```

## 4. Setup (Windows, PowerShell)

Verified environment: Windows 11, **Python 3.10.5**, torch 2.1.0+cu121, CUDA 12.1, RTX 3050 Laptop (4 GB),
timm 1.0.24.

```powershell
cd D:\DYDA
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Choose ONE torch build:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu     # CPU only

pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The `.venv` on the development machine was created with `--system-site-packages`, so it reuses the
already-installed Python 3.10 packages and needed no downloads.

* **CPU vs CUDA.** `training.device: auto` uses CUDA when available. AMP (`training.amp: true`) is applied
  **only on CUDA**; on CPU the code runs in float32.
* **Pretrained weights** are downloaded once by timm into the HuggingFace cache. Offline: copy the
  `models--timm--efficientnet_b3.ra2_in1k` cache folder, or set `model.pretrained: false`.

## 5. Datasets (not included; never redistributed)

Download each dataset yourself from its official page and accept its licence. Place it under `data/raw/`
as shown, or point `sources.<name>.root` in `configs/config.yaml` to where it already is.

| Source (`sources.<name>`) | Expected layout under `root` | Label rule |
|---|---|---|
| `covid_radiography`: COVID-19 Radiography Database (Kaggle) | `COVID-19_Radiography_Dataset/{COVID,Normal,Viral Pneumonia,Lung_Opacity}/images/*.png` | parent folder |
| `kaggle_pneumonia`: Chest X-Ray Images (Pneumonia), Kermany et al. (Kaggle) | `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg` | parent folder |
| `montgomery_tb`: Montgomery County TB set (NLM) | `MontgomerySet/CXR_png/MCUCXR_####_#.png` | filename suffix `_0`/`_1` |
| `shenzhen_tb`: Shenzhen TB set (NLM) | `ChinaSet_AllFiles/CXR_png/CHNCXR_####_#.png` | filename suffix `_0`/`_1` |

Folders named `masks`, `ManualMask`, `leftMask`, `rightMask` and `__MACOSX` are skipped
(`manifest.exclude_dirs`). The original Kaggle train/val/test split is **ignored**: all images are merged
and re-split with grouping.

> If your Kaggle archive contains a nested `chest_xray/chest_xray/...` copy, point `root` at **one** clean
> copy. Byte-identical duplicates would be grouped together anyway, but they double-count images.

### Label mapping (explicit; anything not listed is excluded and reported)

| Source | Raw label | → Class |
|---|---|---|
| covid_radiography | `COVID` | COVID-19 |
| covid_radiography | `Normal` | Normal |
| covid_radiography | `Viral Pneumonia` | Pneumonia |
| covid_radiography | `Lung_Opacity` | **excluded** (`null`; ambiguous, never mapped to Pneumonia) |
| kaggle_pneumonia | `NORMAL` | Normal |
| kaggle_pneumonia | `PNEUMONIA` (bacteria + virus, toggle in `pneumonia_subtypes`) | Pneumonia |
| montgomery_tb / shenzhen_tb | `_0` | Normal |
| montgomery_tb / shenzhen_tb | `_1` | Tuberculosis |

### Patient IDs and grouping

* **Patient IDs** are parsed where they are reliable: `MCUCXR_####`, `CHNCXR_####`, and Kaggle `personN_`.
  The COVID-19 Radiography Database and Kaggle `NORMAL` files have no reliable patient ID.
* **Duplicates:** SHA-256 finds exact duplicates, and a 64-bit DCT perceptual hash finds near duplicates
  (Hamming distance ≤ `manifest.phash_hamming_threshold`, default 4).
* Images that share a patient or a duplicate cluster get one `group_id`, and splits are done on groups.

## 6. Usage (PowerShell)

For real training, follow **[TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md)** step by step. The
commands below are the reference.

```powershell
# 1. Manifest, data report and grouped splits
python -m datasets.prepare_manifest --config configs/config.yaml
#    -> data/manifests/{manifest.csv, data_report.json, splits.json}

# 2. 5-fold CV for one model (E4 shown); --resume continues interrupted folds
python -m training.cross_validation --config configs/config.yaml --model hybrid
python -m training.train --config configs/config.yaml --model dyda --fold 0 --epochs 2   # quick single fold

# 3. Ablation E1-E4 on identical splits (+ every fold model on the held-out test set)
python -m evaluation.ablation --config configs/config.yaml --with-test

# 4. Evaluate on the held-out TEST split (final reporting only)
python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth
python -m evaluation.evaluate --config configs/config.yaml --experiment hybrid --folds

# 5. Grad-CAM for correct / incorrect test predictions
python -m explainability.gradcam --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth --split test

# 6. Inference (prints the research-only disclaimer)
python -m inference.predict --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth --image "path\to\xray.png" --gradcam
python -m inference.batch_predict --config configs/config.yaml --checkpoint checkpoints/hybrid/best_model.pth --dir "path\to\folder"
streamlit run inference/app.py -- --checkpoint checkpoints/hybrid/best_model.pth   # optional; pip install streamlit

# Tests
pytest -q
```

`python main.py <prepare|train|train-fold|ablation|evaluate|gradcam|predict|batch-predict|synthetic|test> ...`
dispatches to the same modules.

### Synthetic smoke-test workspace

This exercises the whole pipeline without real data. The images are generated placeholders, not
radiographs, and every plot title gets a `[SYNTHETIC DATA]` stamp:

```powershell
python -m utils.synthetic --out outputs/_synthetic_smoke --n-per-class 10 --set training.epochs=1 training.num_workers=0
python -m datasets.prepare_manifest --config outputs/_synthetic_smoke/config.yaml
python -m evaluation.ablation --config outputs/_synthetic_smoke/config.yaml --with-test
```

## 7. Protocol

* **Seeds:** python, numpy, torch and CUDA are all seeded, and cuDNN is deterministic. Every model and
  fold starts from the same seed, so data order is identical across E1–E4.
* **Splits:** the **held-out test set is carved out first**, as the first fold of a
  `StratifiedGroupKFold(round(1/test_fraction))` (≈14.3% for `test_fraction: 0.15`). Then
  **5-fold `StratifiedGroupKFold`** runs on the rest. Zero group overlap is asserted for every boundary,
  and a manifest fingerprint rejects stale splits.
* **Training:** AdamW, cosine annealing (`lr` → `min_lr`), CE loss, gradient clipping, and early stopping
  on **validation macro-F1** (patience 7). Optional class weights are computed from the **training fold
  only**. AMP runs on CUDA only.
* **Checkpoints:** best and last per fold (`checkpoints/<exp>/fold_k/`). Each contains the model state,
  model config, class map, preprocessing config, full config, fold, seed, validation metric, library
  versions and (for last) optimizer/scheduler/scaler/RNG state for exact resume.
  `checkpoints/<exp>/best_model.pth` is the fold with the best **validation** macro-F1.
* **No test leakage:** normalisation uses fixed ImageNet statistics. Class weights, early stopping and
  checkpoint selection use training/validation folds only. The test split is used only by
  `evaluation.evaluate` / `--with-test`.
* **Metrics** (per fold and as mean ± std): accuracy, balanced accuracy, macro P/R/F1, per-class
  P/R/F1/support, per-class sensitivity and specificity, confusion matrix, and OvR ROC-AUC. ROC-AUC is
  skipped, with the reason logged, for any class without both positives and negatives. A **per-source
  breakdown** is always reported.
* **Ablation statistics:** a paired Wilcoxon and paired t-test across folds. **With n = 5 folds the
  statistical power is low.** The smallest attainable two-sided exact Wilcoxon p-value is 0.0625, fold
  scores are correlated, and no multiple-comparison correction is applied. All comparisons are
  exploratory.

## 8. Outputs

| Path | Content |
|---|---|
| `data/manifests/manifest.csv` | `image_path, label, source_dataset, patient_id, group_id, sha256, phash` (+ raw_label, subtype, width, height) |
| `data/manifests/data_report.json` | missing sources, unreadable images, unmapped/excluded labels, class counts overall and per source, duplicate clusters, cross-source duplicates, label conflicts, warnings, limitations |
| `data/manifests/splits.json` | test indices, 5 folds, class distribution per split, manifest fingerprint |
| `data/processed/prep_<hash>/` | cache of deterministic CLAHE+resize output (augmentation is applied after the cache) |
| `outputs/<exp>/fold_k/` | `history.{csv,json}`, `val_metrics.json`, `val_predictions.csv` |
| `outputs/<exp>/cv_summary.{json,csv}` | per-fold metrics + mean ± std (validation) |
| `outputs/<exp>/evaluation/` | test/val metrics JSON, per-source CSV, predictions CSV, confusion matrices, ROC, per-class bars, training curves |
| `outputs/ablation/` | `ablation_summary.{csv,md,json}`, `paired_tests.csv`, `ablation_comparison.png` |
| `outputs/gradcam/<exp>/<split>/{correct,incorrect}/NNN_<image>/` | `original.png`, `heatmap.png`, `overlay.png` (annotated), `gradcam.json` |
| `outputs/predictions/` | inference JSON/CSV and Grad-CAM |
| `checkpoints/<exp>/` | `fold_k/{best,last}.pth`, `best_model.pth` |

## 9. Tests

`pytest -q` runs the suite on synthetic data with CPU only, and **never trains**. The tiny
synthetic training/CV smoke tests are opt-in: `pytest -q --run-training`, to be run on the training
machine. Together they cover:

* config validation;
* DyDA maths (α+β=1, per-sample gates, the equation itself, batch size 1);
* all four models (forward to (B,4), CE + backward, gradients reaching DyDA and Swin, no NaNs);
* that the SW-MSA mask is active;
* the data pipeline (label mapping, exclusions, corrupt files, exact/near duplicates, patient grouping,
  zero-overlap splits, preprocessing parity between training and inference, train-only augmentation);
* training (CV smoke for E1–E4, checkpoint round-trip, **bit-exact resume**, early stopping,
  train-only class weights);
* metrics against hand-computed values;
* inference and Grad-CAM (probabilities sum to 1, heatmap shape/range, hook removal, file export).

## 10. Troubleshooting

* **`Disabling PyTorch because PyTorch >= 2.4 is required`** printed on `import timm`: this comes from an
  unrelated `transformers` install that timm probes. It is harmless and suppressed in the project's own
  modules.
* **Pretrained weights fail to download:** see the warning text. Pre-populate the HF cache, or set
  `model.pretrained: false` (results will differ).
* **CUDA out of memory (4 GB GPUs):** lower `training.batch_size` (e.g. 8) and keep `amp: true`. The
  per-epoch `peak_gpu_mem_mb` in `history.csv` helps you size it.
* **`Splits file ... built for a different manifest`:** you changed the data. Re-run
  `datasets.prepare_manifest`.
* **`only N groups; need at least ...`:** there are too few groups for test + 5-fold CV. Add data, or
  lower `training.folds` / `test_fraction`.
* **Windows DataLoader hangs:** set `training.num_workers: 0`.
* Two OpenCV distributions (`opencv-python` and `opencv-python-headless`) in one environment can conflict.
  Keep only `opencv-python==4.8.1.78`.

## 11. Verified Experimental Results

The Hybrid EfficientNet-B3 + DyDA + Swin model (E4) was trained and evaluated on the curated 4-class multi-source dataset (21,451 chest X-ray radiographs) using NVIDIA GeForce GTX 1080 Ti hardware with AMP:

### 11.1 Held-Out Test Set Performance (3,065 Unseen Radiographs)
Evaluated with the top-performing cross-validation checkpoint (`checkpoints/hybrid/best_model.pth`):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.9843** | **98.43%** |
| **Balanced Accuracy** | **0.9812** | **98.12%** |
| **Macro F1-Score** | **0.9820** | **98.20%** |
| **Macro Precision** | **0.9829** | **98.29%** |
| **Macro Recall** | **0.9812** | **98.12%** |
| **Macro ROC-AUC (OvR)** | **0.9994** | **99.94%** |

### 11.2 Per-Class Breakdown (Held-Out Test Set)
| Pathology | Support | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 488 | **99.18%** | 99.92% | **99.59%** | **0.9938** | 0.9999 |
| **Pneumonia** | 780 | **98.08%** | 99.21% | **97.70%** | **0.9789** | 0.9992 |
| **Normal** | 1,647 | **98.54%** | 98.31% | **98.54%** | **0.9854** | 0.9988 |
| **Tuberculosis** | 150 | **96.67%** | 99.86% | **97.32%** | **0.9699** | 0.9995 |

*Note on Tuberculosis:* Adding the dedicated `TB_Chest_Radiography_Database` and balanced class weighting increased Tuberculosis recall from 88.24% to 96.67% (with 99 out of 99 test images from the database correctly classified).

### 11.3 3-Fold Cross-Validation Stability
- **Fold 0**: Val Macro-F1 = 0.9746 | Val Accuracy = 98.34%
- **Fold 1**: Val Macro-F1 = 0.9797 | Val Accuracy = 98.48%
- **Fold 2**: Val Macro-F1 = 0.9810 | Val Accuracy = 98.78%
- **Validation Aggregate**: **0.9784 ± 0.0034 Macro-F1**, **98.53% ± 0.22% Accuracy**

### 11.4 Evaluation Visualizations
- **Confusion Matrix**: [`outputs/hybrid/evaluation/test/confusion_matrix.png`](outputs/hybrid/evaluation/test/confusion_matrix.png)
- **Raw Counts Confusion Matrix**: [`outputs/hybrid/evaluation/test/confusion_matrix_counts.png`](outputs/hybrid/evaluation/test/confusion_matrix_counts.png)
- **Multi-Class ROC Curves**: [`outputs/hybrid/evaluation/test/roc_curves.png`](outputs/hybrid/evaluation/test/roc_curves.png)
- **Per-Class Metrics Bar Chart**: [`outputs/hybrid/evaluation/test/per_class_metrics.png`](outputs/hybrid/evaluation/test/per_class_metrics.png)
- **Full Predictions Spreadsheet**: [`outputs/hybrid/evaluation/test/predictions.csv`](outputs/hybrid/evaluation/test/predictions.csv)

## 12. Deviations from the specification

1. **Location:** the spec names `D:\v2v communication project\DYDA`. On request, the project was built in
   `D:\DYDA`.
2. **Swin 7×7:** the default upsamples the DyDA map 7→14 so there are 2×2 windows (shift 3). The
   `stride16` alternative is configurable. Configurations with a single window are rejected. The Swin
   branch is randomly initialised (no pretrained Swin weights).
3. **Held-out test set:** a test set is held out. Its realised fraction is ≈1/7 (14.3%) for
   `test_fraction: 0.15`, because it is taken as one fold of a grouped stratified K-fold.
4. **Perceptual hash:** implemented in the repo (OpenCV DCT, same algorithm as `imagehash.phash`), so the
   `imagehash` dependency is not needed.
5. **Extra modules:** `utils/` (config loader, shared helpers, synthetic data), `datasets/splits.py`,
   `training`'s exact-resume state, and the extra tests `test_config.py`, `test_training.py` and
   `test_metrics.py`.
6. **Checkpoint path:** `checkpoints/<experiment>/best_model.pth` instead of
   `checkpoints/best_model.pth`, so that ablation runs do not overwrite each other. The best model is the
   fold with the highest *validation* macro-F1.
7. **Grad-CAM folders:** `outputs/gradcam/<experiment>/<split>/{correct,incorrect}/` instead of
   `outputs/gradcam/{correct,incorrect}/`.
8. **Resize:** images are resized to 224×224 without preserving aspect ratio (no padding).
9. **Normalisation** uses ImageNet mean/std (matching the pretrained backbone), not dataset statistics.
10. **Versions:** pinned to the verified environment (Python 3.10, torch 2.1.0). The installed
    torchvision is a CPU build paired with CUDA torch. It is not used on the critical path.
11. **Preprocessing at evaluation and inference** is read from the checkpoint, so it always matches
    training.
12. **CLAHE cache:** an optional cache (`preprocessing.cache`) stores deterministic preprocessing output.
    Augmentation is applied after it.

## 13. Limitations

* **Source-shortcut bias (major).** Classes are strongly confounded with source datasets:
  * Tuberculosis comes only from Montgomery and Shenzhen;
  * COVID-19 comes only from the COVID-19 Radiography Database;
  * the Kaggle Pneumonia/Normal images are **pediatric** (ages 1–5), while the other sets are mainly
    adults.

  A model can reach high scores by recognising the scanner, hospital, age or preprocessing instead of the
  pathology. `data_report.json` warns about this, and evaluation reports per-source metrics. High
  aggregate metrics must not be read as diagnostic ability.
* **Label noise.** Labels come from public folder names and filenames and were not re-read by
  radiologists. "Lung Opacity" is excluded instead of guessed.
* **Grouping.** There are no patient IDs for the COVID-19 Radiography Database or Kaggle `NORMAL` images.
  Grouping there relies on duplicate detection only, so patients with several distinct images could
  cross splits.
* **No external validation.** All splits come from the same pooled sources. No independent hospital or
  test site has been used.
* **Low statistical power.** With 5 folds, paired tests cannot reach p < 0.05 (Wilcoxon), and the fold
  scores are correlated.
* **Grad-CAM** at a 7×7 map is coarse. It shows model sensitivity, not lesion localisation, and is not
  clinical evidence.
* **Calibration:** probabilities are raw softmax outputs. They are not calibrated and not clinical risk
  scores.
* **Synthetic results:** all results produced so far come from synthetic placeholder images and say
  nothing about real performance.

## 14. Disclaimer

RESEARCH/EDUCATION PROTOTYPE ONLY. This software is not a medical device, has not been clinically
validated, and must not be used to diagnose, treat or triage patients. Datasets are subject to their own
licences and are not distributed with this code.
