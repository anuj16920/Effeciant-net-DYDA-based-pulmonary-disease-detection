#!/usr/bin/env bash
# ==============================================================================
# 4-Class Medical X-Ray Training Launcher (High-Performance Hardware Optimized)
# Classes: COVID-19, Pneumonia, Tuberculosis, Normal
# Target Hardware: NVIDIA GTX 1080 Ti (11GB VRAM) + 12-core CPU + 46GB RAM
# ==============================================================================
set -e

# Navigate to project root
cd "$(dirname "$0")"

# 1. Hardware & Memory Tuning Environment Variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYTHONUNBUFFERED=1

PYTHON="./.venv/bin/python"

# Default to running Hybrid (E4) or DyDA (E3) or full ablation suite
TARGET="${1:-hybrid}"
FOLDS="${2:-3}"

RESUME_ARG=""
TEE_MODE="tee"
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME_ARG="--resume"
        TEE_MODE="tee -a"
    fi
done

echo "============================================================"
echo " Starting 4-Class Training Setup"
echo " Target Model : $TARGET"
echo " Folds        : $FOLDS"
echo " Resume       : ${RESUME_ARG:-false}"
echo " Classes      : COVID-19, Pneumonia, Tuberculosis, Normal"
echo " VRAM Profile : Batch Size 104 with AMP (~9.5 - 10.2 GB VRAM)"
echo " RAM Profile  : Zero-latency in-memory dataset cache (12 workers)"
echo " Log file     : training_${TARGET}_4class.log"
echo "============================================================"

if [ "$TARGET" = "ablation" ]; then
    echo "Running complete 4-model ablation suite (backbone, cbam, dyda, hybrid)..."
    $PYTHON -m evaluation.ablation \
        --config configs/config.yaml \
        --models backbone,cbam,dyda,hybrid \
        --folds "$FOLDS" \
        --with-test 2>&1 | tee ablation_run.log
else
    echo "Running cross-validation for model: $TARGET..."
    $PYTHON -m training.cross_validation \
        --config configs/config.yaml \
        --model "$TARGET" \
        --folds "$FOLDS" $RESUME_ARG 2>&1 | $TEE_MODE "training_${TARGET}_4class.log"

    echo "Evaluating best checkpoint on held-out test split..."
    $PYTHON -m evaluation.evaluate \
        --config configs/config.yaml \
        --checkpoint "checkpoints/${TARGET}/best_model.pth" 2>&1 | tee -a "training_${TARGET}_4class.log"

    echo "Evaluating all fold models and generating figures..."
    $PYTHON -m evaluation.evaluate \
        --config configs/config.yaml \
        --experiment "$TARGET" \
        --folds 2>&1 | tee -a "training_${TARGET}_4class.log"
fi

echo "============================================================"
echo " Training and Evaluation Complete!"
echo "============================================================"
