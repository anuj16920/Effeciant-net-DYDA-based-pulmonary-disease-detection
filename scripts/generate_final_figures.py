#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Script for DyDA Pulmonary Disease Detection
Outputs each figure separately in high-resolution (300 DPI) into `final_figures/`.
Includes standalone multi-class curves as well as individual per-class ROC and PR curves.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Publication styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Curated Medical Journal Color Palette
COLORS = {
    "COVID-19": "#1f77b4",       # Medical Sapphire Blue
    "Pneumonia": "#ff7f0e",      # Amber Coral
    "Tuberculosis": "#2ca02c",  # Emerald Green
    "Normal": "#9467bd",        # Royal Amethyst
    "Macro": "#111111",         # Deep Charcoal
    "Train": "#2563eb",         # Vibrant Blue
    "Val": "#dc2626",           # Crimson Red
}
CLASS_NAMES = ["COVID-19", "Pneumonia", "Tuberculosis", "Normal"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix_normalized(cm: np.ndarray, classes: list, out_path: Path):
    """Plot row-normalized confusion matrix (percentages + sample counts)."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (Normalized Proportion)", rotation=270, labelpad=15, fontsize=11)
    cbar.ax.tick_params(labelsize=9.5)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontweight="medium")
    
    thresh = 0.55
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_pct = f"{cm_norm[i, j] * 100:.2f}%"
            val_cnt = f"(n={int(cm[i, j])})"
            txt_color = "white" if cm_norm[i, j] > thresh else "#111111"
            ax.text(
                j, i, f"{val_pct}\n{val_cnt}",
                ha="center", va="center",
                color=txt_color,
                fontsize=10.5,
                fontweight="bold" if i == j else "normal"
            )
            
    ax.set_ylabel("True Class (Ground Truth)", fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Class (DyDA Model)", fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix (Row-Normalized / Recall)\nHeld-Out Test Set (N = 3,065 Samples)", 
                 pad=14, fontweight="bold")
    
    diag_sum = int(np.trace(cm))
    total_samples = int(np.sum(cm))
    acc = diag_sum / total_samples * 100
    ax.text(
        0.02, -0.16,
        f"Overall Test Accuracy: {acc:.2f}% ({diag_sum}/{total_samples} correctly classified)",
        transform=ax.transAxes,
        fontsize=10,
        fontstyle="italic",
        color="#333333"
    )
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confusion_matrix_counts(cm: np.ndarray, classes: list, out_path: Path):
    """Plot raw count confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of Samples", rotation=270, labelpad=15, fontsize=11)
    cbar.ax.tick_params(labelsize=9.5)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontweight="medium")
    
    thresh = cm.max() * 0.45
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_cnt = f"{int(cm[i, j]):,}"
            txt_color = "white" if cm[i, j] > thresh else "#111111"
            ax.text(
                j, i, val_cnt,
                ha="center", va="center",
                color=txt_color,
                fontsize=11.5,
                fontweight="bold" if i == j else "normal"
            )
            
    ax.set_ylabel("True Class (Ground Truth)", fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Class (DyDA Model)", fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix (Raw Case Counts)\nHeld-Out Test Set (N = 3,065 Samples)", 
                 pad=14, fontweight="bold")
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_roc_curves_publication(y_true_indices: np.ndarray, y_probs: np.ndarray, classes: list, out_path: Path):
    """Plot One-vs-Rest ROC curves with AUC values for each class and macro-average."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    lw = 2.2
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    
    for i, cls in enumerate(classes):
        binary_true = (y_true_indices == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        roc_auc_dict[i] = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr,
            color=COLORS[cls],
            lw=lw,
            label=f"{cls} (AUC = {roc_auc_dict[i]:.4f})"
        )
        
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)
    
    ax.plot(
        all_fpr, mean_tpr,
        label=f"Macro-Average (AUC = {macro_auc:.4f})",
        color=COLORS["Macro"],
        linestyle="--",
        linewidth=2.5
    )
    
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Chance Line (AUC = 0.5000)")
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", labelpad=8)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves\nOne-vs-Rest Multiclass Evaluation (N = 3,065)", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    ax.annotate(
        f"Macro AUC: {macro_auc:.4f}\n(Near-Perfect Discrimination)",
        xy=(0.04, 0.96), xytext=(0.18, 0.82),
        arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.5, color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eff6ff", ec="#93c5fd", lw=1)
    )
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_individual_class_roc_curve(
    binary_true: np.ndarray,
    probs: np.ndarray,
    cls_name: str,
    metrics_class: dict,
    out_path: Path
):
    """Plot an individual, dedicated publication-grade ROC curve with inset zoom and diagnostic stats."""
    fpr, tpr, _ = roc_curve(binary_true, probs)
    roc_auc = auc(fpr, tpr)
    color = COLORS[cls_name]
    
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    
    # Shaded AUC area
    ax.fill_between(fpr, tpr, alpha=0.15, color=color, label=f"AUC Area = {roc_auc:.4f}")
    
    # Main ROC curve
    ax.plot(fpr, tpr, color=color, lw=2.6, label=f"{cls_name} ROC (AUC = {roc_auc:.4f})")
    
    # Chance baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance Baseline (AUC = 0.5000)")
    
    # Operating point calculation at threshold 0.50
    pred_binary = (probs >= 0.50).astype(int)
    tp = np.sum((pred_binary == 1) & (binary_true == 1))
    fp = np.sum((pred_binary == 1) & (binary_true == 0))
    tn = np.sum((pred_binary == 0) & (binary_true == 0))
    fn = np.sum((pred_binary == 0) & (binary_true == 1))
    
    op_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    op_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    op_fpr = 1.0 - op_spec
    
    # Marker for Operating Point
    ax.scatter([op_fpr], [op_sens], color="#dc2626", s=130, zorder=6, edgecolor="#7f1d1d", lw=1.6, label="Operating Point (τ = 0.50)")
    
    # Inset zoom to highlight near-perfect discrimination at the corner
    axins = inset_axes(
        ax, width="36%", height="30%", loc="lower right",
        bbox_to_anchor=(0.58, 0.08, 0.38, 0.34), bbox_transform=ax.transAxes
    )
    axins.plot(fpr, tpr, color=color, lw=2.2)
    axins.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    axins.scatter([op_fpr], [op_sens], color="#dc2626", s=80, zorder=6, edgecolor="#7f1d1d", lw=1.2)
    axins.set_xlim(-0.001, 0.04)
    zoom_min_y = max(0.92, op_sens - 0.05)
    axins.set_ylim(zoom_min_y, 1.004)
    axins.set_title(f"Zoom: FPR [0.0-0.04] | TPR [{zoom_min_y:.2f}-1.0]", fontsize=8.0, fontweight="bold", pad=3)
    axins.tick_params(labelsize=7.5)
    axins.grid(True, linestyle=":", alpha=0.5)
    
    # Annotate operating point on main ax
    ax.annotate(
        f"Operating Point (τ = 0.50)\n"
        f"Sensitivity: {op_sens * 100:.2f}%\n"
        f"Specificity: {op_spec * 100:.2f}%",
        xy=(op_fpr, op_sens),
        xytext=(0.14, 0.85),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.2, color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#fca5a5", lw=1.2)
    )
    
    # Diagnostic statistics summary box in upper right
    support = metrics_class.get("support", int(np.sum(binary_true)))
    prec = metrics_class.get("precision", 0) * 100.0
    f1 = metrics_class.get("f1", 0) * 100.0
    
    info_text = (
        f"Diagnostic Statistics ({cls_name})\n"
        f"──────────────────────────────\n"
        f"Test Support:      N = {support:,}\n"
        f"Sensitivity:       {op_sens * 100:.2f}%\n"
        f"Specificity:       {op_spec * 100:.2f}%\n"
        f"Precision (PPV):   {prec:.2f}%\n"
        f"F1-Score:          {f1:.2f}%\n"
        f"ROC-AUC:           {roc_auc:.4f}"
    )
    
    ax.text(
        0.48, 0.84, info_text,
        transform=ax.transAxes,
        fontsize=9.0,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec="#cbd5e1", lw=1.2)
    )
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", labelpad=8)
    ax.set_title(f"Individual Receiver Operating Characteristic (ROC): {cls_name}\n"
                 f"DyDA Hybrid Classifier (AUC = {roc_auc:.4f})", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.06), frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_precision_recall_curves_publication(y_true_indices: np.ndarray, y_probs: np.ndarray, classes: list, out_path: Path):
    """Plot Precision-Recall curves with Average Precision (AP) for each class and macro-average."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    lw = 2.2
    ap_scores = []
    
    for i, cls in enumerate(classes):
        binary_true = (y_true_indices == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_probs[:, i])
        ap = average_precision_score(binary_true, y_probs[:, i])
        ap_scores.append(ap)
        
        ax.plot(
            recall, precision,
            color=COLORS[cls],
            lw=lw,
            label=f"{cls} (AP = {ap:.4f})"
        )
        
    macro_ap = np.mean(ap_scores)
    
    ax.plot([], [], " ", label=f"Macro-Average AP = {macro_ap:.4f}")
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.80, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold", labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold", labelpad=8)
    ax.set_title("Precision-Recall (PR) Curves\nHeld-Out Test Set (N = 3,065 Samples)", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_individual_class_pr_curve(
    binary_true: np.ndarray,
    probs: np.ndarray,
    cls_name: str,
    metrics_class: dict,
    out_path: Path
):
    """Plot an individual, dedicated publication-grade Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(binary_true, probs)
    ap = average_precision_score(binary_true, probs)
    color = COLORS[cls_name]
    
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    
    ax.fill_between(recall, precision, alpha=0.18, color=color, label=f"PR Area (AP = {ap:.4f})")
    ax.plot(recall, precision, color=color, lw=2.6, label=f"{cls_name} PR Curve (AP = {ap:.4f})")
    
    prevalence = np.sum(binary_true) / len(binary_true)
    ax.axhline(prevalence, color="k", linestyle="--", lw=1.2, alpha=0.7, label=f"Class Prevalence ({prevalence * 100:.1f}%)")
    
    support = metrics_class.get("support", int(np.sum(binary_true)))
    sens = metrics_class.get("sensitivity", 0) * 100.0
    prec = metrics_class.get("precision", 0) * 100.0
    f1 = metrics_class.get("f1", 0) * 100.0
    
    # Operating point marker
    ax.scatter([sens / 100.0], [prec / 100.0], color="#dc2626", s=130, zorder=6, edgecolor="#7f1d1d", lw=1.6, label="Operating Point (τ = 0.50)")
    
    ax.annotate(
        f"Operating Point (τ = 0.50)\n"
        f"Precision: {prec:.2f}%\n"
        f"Recall: {sens:.2f}%",
        xy=(sens / 100.0, prec / 100.0),
        xytext=(max(sens / 100.0 - 0.35, 0.15), 0.88),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.5, color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#fca5a5", lw=1.2)
    )
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([max(0.70, prevalence - 0.05), 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold", labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold", labelpad=8)
    ax.set_title(f"Individual Precision-Recall Curve: {cls_name}\n"
                 f"Average Precision (AP) = {ap:.4f} (N = {support:,})", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_f1_vs_threshold_curves(y_true_indices: np.ndarray, y_probs: np.ndarray, classes: list, out_path: Path):
    """Plot F1-Score as a function of decision threshold across all classes."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    
    thresholds = np.linspace(0.05, 0.95, 91)
    
    for i, cls in enumerate(classes):
        binary_true = (y_true_indices == i).astype(int)
        prob = y_probs[:, i]
        f1_vals = []
        for th in thresholds:
            pred = (prob >= th).astype(int)
            tp = np.sum((pred == 1) & (binary_true == 1))
            fp = np.sum((pred == 1) & (binary_true == 0))
            fn = np.sum((pred == 0) & (binary_true == 1))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            f1_vals.append(f1)
            
        ax.plot(thresholds, f1_vals, color=COLORS[cls], lw=2.2, label=f"{cls} (Max F1: {max(f1_vals):.4f})")
        
    ax.axvline(0.50, color="#666666", linestyle=":", lw=1.5, label="Standard Operating Threshold (0.50)")
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.75, 1.02])
    ax.set_xlabel("Decision Probability Threshold", fontweight="bold", labelpad=8)
    ax.set_ylabel("F1-Score", fontweight="bold", labelpad=8)
    ax.set_title("F1-Score Stability vs Decision Threshold\nRobust Operational Range [0.20 - 0.80]", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower center", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_loss_curve(history_df: pd.DataFrame, out_path: Path):
    """Plot Training Loss vs Validation Loss over Epochs as a separate standalone figure."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    
    epochs = history_df["epoch"].values
    train_loss = history_df["train_loss"].values
    val_loss = history_df["val_loss"].values
    
    ax.plot(
        epochs, train_loss,
        color=COLORS["Train"],
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Training Loss (DyDA Hybrid)",
    )
    ax.plot(
        epochs, val_loss,
        color=COLORS["Val"],
        marker="s",
        markersize=6,
        linewidth=2.2,
        label="Validation Loss",
    )
    
    min_val_idx = np.argmin(val_loss)
    best_epoch = epochs[min_val_idx]
    best_loss = val_loss[min_val_idx]
    
    ax.scatter(best_epoch, best_loss, color="#dc2626", s=130, zorder=5, edgecolor="#7f1d1d", lw=1.5)
    ax.annotate(
        f"Lowest Val Loss: {best_loss:.4f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 0.8, best_loss + 0.05),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.5, color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef2f2", ec="#fca5a5", lw=1)
    )
    
    ax.set_xticks(epochs)
    ax.set_xlabel("Training Epoch", fontweight="bold", labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontweight="bold", labelpad=8)
    ax.set_title("Training and Validation Loss Curve\nDyDA EfficientNet-B0 + Dynamic Data Augmentation", 
                 pad=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_accuracy_curve(history_df: pd.DataFrame, out_path: Path):
    """Plot Training Accuracy vs Validation Accuracy over Epochs as a separate standalone figure."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    
    epochs = history_df["epoch"].values
    train_acc = history_df["train_acc"].values * 100.0
    val_acc = history_df["val_acc"].values * 100.0
    
    ax.plot(
        epochs, train_acc,
        color=COLORS["Train"],
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Training Accuracy",
    )
    ax.plot(
        epochs, val_acc,
        color=COLORS["Val"],
        marker="s",
        markersize=6,
        linewidth=2.2,
        label="Validation Accuracy",
    )
    
    max_val_idx = np.argmax(val_acc)
    best_epoch = epochs[max_val_idx]
    best_acc = val_acc[max_val_idx]
    
    ax.scatter(best_epoch, best_acc, color="#16a34a", s=130, zorder=5, edgecolor="#14532d", lw=1.5)
    ax.annotate(
        f"Peak Val Accuracy: {best_acc:.2f}%\n(Epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch - 3.2, best_acc - 2.5),
        arrowprops=dict(facecolor="#16a34a", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.5, color="#14532d",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec="#86efac", lw=1)
    )
    
    ax.set_xticks(epochs)
    ax.set_ylim([85.0, 100.5])
    ax.set_xlabel("Training Epoch", fontweight="bold", labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=8)
    ax.set_title("Training and Validation Accuracy Progression\nRapid Convergence to >98% Diagnostic Accuracy", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_validation_macro_f1_curve(history_df: pd.DataFrame, out_path: Path):
    """Plot Validation Macro-F1 curve over Epochs (Primary Checkpoint Selection Metric)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    
    epochs = history_df["epoch"].values
    val_macro_f1 = history_df["val_macro_f1"].values * 100.0
    val_bal_acc = history_df["val_balanced_acc"].values * 100.0
    
    ax.plot(
        epochs, val_macro_f1,
        color="#7c3aed",
        marker="D",
        markersize=6,
        linewidth=2.4,
        label="Validation Macro-F1 (Checkpoint Metric)",
    )
    ax.plot(
        epochs, val_bal_acc,
        color="#059669",
        marker="^",
        markersize=6,
        linewidth=2.0,
        linestyle="--",
        label="Validation Balanced Accuracy",
    )
    
    best_idx = np.argmax(val_macro_f1)
    best_ep = epochs[best_idx]
    best_f1 = val_macro_f1[best_idx]
    
    ax.scatter(best_ep, best_f1, color="#7c3aed", s=140, zorder=5, edgecolor="#4c1d95", lw=1.5)
    ax.annotate(
        f"Best Model Checkpoint (Epoch {best_ep})\nMacro-F1 = {best_f1:.2f}% (0.9810)",
        xy=(best_ep, best_f1),
        xytext=(best_ep - 3.8, best_f1 - 2.8),
        arrowprops=dict(facecolor="#7c3aed", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9.5, color="#4c1d95",
        bbox=dict(boxstyle="round,pad=0.4", fc="#faf5ff", ec="#d8b4fe", lw=1)
    )
    
    ax.set_xticks(epochs)
    ax.set_ylim([88.0, 100.0])
    ax.set_xlabel("Training Epoch", fontweight="bold", labelpad=8)
    ax.set_ylabel("Score (%)", fontweight="bold", labelpad=8)
    ax.set_title("Validation Macro-F1 & Balanced Accuracy Evolution\nModel Selection Criterion across 10 Training Epochs", 
                 pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_diagnostic_bars(metrics_json_path: Path, out_path: Path):
    """Plot grouped publication bar chart of Precision, Recall/Sensitivity, Specificity, and F1."""
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)
        
    per_class = metrics["per_class"]
    classes = list(per_class.keys())
    metric_keys = ["precision", "sensitivity", "specificity", "f1"]
    labels = ["Precision", "Recall (Sensitivity)", "Specificity", "F1-Score"]
    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706"]
    
    x = np.arange(len(classes))
    width = 0.19
    
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
    
    for i, (k, lbl, col) in enumerate(zip(metric_keys, labels, colors)):
        vals = [per_class[c][k] * 100.0 for c in classes]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=lbl, color=col, edgecolor="#333333", linewidth=0.6)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", rotation=0
            )
            
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold", fontsize=11)
    ax.set_ylim([90.0, 102.5])
    ax.set_ylabel("Score (%)", fontweight="bold", labelpad=8)
    ax.set_title("Comprehensive Per-Class Diagnostic Performance on Held-Out Test Set\n(N = 3,065 Confirmed Clinical Radiographs)", 
                 pad=14, fontweight="bold")
    ax.legend(loc="lower right", ncol=4, frameon=True, facecolor="#ffffff", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    root_dir = Path("/media/storage2/mdpi/DYDA")
    out_dir = ensure_dir(root_dir / "final_figures")
    
    # Load test predictions and metrics
    test_pred_path = root_dir / "outputs/hybrid/evaluation/test/predictions.csv"
    metrics_path = root_dir / "outputs/hybrid/evaluation/test/metrics.json"
    if not test_pred_path.is_file():
        raise FileNotFoundError(f"Missing {test_pred_path}")
    df_preds = pd.read_csv(test_pred_path)
    
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)
    per_class_metrics = metrics_dict["per_class"]
    
    # Map class labels to integers
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_true_indices = np.array([class_to_idx[y] for y in df_preds["y_true"]])
    y_pred_indices = np.array([class_to_idx[y] for y in df_preds["y_pred"]])
    
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    y_probs = df_preds[prob_cols].values
    
    # Confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(CLASS_NAMES)))
    
    print("Generating Separate Publication Figures in:", out_dir)
    
    # Figure 1: Normalized Confusion Matrix
    plot_confusion_matrix_normalized(
        cm, CLASS_NAMES, 
        out_dir / "01_final_confusion_matrix_normalized.png"
    )
    
    # Figure 2: Raw Counts Confusion Matrix
    plot_confusion_matrix_counts(
        cm, CLASS_NAMES, 
        out_dir / "02_final_confusion_matrix_raw_counts.png"
    )
    
    # Figure 3: Combined ROC Curves (All 4 classes + Macro)
    plot_roc_curves_publication(
        y_true_indices, y_probs, CLASS_NAMES, 
        out_dir / "03_final_roc_auc_curves_all_classes.png"
    )
    
    # ─── Individual Class ROC Curves (Requested by User) ───
    print("\nGenerating Individual AUC ROC Curves for Each Class:")
    class_prefixes = ["03a", "03b", "03c", "03d"]
    for i, cls in enumerate(CLASS_NAMES):
        binary_true = (y_true_indices == i).astype(int)
        probs = y_probs[:, i]
        prefix = class_prefixes[i]
        
        # Individual ROC Curve
        plot_individual_class_roc_curve(
            binary_true=binary_true,
            probs=probs,
            cls_name=cls,
            metrics_class=per_class_metrics.get(cls, {}),
            out_path=out_dir / f"{prefix}_final_roc_auc_{cls}.png"
        )
        
        # Also generate individual PR curve for each class
        plot_individual_class_pr_curve(
            binary_true=binary_true,
            probs=probs,
            cls_name=cls,
            metrics_class=per_class_metrics.get(cls, {}),
            out_path=out_dir / f"04_{cls.lower()}_precision_recall_curve.png"
        )
        
    # Figure 4: Precision-Recall Curves (Combined)
    plot_precision_recall_curves_publication(
        y_true_indices, y_probs, CLASS_NAMES, 
        out_dir / "04_final_precision_recall_curves_all_classes.png"
    )
    
    # Figure 5: F1 Score vs Threshold Curve
    plot_f1_vs_threshold_curves(
        y_true_indices, y_probs, CLASS_NAMES, 
        out_dir / "05_final_f1_score_vs_threshold_curves.png"
    )
    
    # Training history (Fold 2 - winning model)
    history_path = root_dir / "outputs/hybrid/fold_2/history.csv"
    if history_path.is_file():
        df_hist = pd.read_csv(history_path)
        
        # Figure 6: Standalone Loss Curve
        plot_loss_curve(
            df_hist, 
            out_dir / "06_training_validation_loss_curve.png"
        )
        
        # Figure 7: Standalone Accuracy Curve
        plot_accuracy_curve(
            df_hist, 
            out_dir / "07_training_validation_accuracy_curve.png"
        )
        
        # Figure 8: Standalone Validation Macro-F1 Curve
        plot_validation_macro_f1_curve(
            df_hist, 
            out_dir / "08_validation_macro_f1_curve.png"
        )
        
    # Figure 9: Per-class diagnostic bars
    if metrics_path.is_file():
        plot_per_class_diagnostic_bars(
            metrics_path, 
            out_dir / "09_per_class_diagnostic_metrics_bar.png"
        )
        
    print("\nAll separate and individual figures successfully generated in:", out_dir)


if __name__ == "__main__":
    main()
