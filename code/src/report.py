"""Plotting: confusion matrix heatmap, ROC/PR curves, training curves, qualitative overlays."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Background", "Foreground"])
    ax.set_yticklabels(["Background", "Foreground"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)", ha="center", va="center",
                     color="white" if cm[i, j] > total / 4 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(roc, pr, out_path, title):
    fpr, tpr, auc = roc
    precision, recall, ap = pr
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC — {title}")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"AP = {ap:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — {title}")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, out_path, title):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE+Dice Loss")
    axes[0].set_title(f"Loss — {title}")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_dice"] for h in history], label="val Dice", color="green")
    axes[1].plot(epochs, [h["val_iou"] for h in history], label="val IoU", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Validation metrics — {title}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qualitative(qualitative, out_path, title, n=4):
    images, masks, preds = qualitative
    n = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray")
        axes[i, 1].set_title("Ground truth" if i == 0 else "")
        axes[i, 2].imshow(preds[i, 0].numpy() > 0.5, cmap="gray")
        axes[i, 2].set_title("Prediction" if i == 0 else "")
        for j in range(3):
            axes[i, j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all(result, out_dir, dataset_name, variant):
    os.makedirs(out_dir, exist_ok=True)
    label = f"{dataset_name} / {variant}"
    plot_confusion_matrix(result["confusion_matrix"], os.path.join(out_dir, "confusion_matrix.png"), label)
    plot_roc_pr(result["roc"], result["pr"], os.path.join(out_dir, "roc_pr_curves.png"), label)
    plot_training_curves(result["history"], os.path.join(out_dir, "training_curves.png"), label)
    plot_qualitative(result["qualitative"], os.path.join(out_dir, "qualitative_samples.png"), label)
