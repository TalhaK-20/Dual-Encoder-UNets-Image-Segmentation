"""Pixel-level segmentation metrics, confusion matrix, and ROC/PR curves."""
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def batch_metrics(pred, target, threshold=0.5, eps=1e-8):
    """pred, target: float tensors in [0, 1], any shape. Returns a dict of python floats."""
    pred_bin = (pred > threshold).float()
    target = target.float()

    tp = torch.sum(pred_bin * target)
    fp = torch.sum(pred_bin * (1 - target))
    fn = torch.sum((1 - pred_bin) * target)
    tn = torch.sum((1 - pred_bin) * (1 - target))

    dice = (2.0 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "f1": f1.item(),
    }


class MetricAccumulator:
    """Averages per-image metrics and separately keeps flattened pixel arrays
    for a corpus-level confusion matrix / ROC / PR curve."""

    def __init__(self):
        self.per_image = []
        self._probs = []
        self._targets = []

    def update(self, pred_prob, target):
        """pred_prob, target: [B, 1, H, W] float tensors."""
        for b in range(pred_prob.shape[0]):
            self.per_image.append(batch_metrics(pred_prob[b], target[b]))
        # Subsample pixels for the corpus-level curves so memory stays bounded
        # on large datasets (e.g. PanNuke) without biasing the curve shape.
        p = pred_prob.detach().cpu().numpy().reshape(-1)
        t = target.detach().cpu().numpy().reshape(-1)
        if p.size > 200_000:
            idx = np.random.choice(p.size, 200_000, replace=False)
            p, t = p[idx], t[idx]
        self._probs.append(p)
        self._targets.append(t)

    def summary(self):
        keys = self.per_image[0].keys()
        agg = {f"{k}_mean": float(np.mean([m[k] for m in self.per_image])) for k in keys}
        agg.update({f"{k}_std": float(np.std([m[k] for m in self.per_image])) for k in keys})
        agg["n_images"] = len(self.per_image)
        return agg

    def confusion_matrix(self, threshold=0.5):
        probs = np.concatenate(self._probs)
        targets = np.concatenate(self._targets)
        preds = (probs > threshold).astype(int)
        cm = confusion_matrix(targets.astype(int), preds, labels=[0, 1])
        return cm  # [[TN, FP], [FN, TP]]

    def roc_curve(self):
        probs = np.concatenate(self._probs)
        targets = np.concatenate(self._targets)
        fpr, tpr, _ = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        return fpr, tpr, auc

    def pr_curve(self):
        probs = np.concatenate(self._probs)
        targets = np.concatenate(self._targets)
        precision, recall, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        return precision, recall, ap
