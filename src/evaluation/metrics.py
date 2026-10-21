"""Evaluation metrics and plots."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval of an accuracy measured on `total` samples."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_metrics(y_true, y_pred, classes: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    low, high = wilson_interval(correct, n)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
    )
    return {
        "n_samples": n,
        "n_errors": n - correct,
        "accuracy": correct / n,
        "accuracy_ci95": [low, high],
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
            for c in classes
        },
    }


def text_report(y_true, y_pred, classes: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(classes))), target_names=classes, digits=3, zero_division=0
    )


def plot_confusion_matrix(y_true, y_pred, classes: list[str], path: Path, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(errors: list[dict], path: Path, max_images: int = 12, cols: int = 4) -> bool:
    """Grid of misclassified images. Each error: {path, true, pred, conf}. Returns False if none."""
    if not errors:
        return False
    errors = errors[:max_images]
    rows = math.ceil(len(errors) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, e in zip(axes.ravel(), errors):
        with Image.open(e["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"true: {e['true']}\npred: {e['pred']} ({e['conf']:.0%})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
