"""Confusion matrix, per-class metrics and reporting (no sklearn needed)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def classification_report(
    matrix: np.ndarray, class_names: Sequence[str]
) -> dict:
    """Precision / recall / F1 per class plus macro and weighted averages."""
    matrix = np.asarray(matrix, dtype=np.float64)
    num_classes = matrix.shape[0]
    per_class = []
    for index in range(num_classes):
        tp = matrix[index, index]
        support = matrix[index, :].sum()
        predicted = matrix[:, index].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            {
                "class": class_names[index],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    macro_f1 = float(np.mean([c["f1"] for c in per_class])) if per_class else 0.0
    weighted_f1 = (
        float(sum(c["f1"] * c["support"] for c in per_class) / total) if total else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": int(total),
        "per_class": per_class,
    }


def format_report(report: dict, confusions: list[tuple[str, str, int]] | None = None) -> str:
    lines = [
        f"{'class':<24} {'precision':>9} {'recall':>8} {'f1':>8} {'support':>8}",
        "-" * 61,
    ]
    for row in report["per_class"]:
        lines.append(
            f"{row['class'][:24]:<24} {row['precision']:>9.3f} {row['recall']:>8.3f} "
            f"{row['f1']:>8.3f} {row['support']:>8d}"
        )
    lines.append("-" * 61)
    lines.append(
        f"{'accuracy':<24} {report['accuracy']:>9.3f} "
        f"{'macro f1':>17} {report['macro_f1']:>8.3f} {report['num_samples']:>8d}"
    )
    if confusions:
        lines.append("")
        lines.append("Most frequent mistakes:")
        for true_name, pred_name, count in confusions:
            lines.append(f"  {true_name} -> {pred_name}: {count}")
    return "\n".join(lines)


def top_confusions(
    matrix: np.ndarray, class_names: Sequence[str], limit: int = 5
) -> list[tuple[str, str, int]]:
    pairs = []
    for true in range(matrix.shape[0]):
        for pred in range(matrix.shape[1]):
            if true != pred and matrix[true, pred] > 0:
                pairs.append((int(matrix[true, pred]), true, pred))
    pairs.sort(reverse=True)
    return [(class_names[t], class_names[p], count) for count, t, p in pairs[:limit]]
