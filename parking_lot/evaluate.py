"""Evaluate a checkpoint on the held-out split and plot a confusion matrix.

python -m parking_lot.evaluate --data "/path/to/Parking Lot.v10i.multiclass" \
    --checkpoint runs/parking/best.pt --out runs/parking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .data import build_dataloader, build_transforms, load_dataset  # noqa: E402
from .inference import Predictor  # noqa: E402
from .metrics import classification_report, confusion_matrix, format_report, top_confusions  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained parking space classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Dataset root (Roboflow export folder).")
    parser.add_argument("--checkpoint", default="runs/parking/best.pt")
    parser.add_argument("--split", default="auto", choices=["auto", "test", "val", "train"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default=None, help="Output directory (default: checkpoint folder).")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def pick_split(bundle, requested: str) -> tuple[str, list]:
    if requested != "auto":
        samples = getattr(bundle, requested)
        if not samples:
            raise SystemExit(f"Split '{requested}' is empty.")
        return requested, samples
    for name in ("test", "val"):
        samples = getattr(bundle, name)
        if samples:
            return name, samples
    raise SystemExit("No test or validation samples available.")


@torch.no_grad()
def collect_predictions(predictor: Predictor, loader):
    true_labels, predictions = [], []
    for images, targets in loader:
        logits = predictor.model(images.to(predictor.device))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        true_labels.extend(targets.tolist())
    return true_labels, predictions


def plot_confusion_matrix(matrix: np.ndarray, class_names, path: Path, normalize: bool = True) -> None:
    display = matrix.astype(np.float64)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)
    size = max(4.0, 0.9 * len(class_names) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size), dpi=150)
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    threshold = display.max() * 0.6 if display.size else 0.5
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = display[row, col]
            text = f"{value:.2f}" if normalize else f"{matrix[row, col]:d}"
            ax.text(
                col, row, text, ha="center", va="center", fontsize=9,
                color="white" if value > threshold else "#1b1b1b",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    bundle = load_dataset(args.data)
    split, samples = pick_split(bundle, args.split)

    loader = build_dataloader(
        samples, build_transforms(predictor.img_size, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        device=predictor.device,
    )
    true_labels, predictions = collect_predictions(predictor, loader)
    matrix = confusion_matrix(true_labels, predictions, len(predictor.classes))
    report = classification_report(matrix, predictor.classes)

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).expanduser().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {split} ({len(samples)} images)")
    print(format_report(report, top_confusions(matrix, predictor.classes)))
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(" " * 16 + "  ".join(f"{name[:10]:>10}" for name in predictor.classes))
    for name, row in zip(predictor.classes, matrix):
        print(f"{name[:16]:<16}" + "  ".join(f"{value:>10d}" for value in row))

    plot_confusion_matrix(matrix, predictor.classes, out_dir / "confusion_matrix.png")
    (out_dir / f"report_{split}.json").write_text(
        json.dumps({"split": split, "checkpoint": str(args.checkpoint), "report": report,
                    "matrix": matrix.tolist(), "classes": predictor.classes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {out_dir / 'confusion_matrix.png'}")
    print(f"Saved {out_dir / f'report_{split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
