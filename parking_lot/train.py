"""Train the parking space occupancy classifier.

Example
-------
python -m parking_lot.train --data "/path/to/Parking Lot.v10i.multiclass" \
    --out runs/parking-v1 --epochs 12 --model mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import (
    MEAN,
    STD,
    build_dataloader,
    build_transforms,
    class_weights,
    load_dataset,
)
from .metrics import classification_report, confusion_matrix, format_report, top_confusions
from .model import BACKBONES, DEFAULT_BACKBONE, build_model, count_parameters, resolve_device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a parking space state classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Dataset root (Roboflow export folder).")
    parser.add_argument("--out", default="runs/parking", help="Output directory for checkpoints.")
    parser.add_argument("--model", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation share when the export has no valid/ folder.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many epochs without val macro-F1 improvement (0 = off).")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override the classifier dropout probability.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps ...")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch.")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency loss weighting.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--no-hflip", action="store_true", help="Disable horizontal flip augmentation.")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, true_labels, predictions = 0.0, [], []
    amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        true_labels.extend(targets.cpu().tolist())

    return {
        "loss": total_loss / max(1, len(true_labels)),
        "matrix": confusion_matrix(true_labels, predictions, num_classes),
        "targets": true_labels,
        "predictions": predictions,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    bundle = load_dataset(args.data, val_fraction=args.val_fraction, seed=args.seed)
    print(bundle.summary())
    if not bundle.val:
        raise SystemExit("Validation split is empty; lower --val-fraction or add a valid/ folder.")

    num_classes = len(bundle.classes)
    train_loader = build_dataloader(
        bundle.train, build_transforms(args.img_size, train=True, hflip=not args.no_hflip),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, device=device,
    )
    val_loader = build_dataloader(
        bundle.val, build_transforms(args.img_size, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, device=device,
    )

    model = build_model(
        args.model, num_classes=num_classes,
        pretrained=not args.no_pretrained, dropout=args.dropout,
    ).to(device)
    print(
        f"Model: {args.model} ({count_parameters(model):,} trainable params) on {device.type}"
    )

    if args.no_class_weights:
        weights = None
    else:
        weights = class_weights(bundle.train, num_classes).to(device)
        print("Class weights: " + ", ".join(
            f"{name}={w:.2f}" for name, w in zip(bundle.classes, weights.tolist())
        ))

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "seconds"]
        )

    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running_loss, seen, correct = 0.0, 0, 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

        val = evaluate_loader(model, val_loader, criterion, device, num_classes)
        val_report = classification_report(val["matrix"], bundle.classes)
        val.update(val_report)
        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - started
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val["loss"], 5),
            "val_acc": round(val["accuracy"], 5),
            "val_macro_f1": round(val["macro_f1"], 5),
            "lr": lr,
            "seconds": round(elapsed, 2),
        }
        history.append(record)
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(list(record.values()))

        print(
            f"epoch {epoch:>3}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f}"
            f" | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} f1 {val['macro_f1']:.4f}"
            f" | lr {lr:.2e} | {elapsed:.1f}s"
        )

        checkpoint = {
            "model_name": args.model,
            "num_classes": num_classes,
            "classes": bundle.classes,
            "img_size": args.img_size,
            "mean": list(MEAN),
            "std": list(STD),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_macro_f1": float(val["macro_f1"]),
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, stale_epochs = float(val["macro_f1"]), epoch, 0
            torch.save(checkpoint, out_dir / "best.pt")
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"No improvement for {stale_epochs} epochs — stopping early.")
                break

    print(f"\nBest validation macro-F1: {best_f1:.4f} (epoch {best_epoch})")

    # Reload the best weights and report on validation once more.
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    final_val = evaluate_loader(model, val_loader, criterion, device, num_classes)
    final_report = classification_report(final_val["matrix"], bundle.classes)
    print(format_report(final_report, top_confusions(final_val["matrix"], bundle.classes)))

    summary = {
        "classes": bundle.classes,
        "model": args.model,
        "img_size": args.img_size,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "val": final_report,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved checkpoint: {out_dir / 'best.pt'}")
    print(f"Saved metrics:    {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
