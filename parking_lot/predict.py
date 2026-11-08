"""Classify individual parking space crops.

python -m parking_lot.predict --checkpoint runs/parking/best.pt some_crop.jpg
python -m parking_lot.predict --checkpoint runs/parking/best.pt --topk 2 crops/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import IMAGE_EXTENSIONS, load_rgb
from .inference import Predictor


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the state of parking space crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Image files and/or directories.")
    parser.add_argument("--checkpoint", default="runs/parking/best.pt")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--json", default=None, help="Also write predictions to this JSON file.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def collect_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            images.extend(
                sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return images


def main(argv=None) -> int:
    args = parse_args(argv)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    images = collect_images(args.paths)
    if not images:
        raise SystemExit("No images found.")

    results = []
    width = max(len(str(p)) for p in images)
    for path in images:
        prediction = predictor.predict(load_rgb(path), topk=args.topk)
        row = {"path": str(path), "predictions": prediction}
        results.append(row)
        detail = "  ".join(f"{p['label']} {p['confidence']:.3f}" for p in prediction)
        print(f"{str(path):<{width}}  {detail}")

    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
