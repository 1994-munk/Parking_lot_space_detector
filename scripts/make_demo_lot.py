"""Build a synthetic lot image + slots.json from dataset crops.

The classification export has no whole-lot photos, so this tiles real space
crops onto an asphalt background. Handy to smoke-test `parking_lot.slots`
before you map the real slot coordinates of your camera view.

python scripts/make_demo_lot.py --data "/path/to/Parking Lot.v10i.multiclass" \
    --out demo --rows 3 --cols 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parking_lot.data import load_dataset, load_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="demo")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--tile", type=int, default=180, help="Slot tile size in pixels.")
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="Rotate/flip tiles for variety.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_dataset(args.data)
    samples = getattr(bundle, args.split) or bundle.train
    rng = random.Random(args.seed)
    picks = rng.choices(samples, k=args.rows * args.cols)

    width = args.cols * args.tile + (args.cols + 1) * args.gap
    height = args.rows * args.tile + (args.rows + 1) * args.gap + 34
    canvas = Image.new("RGB", (width, height), (58, 58, 61))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 30), fill=(28, 32, 40))

    slots = []
    for index, (path, _label) in enumerate(picks):
        row, col = divmod(index, args.cols)
        x = args.gap + col * (args.tile + args.gap)
        y = 34 + args.gap + row * (args.tile + args.gap)
        tile = load_rgb(path).resize((args.tile, args.tile), Image.BILINEAR)
        if args.augment:
            if rng.random() < 0.5:
                tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
            tile = tile.rotate(rng.uniform(-6, 6), expand=False, fillcolor=(58, 58, 61))
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + args.tile, y + args.tile), outline=(235, 235, 235), width=3)
        slots.append({"id": f"{chr(65 + row)}{col + 1}", "box": [x, y, x + args.tile, y + args.tile]})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "lot.jpg"
    canvas.save(image_path, quality=92)
    slots_path = out_dir / "slots.json"
    slots_path.write_text(
        json.dumps({"image": image_path.name, "normalized": False, "slots": slots}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {image_path} ({width}x{height}) and {slots_path} with {len(slots)} slots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
