"""Score every parking space of a lot and draw an occupancy overlay.

The classifier works on cropped spaces, so a lot image is described by a JSON
file mapping slot ids to boxes::

    {
      "image": "lot.jpg",
      "normalized": false,
      "slots": [
        {"id": "A1", "box": [120, 240, 210, 330]},
        {"id": "A2", "box": [220, 240, 310, 330]}
      ]
    }

Boxes may also be normalised to 0..1 (`"normalized": true`). The original lot
image can be overridden with --image.

python -m parking_lot.slots --checkpoint runs/parking/best.pt \
    --slots demo/slots.json --out demo/annotated.jpg
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data import load_rgb
from .inference import Predictor

EMPTY_HINTS = ("empty", "vacant", "free", "available", "unoccupied", "no_car", "nocar", "no car", "open")
FREE_COLOR = (34, 197, 94)
OCCUPIED_COLOR = (239, 68, 68)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify parking slots defined in a JSON file and draw the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slots", required=True, help="JSON file with slot boxes.")
    parser.add_argument("--checkpoint", default="runs/parking/best.pt")
    parser.add_argument("--image", default=None, help="Override the image path stored in the JSON.")
    parser.add_argument("--out", default=None, help="Annotated image path.")
    parser.add_argument("--json", default=None, help="Per-slot results as JSON.")
    parser.add_argument("--empty-classes", default=None,
                        help="Comma separated class names that mean 'free'. Default: auto-detect.")
    parser.add_argument("--crop-pad", type=float, default=0.02,
                        help="Expand each box by this fraction of its size before cropping.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_slots(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict) or "slots" not in payload:
        raise ValueError(f"{path} must contain a 'slots' list.")
    return payload, list(payload["slots"])


def resolve_image_path(explicit: str | None, payload: dict, slots_path: Path) -> Path:
    """Resolve --image, or the JSON 'image' key relative to the JSON's folder."""
    raw = explicit or payload.get("image") or payload.get("image_path") or ""
    if not raw:
        raise SystemExit("No lot image given: use --image or an 'image' key in the JSON.")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    beside_json = slots_path.parent / candidate
    if beside_json.is_file() or not candidate.is_file():
        return beside_json
    return candidate


def normalize_slots(slots: list[dict], width: int, height: int, normalized: bool) -> list[dict]:
    boxes = []
    for index, slot in enumerate(slots):
        raw = slot.get("box") if isinstance(slot, dict) else slot
        if not raw or len(raw) != 4:
            raise ValueError(f"Slot {index} needs a 4-value 'box' [x1, y1, x2, y2].")
        x1, y1, x2, y2 = (float(v) for v in raw)
        if normalized or max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            raise ValueError(f"Slot {index} box is too small after clipping: {raw}")
        slot_id = str(slot.get("id", slot.get("name", index + 1))) if isinstance(slot, dict) else str(index + 1)
        boxes.append({"id": slot_id, "box": [x1, y1, x2, y2]})
    return boxes


def resolve_empty_classes(classes: list[str], explicit: str | None) -> list[str]:
    if explicit:
        chosen = [name.strip() for name in explicit.split(",") if name.strip()]
        missing = [name for name in chosen if name not in classes]
        if missing:
            raise ValueError(f"--empty-classes not in checkpoint classes {classes}: {missing}")
        return chosen
    detected = [name for name in classes if any(hint in name.lower() for hint in EMPTY_HINTS)]
    if not detected:
        raise ValueError(
            f"Could not tell which class means 'free' from {classes}. "
            "Pass --empty-classes (e.g. --empty-classes empty)."
        )
    return detected


def crop(image: Image.Image, box, pad: float) -> Image.Image:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return image.crop(
        (
            max(0, int(round(x1 - dx))),
            max(0, int(round(y1 - dy))),
            min(image.width, int(round(x2 + dx))),
            min(image.height, int(round(y2 + dy))),
        )
    )


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    """Trim a caption with an ellipsis so it stays inside its slot box."""
    if max_width <= 0:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip() + "\u2026"
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    return ""


def annotate(image: Image.Image, slots: list[dict], occupancy: float, counts: dict) -> Image.Image:
    rgb = image.convert("RGB")
    scale = max(1.0, rgb.width / 1200)
    font = _font(max(12, int(round(16 * scale))))
    stroke = max(2, int(round(3 * scale)))
    pad = max(6, int(round(8 * scale)))

    headline = (
        f"Occupancy {occupancy * 100:.1f}%  |  free {counts.get('free', 0)}"
        f"  |  occupied {counts.get('occupied', 0)}"
    )
    head_box = font.getbbox(headline)
    head_h = head_box[3] - head_box[1]
    banner = head_h + 2 * pad

    # Draw on a canvas with a banner above the photo so slot labels never collide
    # with the summary; all slot drawing is shifted down by the banner height.
    canvas = Image.new("RGB", (rgb.width, rgb.height + banner), (15, 23, 42))
    canvas.paste(rgb, (0, banner))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((pad, pad - head_box[1]), headline, fill=(248, 250, 252, 255), font=font)

    for slot in slots:
        color = FREE_COLOR if slot["is_empty"] else OCCUPIED_COLOR
        x1, y1, x2, y2 = slot["box"]
        draw.rectangle((x1, y1 + banner, x2, y2 + banner), outline=(*color, 255), width=stroke)
        caption = _fit_text(
            draw,
            f"{slot['id']} {slot['label']} {slot['confidence']:.2f}",
            font,
            (x2 - x1) - 2 * pad,
        )
        if not caption:
            continue
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
        top = max(banner, y1 + banner - text_h - pad)
        draw.rectangle(
            (x1, top, min(x2, x1 + text_w + pad), top + text_h + pad // 2), fill=(*color, 235)
        )
        draw.text((x1 + pad // 2, top + 1), caption, fill=(255, 255, 255, 255), font=font)
    return canvas


def main(argv=None) -> int:
    args = parse_args(argv)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    empty_classes = resolve_empty_classes(predictor.classes, args.empty_classes)

    slots_path = Path(args.slots).expanduser()
    payload, raw_slots = load_slots(slots_path)
    image_path = resolve_image_path(args.image, payload, slots_path)
    if not image_path.is_file():
        raise SystemExit(f"Lot image not found: {image_path} (use --image or an 'image' key in the JSON).")

    image = load_rgb(image_path)
    normalized = bool(payload.get("normalized", False))
    slots = normalize_slots(raw_slots, image.width, image.height, normalized)
    crops = [crop(image, slot["box"], args.crop_pad) for slot in slots]

    probabilities = predictor.predict_proba(crops)
    for slot, distribution in zip(slots, probabilities):
        confidence, index = distribution.max(dim=0)
        label = predictor.classes[int(index)]
        slot.update(
            label=label,
            label_id=int(index),
            confidence=float(confidence),
            is_empty=label in empty_classes,
        )

    free = sum(1 for slot in slots if slot["is_empty"])
    occupied = len(slots) - free
    occupancy = occupied / len(slots) if slots else 0.0
    counts_by_label = Counter(slot["label"] for slot in slots)
    counts = {"free": free, "occupied": occupied}

    annotated = annotate(image, slots, occupancy, counts)
    out_path = Path(args.out).expanduser() if args.out else slots_path.parent / f"{image_path.stem}_annotated.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, quality=92)

    results = {
        "image": str(image_path),
        "checkpoint": str(args.checkpoint),
        "empty_classes": empty_classes,
        "summary": {
            "total": len(slots),
            "free": free,
            "occupied": occupied,
            "occupancy": round(occupancy, 4),
            "counts_by_label": dict(counts_by_label),
        },
        "slots": slots,
        "annotated_image": str(out_path),
    }
    json_path = Path(args.json).expanduser() if args.json else out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Slots: {len(slots)}  free: {free}  occupied: {occupied}  occupancy: {occupancy * 100:.1f}%")
    print("By label: " + ", ".join(f"{name}={count}" for name, count in counts_by_label.most_common()))
    low = sorted((s for s in slots if s["confidence"] < 0.6), key=lambda s: s["confidence"])[:5]
    if low:
        print("Low confidence slots: " + ", ".join(f"{s['id']}={s['confidence']:.2f}" for s in low))
    print(f"Saved {out_path}")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
