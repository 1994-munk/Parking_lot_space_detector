"""Checkpoint loading and batched inference shared by the CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torchvision import transforms

from .data import MEAN, STD
from .model import build_model, resolve_device


class Predictor:
    """A trained parking space classifier ready for images."""

    def __init__(self, model, classes: Sequence[str], img_size: int, device: torch.device,
                 mean: Sequence[float] = MEAN, std: Sequence[float] = STD) -> None:
        self.model = model.eval()
        self.classes = list(classes)
        self.img_size = img_size
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(round(img_size * 1.15))),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(tuple(mean), tuple(std)),
            ]
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Predictor":
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved = resolve_device(device)
        payload = torch.load(path, map_location=resolved, weights_only=False)
        model = build_model(
            payload["model_name"],
            num_classes=payload["num_classes"],
            pretrained=False,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(resolved)
        return cls(
            model=model,
            classes=payload["classes"],
            img_size=payload.get("img_size", 224),
            device=resolved,
            mean=payload.get("mean", MEAN),
            std=payload.get("std", STD),
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def predict_proba(self, images: Iterable[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Return an (n, num_classes) tensor of probabilities."""
        images = list(images)
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [self.preprocess(image) for image in images[start:start + batch_size]]
            ).to(self.device)
            logits = self.model(batch)
            outputs.append(torch.softmax(logits, dim=1).cpu())
        if not outputs:
            return torch.zeros((0, len(self.classes)))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 1) -> list[dict]:
        probabilities = self.predict_proba([image])[0]
        topk = max(1, min(topk, len(self.classes)))
        values, indices = probabilities.topk(topk)
        return [
            {
                "label": self.classes[int(index)],
                "label_id": int(index),
                "confidence": float(value),
            }
            for value, index in zip(values, indices)
        ]
