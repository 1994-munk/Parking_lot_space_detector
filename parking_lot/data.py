"""Dataset discovery, splitting, transforms and dataloaders.

The expected input is a Roboflow *multi-class classification* export, which
stores one folder per class inside each split::

    Parking Lot.v10i.multiclass/
        train/empty/*.jpg
        train/occupied/*.jpg
        valid/empty/*.jpg
        ...
        test/empty/*.jpg

A single folder of class sub-directories (no train/ valid/ test/ wrappers) is
also accepted; in that case a stratified validation split is carved out of it.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Some Roboflow exports contain barely-truncated JPEGs; decode what we can
# instead of crashing a whole epoch on one bad file.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_SPLIT_NAMES = ("train", "training")
VAL_SPLIT_NAMES = ("valid", "val", "validation")
TEST_SPLIT_NAMES = ("test", "testing")

Sample = tuple[Path, int]


@dataclass
class DatasetBundle:
    """Resolved samples for each split, with a shared class index."""

    root: Path
    classes: list[str]
    train: list[Sample] = field(default_factory=list)
    val: list[Sample] = field(default_factory=list)
    test: list[Sample] = field(default_factory=list)
    layout: str = "unknown"

    def counts(self, split: str) -> list[int]:
        counts = Counter(label for _, label in getattr(self, split))
        return [counts.get(i, 0) for i in range(len(self.classes))]

    def summary(self) -> str:
        lines = [f"Dataset: {self.root}  (layout: {self.layout})"]
        lines.append(f"Classes ({len(self.classes)}): {', '.join(self.classes)}")
        header = f"{'split':<6} {'total':>7}  " + "  ".join(
            f"{name[:14]:>14}" for name in self.classes
        )
        lines.append(header)
        for split in ("train", "val", "test"):
            total = len(getattr(self, split))
            if total == 0 and split == "test":
                continue
            cells = "  ".join(f"{c:>14}" for c in self.counts(split))
            lines.append(f"{split:<6} {total:>7}  {cells}")
        return "\n".join(lines)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _class_dirs(directory: Path) -> list[Path]:
    """Immediate sub-directories of ``directory`` that contain images."""
    if not directory.is_dir():
        return []
    out = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and any(_is_image(f) for f in child.rglob("*") if f.is_file()):
            out.append(child)
    return out


def _images_in(directory: Path) -> list[Path]:
    return sorted(f for f in directory.rglob("*") if f.is_file() and _is_image(f))


def _find_split_dir(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir() and _class_dirs(candidate):
            return candidate
    return None


def _samples(split_dir: Path, classes: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(classes):
        class_dir = split_dir / class_name
        if class_dir.is_dir():
            samples.extend((path, label) for path in _images_in(class_dir))
    return samples


def stratified_split(
    samples: Iterable[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split samples per class so every class appears in both parts."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda s: str(s[0]))
        rng.shuffle(items)
        if val_fraction == 0.0 or len(items) < 2:
            train.extend(items)
            continue
        n_val = max(1, round(len(items) * val_fraction))
        n_val = min(n_val, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def load_dataset(
    root: str | Path, val_fraction: float = 0.2, seed: int = 42
) -> DatasetBundle:
    """Discover classes and splits for a classification export."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    train_dir = _find_split_dir(root, TRAIN_SPLIT_NAMES)
    layout = "roboflow-splits" if train_dir else "single-split"

    # When there is no train/ wrapper, the export root itself holds the classes.
    base_dir = train_dir or root
    class_dirs = _class_dirs(base_dir)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders with images found in {base_dir}. Expected "
            "'train/<class>/*.jpg' (Roboflow multi-class classification export)."
        )
    classes = [d.name for d in class_dirs]

    train_samples = _samples(base_dir, classes)

    val_dir = _find_split_dir(root, VAL_SPLIT_NAMES)
    if val_dir is not None:
        val_samples = _samples(val_dir, classes)
    else:
        train_samples, val_samples = stratified_split(train_samples, val_fraction, seed)

    test_dir = _find_split_dir(root, TEST_SPLIT_NAMES)
    test_samples = _samples(test_dir, classes) if test_dir is not None else []

    for class_name, label in zip(classes, range(len(classes))):
        if not any(lbl == label for _, lbl in train_samples):
            raise ValueError(
                f"Class '{class_name}' has no images in the training split "
                f"({base_dir / class_name})."
            )

    return DatasetBundle(
        root=root,
        classes=classes,
        train=train_samples,
        val=val_samples,
        test=test_samples,
        layout=layout,
    )


def load_rgb(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:  # pragma: no cover - bad files
        raise RuntimeError(f"Could not read image {path}: {exc}") from exc


class ParkingSpaceDataset(Dataset):
    """Cropped parking space images with integer labels."""

    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int = 224, train: bool = False, hflip: bool = True):
    """Augmentation is deliberately mild: crops come pre-resized and aligned."""
    if train:
        steps = [
            transforms.RandomResizedCrop(
                img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True
            ),
        ]
        if hflip:
            steps.append(transforms.RandomHorizontalFlip())
        steps += [
            transforms.RandomRotation(7, fill=0),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        return transforms.Compose(steps)

    return transforms.Compose(
        [
            transforms.Resize(int(round(img_size * 1.15))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def class_weights(samples: Sequence[Sample], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights so rare classes are not ignored."""
    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    weights = []
    for index in range(num_classes):
        count = counts.get(index, 0)
        weights.append(total / (num_classes * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloader(
    samples: Sequence[Sample],
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> DataLoader:
    pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        ParkingSpaceDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
