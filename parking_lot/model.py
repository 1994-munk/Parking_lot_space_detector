"""Model factory built on torchvision backbones."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision import models

# name -> (factory, weights enum name)
BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights"),
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet34": (models.resnet34, "ResNet34_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),
}

DEFAULT_BACKBONE = "mobilenet_v3_small"


def _replace_head(model: nn.Module, num_classes: int) -> None:
    """Swap the final linear layer of the common torchvision heads."""
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            for index in reversed(range(len(classifier))):
                layer = classifier[index]
                if isinstance(layer, nn.Linear):
                    classifier[index] = nn.Linear(layer.in_features, num_classes)
                    return
        elif isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    heads = getattr(model, "heads", None)
    if heads is not None and hasattr(heads, "head") and isinstance(heads.head, nn.Linear):
        heads.head = nn.Linear(heads.head.in_features, num_classes)
        return

    raise ValueError(f"Don't know how to replace the classifier head of {type(model).__name__}")


def build_model(
    name: str = DEFAULT_BACKBONE,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float | None = None,
) -> nn.Module:
    """Create a classifier. Falls back to random init if weights can't download."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(BACKBONES)}")

    factory, weights_name = BACKBONES[name]
    model = None
    if pretrained:
        try:
            weights = getattr(models, weights_name).DEFAULT
            model = factory(weights=weights)
        except Exception as exc:  # offline / blocked download — still trainable
            warnings.warn(
                f"Could not load pretrained '{name}' weights ({exc}). "
                "Continuing with randomly initialised weights.",
                stacklevel=2,
            )
    if model is None:
        model = factory(weights=None)

    _replace_head(model, num_classes)

    if dropout is not None and hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            for index, layer in enumerate(classifier):
                if isinstance(layer, nn.Dropout):
                    classifier[index] = nn.Dropout(dropout)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device
