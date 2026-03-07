"""
resnet.py
─────────────────────────────────────────────────────────────────────────────
ResNet50 backbone for NEU-DET surface defect classification.

Strategy: Partial fine-tuning
    Frozen  — conv1, bn1, layer1, layer2  (low-level edges and textures)
    Unfrozen — layer3, layer4, fc          (high-level patterns + classifier)

Rationale:
    Early ResNet layers detect universal low-level features — edges, corners,
    gradients — that transfer well from ImageNet to steel surface textures.
    Freezing them prevents overwriting these on a small dataset (~1260 train
    images) and reduces the number of parameters being optimised.

    Later layers detect high-level semantic patterns that are ImageNet-specific.
    Unfreezing layer3 + layer4 allows the model to adapt to defect-specific
    features like crazing patterns, inclusion shapes, and scratch orientations.

Architecture:
    ResNet50 backbone (pretrained on ImageNet)
    └── Replace final fc layer: 2048 → 6 classes

Grad-CAM target:
    model.backbone.layer4  — last convolutional layer before global pooling.
    Produces the richest spatial activation maps for defect localisation.

Usage:
    from src.models.cnn.resnet import build_resnet

    model = build_resnet(num_classes=6)
    model = model.to(device)

    # Verify frozen/unfrozen layers
    model.print_trainable_layers()
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


# ══════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════

class ResNet50Classifier(nn.Module):
    """
    ResNet50 with a replaced classification head for NEU-DET defect classes.

    Attributes:
        backbone: full ResNet50 with modified final fc layer
                  accessed directly for Grad-CAM hook registration

    Args:
        num_classes: number of output classes (6 for NEU-DET)
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace the final fully connected layer
        # Original: Linear(2048, 1000) for ImageNet
        # Ours:     Linear(2048, 6)    for NEU-DET
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Apply partial fine-tuning strategy
        self._freeze_layers()

    # ── Fine-tuning strategy ───────────────────────────────────────────────

    def _freeze_layers(self) -> None:
        """
        Freeze early layers, unfreeze later layers and classifier head.

        Frozen  (weights fixed):
            conv1, bn1    — edge detectors, universal across domains
            layer1        — low-level texture responses
            layer2        — simple pattern detectors

        Unfrozen (weights updated during training):
            layer3        — mid-level feature combinations
            layer4        — high-level semantic features
            fc            — classifier head (always trained from scratch)
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then selectively unfreeze layer3, layer4, and fc
        for layer in (self.backbone.layer3, self.backbone.layer4, self.backbone.fc):
            for param in layer.parameters():
                param.requires_grad = True

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [batch_size, 3, 224, 224]

        Returns:
            logits: FloatTensor [batch_size, num_classes]
                    Raw unnormalised scores — pass through softmax for
                    probabilities or use directly with CrossEntropyLoss.
        """
        return self.backbone(x)

    # ── Diagnostics ────────────────────────────────────────────────────────

    def print_trainable_layers(self) -> None:
        """
        Print each layer's name and whether its weights will be updated.
        Call this after build_resnet() to verify the fine-tuning strategy.

        Example output:
            conv1         frozen
            bn1           frozen
            layer1        frozen
            layer2        frozen
            layer3        trainable   ← unfrozen
            layer4        trainable   ← unfrozen
            fc            trainable   ← unfrozen
        """
        print(f"\n{'Layer':<20} {'Status'}")
        print("-" * 32)
        for name, module in self.backbone.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            status    = "trainable" if trainable else "frozen"
            print(f"{name:<20} {status}")

    def count_parameters(self) -> dict[str, int]:
        """
        Count trainable vs total parameters.

        Returns:
            {
                "trainable": number of parameters being optimised,
                "frozen":    number of fixed parameters,
                "total":     total parameter count,
            }

        Used in the comparison notebook to evaluate parameter efficiency
        against the ViT model.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())

        return {
            "trainable": trainable,
            "frozen":    total - trainable,
            "total":     total,
        }


# ══════════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════════

def build_resnet(num_classes: int = 6) -> ResNet50Classifier:
    """
    Build and return a ResNet50Classifier with partial fine-tuning applied.

    Args:
        num_classes: number of defect classes (default 6 for NEU-DET)

    Returns:
        ResNet50Classifier — ready to move to device and train

    Example:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_resnet(num_classes=6)
        model = model.to(device)
        model.print_trainable_layers()

        params = model.count_parameters()
        print(f"Trainable: {params['trainable']:,}")
        print(f"Total:     {params['total']:,}")
    """
    model = ResNet50Classifier(num_classes=num_classes)
    return model
