"""
vit.py
─────────────────────────────────────────────────────────────────────────────
Vision Transformer (ViT-B/16) for NEU-DET surface defect classification.

Architecture:
    ViT-B/16 — Base model with 16×16 pixel patches
    - Input image [3, 224, 224] split into 196 patches of 16×16 pixels
    - Each patch embedded as a 768-dimensional vector
    - 197 tokens total (196 patches + 1 CLS token)
    - 12 Transformer encoder blocks with multi-head self-attention
    - CLS token output → classifier head → 6 defect classes

Strategy: Feature extraction
    Frozen   — entire ViT backbone (encoder, patch embedding, positional encoding)
    Unfrozen — heads.head only (classifier)

Rationale:
    ViT-B/16 has ~86M parameters vs ResNet50's ~25M. On a small dataset
    like NEU-DET (~1260 training images), fine-tuning large portions of
    the model risks severe overfitting. Feature extraction keeps pretrained
    ImageNet representations intact and only adapts the classifier to the
    6 defect classes.

Attention visualisation target:
    The last transformer encoder block — its attention weights show which
    patches the CLS token attended to most, producing a patch-level heatmap
    for interpretability comparison against ResNet50 Grad-CAM.

Usage:
    from src.models.transformer.vit import build_vit

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = build_vit(num_classes=6)
    vit_model = vit_model.to(device)

    vit_model.print_trainable_layers()

    params = vit_model.count_parameters()
    print(f"Trainable: {params['trainable']:,}")
    print(f"Total:     {params['total']:,}")
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ══════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════

class ViT_B_16_Classifier(nn.Module):
    """
    ViT-B/16 with a replaced classification head for NEU-DET defect classes.

    The entire backbone is frozen — only the classifier head is trained.
    This is feature extraction, the recommended strategy for ViT on small
    industrial datasets where the parameter count far exceeds training data.

    Args:
        num_classes: number of output classes (6 for NEU-DET)
    """

    def __init__(self, num_classes: int = 6) -> None:
        # Initialise nn.Module internals — always required first
        super().__init__()

        # Load ViT-B/16 with pretrained ImageNet weights
        # ViT_B_16_Weights.DEFAULT always fetches the best available weights
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Read input features of the existing head before replacing it
        # For ViT-B/16 this is always 768 — the transformer embedding dimension
        # Reading it programmatically means this works if you swap to vit_l_16
        in_features = self.model.heads.head.in_features

        # Replace the original classifier head
        # Original: Linear(768, 1000) for ImageNet
        # Ours:     Linear(768, 6)    for NEU-DET
        self.model.heads.head = nn.Linear(in_features, num_classes)

        # Apply feature extraction strategy — freeze all, unfreeze head
        self._freeze_layers()

    # ── Fine-tuning strategy ───────────────────────────────────────────────

    def _freeze_layers(self) -> None:
        """
        Freeze entire backbone, unfreeze only the classifier head.

        Frozen  (weights fixed):
            conv_proj         — patch embedding projection
            encoder           — all 12 transformer blocks
            class_token       — learnable CLS token
            positional embedding

        Unfrozen (weights updated during training):
            heads.head        — classifier Linear(768, 6)
        """
        # First freeze every parameter in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Then unfreeze only the classifier head
        # heads contains only heads.head for ViT-B/16
        for param in self.model.heads.parameters():
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

        Internally runs the full ViT pipeline:
            patch splitting → embedding → positional encoding →
            12 transformer blocks → CLS token → classifier head
        """
        return self.model(x)

    # ── Diagnostics ────────────────────────────────────────────────────────

    def print_trainable_layers(self) -> None:
        """
        Print each top-level module and whether its weights will be updated.
        Call after build_vit() to verify the feature extraction strategy.

        Example output:
            conv_proj            frozen
            encoder              frozen
            heads                trainable  ← only this updates
        """
        print(f"\n{'Layer':<20} {'Status'}")
        print("-" * 32)
        for name, module in self.model.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            status    = "trainable" if trainable else "frozen"
            print(f"{name:<20} {status}")

    def count_parameters(self) -> dict[str, int]:
        """
        Count trainable vs total parameters.

        Returns:
            {
                "trainable": parameters being optimised,
                "frozen":    fixed parameters,
                "total":     total parameter count,
            }

        Used in the comparison notebook alongside ResNet50's count_parameters()
        to evaluate the parameter efficiency of each architecture.

        Expected values for ViT-B/16 with feature extraction:
            trainable  ~4,614    (just the Linear head)
            total      ~86M
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

def build_vit(num_classes: int = 6) -> ViT_B_16_Classifier:
    """
    Build and return a ViT_B_16_Classifier with feature extraction applied.

    Args:
        num_classes: number of defect classes (default 6 for NEU-DET)

    Returns:
        ViT_B_16_Classifier — ready to move to device and train

    Example:
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model = build_vit(num_classes=6)
        vit_model = vit_model.to(device)
        vit_model.print_trainable_layers()
    """
    model = ViT_B_16_Classifier(num_classes=num_classes)
    return model