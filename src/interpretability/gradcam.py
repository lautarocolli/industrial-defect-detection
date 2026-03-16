"""
gradcam.py
─────────────────────────────────────────────────────────────────────────────
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet50 and ViT.

Grad-CAM answers: "which parts of the image did the model focus on?"

It works by:
    1. Capturing target layer activations during the forward pass
    2. Capturing target layer gradients during the backward pass
    3. Weighting each activation by its mean gradient
    4. Producing a spatial heatmap that highlights discriminative regions

Supports two architectures via automatic shape detection:

    ResNet50 — target layer4, activations [2048, 7, 7]
        Weights each channel by its mean gradient across spatial dims,
        produces a [7, 7] heatmap upsampled to [224, 224].

    ViT-B/16 — target last encoder block, activations [197, 768]
        Weights each token by its mean gradient across embedding dims,
        drops the CLS token, reshapes 196 patch scores to [14, 14],
        upsamples to [224, 224].

Usage:
    from src.interpretability.gradcam import GradCAM
    from src.models.cnn.resnet import build_resnet
    from src.models.transformer.vit import build_vit

    # ResNet
    cam = GradCAM(model, target_layer=model.backbone.layer4)

    # ViT
    cam = GradCAM(vit_model, target_layer=vit_model.model.encoder.layers[-1])

    sample  = test_loader.dataset[idx]
    boxes   = test_loader.dataset.get_boxes(sample["image_path"])
    classes = test_loader.dataset.classes

    cam.visualize(sample, classes, boxes=boxes)
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class GradCAM():
    """
    Grad-CAM implementation supporting ResNet50 and ViT-B/16.

    The target layer is passed in at construction time — making this class
    architecture-agnostic. Shape detection in generate() determines which
    computational path to follow.

    Args:
        model:        trained ResNet50Classifier or ViT_B_16_Classifier
        target_layer: nn.Module to hook — layer4 for ResNet, last encoder
                      block for ViT
    """

    def __init__(self, model, target_layer) -> None:
        # Placeholders — filled automatically when hooks fire
        self.activations = None
        self.gradients   = None

        # Forward hook — fires when target_layer produces its output
        # Saves whatever comes out of the layer for later use
        def save_activation(module, input, output):
            self.activations = output
            output.register_hook(lambda grad: setattr(self, 'gradients', grad))

        # Backward hook — fires when gradients flow back through target_layer
        # grad_output[0] contains gradients w.r.t. the layer's output
        # These tell us how much each activation influenced the class score
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Attach both hooks to the target layer
        # They fire automatically on every forward/backward pass
        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(save_gradient)

        # Store model so generate() and visualize() can access it via self
        self.model = model

    def generate(self, image_tensor, class_idx=None):
        """
        Generate a Grad-CAM heatmap for a single image.

        Automatically detects architecture from activation shape:
            [seq, embed]   → ViT path   (e.g. [197, 768])
            [ch, h, w]     → ResNet path (e.g. [2048, 7, 7])

        Args:
            image_tensor: FloatTensor [1, 3, 224, 224] — preprocessed image on device
            class_idx:    int or None — class to visualise. If None, uses the
                          predicted class (argmax of output logits)

        Returns:
            heatmap:   numpy array [224, 224] — values in [0, 1]
                       high values = regions the model focused on
            class_idx: int — the class that was visualised
        """
        # Eval mode — disables dropout, stable BatchNorm statistics
        self.model.eval()

        # Forward pass — hooks fire and save activations in self.activations
        output = self.model(image_tensor)

        # Use predicted class if none specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Clear gradients from previous operations
        self.model.zero_grad()

        # Backward pass on the specific class score — not the loss
        # Hooks fire and save gradients in self.gradients
        # Must happen before detaching activations/gradients
        output[0, class_idx].backward()

        # Detach both from computation graph — done with backprop
        # squeeze(0) removes the batch dimension
        activations = self.activations.detach().squeeze(0)  # remove batch dim
        gradients   = self.gradients.detach().squeeze(0)    # remove batch dim

        if len(activations.shape) == 2:
            # ── ViT path ───────────────────────────────────────────────────
            # activations: [197, 768] — 197 tokens, 768 embedding dims
            # gradients:   [197, 768] — same shape

            # One importance weight per token — mean across embedding dimension
            # "how much did this token's embedding influence the class score?"
            weights = gradients.mean(dim=1)              # [197]

            # Scale each token's activation by its importance weight
            # unsqueeze(1) broadcasts weights [197] → [197, 1] for multiplication
            # sum across embedding dimension → one score per token
            weighted_sum = (activations * weights.unsqueeze(1)).sum(dim=1)  # [197]

            # Drop the CLS token (index 0) — it's not a spatial patch
            # Keep only the 196 patch tokens
            weighted_sum = weighted_sum[1:]              # [197] → [196]

            # ReLU — keep only positive contributions
            heatmap = F.relu(weighted_sum)

            # Normalise to [0, 1]
            heatmap = heatmap / (heatmap.max() + 1e-8)

            # Reshape flat patch sequence into 2D spatial grid
            # 196 patches → 14×14 (224 / 16 = 14 patches per side)
            heatmap = heatmap.reshape(14, 14)

            # Upsample to [224, 224]
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # [14,14] → [1, 1, 14, 14]
            heatmap = F.interpolate(
                heatmap,
                (224, 224),
                mode="bilinear",
                align_corners=False
            )
            heatmap = heatmap.squeeze()                  # [1, 1, 224, 224] → [224, 224]

        else:
            # ── ResNet path ────────────────────────────────────────────────
            # activations: [2048, 7, 7] — 2048 channels, 7×7 spatial map
            # gradients:   [2048, 7, 7] — same shape

            # One importance weight per channel — mean across spatial dims
            # "how much did this channel's spatial map influence the class score?"
            weights = gradients.mean(dim=[1, 2])         # [2048]

            # Weighted sum across channels
            # "bcd,b -> cd": for each spatial position [7,7],
            # multiply each channel b by its weight, then sum across channels
            weighted_sum = torch.einsum("bcd,b -> cd", activations, weights)  # [7, 7]

            # ReLU — keep only positive activations
            heatmap = F.relu(weighted_sum)

            # Normalise to [0, 1]
            heatmap = heatmap / (heatmap.max() + 1e-8)

            # Upsample from [7, 7] → [224, 224]
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # [7,7] → [1, 1, 7, 7]
            heatmap = F.interpolate(
                heatmap,
                (224, 224),
                mode="bilinear",
                align_corners=False
            )
            heatmap = heatmap.squeeze()                  # [1, 1, 224, 224] → [224, 224]

        # Move to CPU and convert to numpy — required for matplotlib
        return heatmap.cpu().numpy(), class_idx

    def visualize(self, sample, classes, boxes=None):
        """
        Generate and display a Grad-CAM heatmap alongside ground truth boxes.

        Shows two plots side by side:
            Left  — original grayscale image with ground truth bounding boxes
            Right — same image with Grad-CAM heatmap overlaid

        The comparison reveals whether the model's attention aligns with
        the annotated defect region — the core interpretability finding.

        Args:
            sample:  dict from NEUDefectDataset.__getitem__ containing
                     "image" (tensor) and "image_path" (str)
            classes: list of class name strings from dataset.classes
            boxes:   FloatTensor [N, 4] from dataset.get_boxes() or None
                     each row is [xmin, ymin, xmax, ymax] in 224×224 space
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Add batch dimension and move to device
        # generate() expects [1, 3, 224, 224] not [3, 224, 224]
        image_tensor = sample["image"].unsqueeze(0).to(device)

        # Single forward+backward pass produces both heatmap and prediction
        heatmap, pred_idx = self.generate(image_tensor)
        pred_class        = classes[pred_idx]

        # Open original untransformed image for display
        # Resized to 224×224 to match heatmap and box coordinate space
        image_path = sample["image_path"]
        image      = Image.open(image_path).convert("L").resize((224, 224))

        # Two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left plot — original image with ground truth boxes
        ax1.imshow(image, cmap="gray")

        # Right plot — image with heatmap overlay
        # Two imshow calls on same axis: second renders on top with transparency
        ax2.imshow(image, cmap="gray")
        ax2.imshow(heatmap, alpha=0.5, cmap="jet")  # jet: blue=low, red=high activation

        # Draw ground truth bounding boxes on left plot
        # Loop handles images with multiple defect instances
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                rect = patches.Rectangle(
                    (xmin, ymin),        # top-left corner
                    xmax - xmin,         # width
                    ymax - ymin,         # height
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"     # transparent fill — border only
                )
                ax1.add_patch(rect)

        # Clean up axes and add titles
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title("Original + Ground Truth Boxes")
        ax2.set_title(f"Grad-CAM: {pred_class}")

        plt.tight_layout()
        plt.show()