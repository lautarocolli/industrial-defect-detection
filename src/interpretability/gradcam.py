"""
gradcam.py
─────────────────────────────────────────────────────────────────────────────
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet50.

Grad-CAM answers: "which parts of the image did the model focus on?"

It works by:
    1. Capturing layer4 activations during the forward pass
    2. Capturing layer4 gradients during the backward pass
    3. Weighting each activation channel by its mean gradient
    4. Producing a spatial heatmap that highlights discriminative regions

The heatmap is overlaid on the original image alongside ground truth
bounding boxes — allowing visual comparison between where the defect
actually is and where the model looked.

Usage:
    from src.interpretability.gradcam import GradCAM

    cam = GradCAM(model)

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
    Grad-CAM implementation for ResNet50Classifier.

    Hooks are registered on model.backbone.layer4 — the last convolutional
    block before global average pooling. This layer retains spatial resolution
    [7, 7] while encoding the richest semantic features, making it the best
    target for defect localisation heatmaps.

    Args:
        model: trained ResNet50Classifier instance
    """

    def __init__(self, model) -> None:
        # Placeholders — filled automatically when hooks fire
        # during forward and backward passes
        self.activations = None
        self.gradients   = None

        # Forward hook — fires when layer4 produces its output
        # Saves the activation maps [1, 2048, 7, 7] for later use
        def save_activation(module, input, output):
            self.activations = output

        # Backward hook — fires when gradients flow back through layer4
        # grad_output[0] contains gradients w.r.t. layer4's output [1, 2048, 7, 7]
        # These tell us how much each activation influenced the class score
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Attach both hooks to layer4
        # They fire automatically on every forward/backward pass from this point
        model.backbone.layer4.register_forward_hook(save_activation)
        model.backbone.layer4.register_full_backward_hook(save_gradient)

        # Store model so generate() and visualize() can access it via self
        self.model = model

    def generate(self, image_tensor, class_idx=None):
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            image_tensor: FloatTensor [1, 3, 224, 224] — preprocessed image on device
            class_idx:    int or None — class to visualise. If None, uses the
                          predicted class (argmax of output logits)

        Returns:
            heatmap:   numpy array [224, 224] — values in [0, 1]
                       high values = regions the model focused on
            class_idx: int — the class that was visualised
        """
        # Eval mode
        self.model.eval()

        # Forward pass — hooks fire and save activations in self.activations
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Clear any gradients from previous operations
        self.model.zero_grad()

        # Backward pass on the specific class score
        output[0, class_idx].backward()

        # Detach from computation graph — we're done with backprop
        # .detach() prevents errors if further operations are performed
        activations = self.activations.detach()  # [1, 2048, 7, 7]
        gradients   = self.gradients.detach()    # [1, 2048, 7, 7]

        # Compute channel importance weights
        weights = gradients.mean(dim=[2, 3]).squeeze(0)   # [2048]

        # Weighted sum of activation channels
        # "bcd,b -> cd" means: for each spatial position [7,7],
        # multiply each channel b by its weight b, then sum across channels
        # Result: [7, 7] — single spatial heatmap
        weighted_sum = torch.einsum("bcd,b -> cd", activations, weights)

        # ReLU — keep only positive activations
        heatmap = F.relu(weighted_sum)

        # Normalise to [0, 1] so values are interpretable as intensities
        # 1e-8 epsilon prevents division by zero if all activations are 0
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Upsample from [7, 7] → [224, 224] to match the original image size
        # F.interpolate requires [batch, channel, h, w] so we add those dims first
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)    # [7,7] → [1, 1, 7, 7]
        heatmap = F.interpolate(
            heatmap,
            (224, 224),
            mode="bilinear",          # smooth interpolation, better than nearest
            align_corners=False
        )
        heatmap = heatmap.squeeze()                    # [1, 1, 224, 224] → [224, 224]

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
        # The tensor version is normalised and looks wrong to human eyes
        image_path = sample["image_path"]
        image      = Image.open(image_path).convert("L")

        # Two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left plot — original image
        ax1.imshow(image, cmap="gray")

        # Right plot — image with heatmap overlay
        # imshow twice on same axis: second call renders on top with alpha transparency
        ax2.imshow(image, cmap="gray")
        ax2.imshow(heatmap, alpha=0.5, cmap="jet")  # jet: blue=low, red=high activation

        # Draw ground truth bounding boxes on left plot if provided
        # Looping handles images with multiple defect instances
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                rect = patches.Rectangle(
                    (xmin, ymin),        # top-left corner
                    xmax - xmin,         # width
                    ymax - ymin,         # height
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"     # transparent fill — just the border
                )
                ax1.add_patch(rect)

        # Clean up axes and add titles
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title("Original + Ground Truth Boxes")
        ax2.set_title(f"Grad-CAM: {pred_class}")

        plt.tight_layout()
        plt.show()