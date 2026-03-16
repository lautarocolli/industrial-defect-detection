"""
attention_viz.py
─────────────────────────────────────────────────────────────────────────────
Attention Rollout visualisation for ViT-B/16.

Attention Rollout answers: "which image patches did the model focus on?"

It works by:
    1. Capturing attention weights from all 12 transformer blocks via hooks
    2. Averaging across attention heads per block
    3. Adding identity matrices to account for residual connections
    4. Multiplying attention matrices across all 12 blocks (rollout)
    5. Extracting CLS token attention to all 196 patch tokens
    6. Reshaping to a 14×14 spatial grid and upsampling to 224×224

The resulting heatmap is overlaid on the original image alongside ground
truth bounding boxes — allowing direct comparison with ResNet50 Grad-CAM
to understand how CNNs and Transformers differently localise defects.

Usage:
    from src.interpretability.attention_viz import AttentionRollout

    rollout = AttentionRollout(vit_model)

    sample  = test_loader.dataset[idx]
    boxes   = test_loader.dataset.get_boxes(sample["image_path"])
    classes = test_loader.dataset.classes

    rollout.visualize(sample, classes, boxes=boxes)
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class AttentionRollout():
    """
    Attention Rollout implementation for ViT-B/16Classifier.

    Hooks are registered on the self_attention submodule of every transformer
    block in model.model.encoder.layers. This captures attention weights from
    all 12 blocks during a single forward pass, which are then combined via
    the rollout algorithm to produce a faithful spatial attention map.

    Args:
        model: trained ViT_B_16_Classifier instance
    """

    def __init__(self, model) -> None:
        # Hook handles — stored so hooks can be removed if needed
        # Never reset — hooks must persist across generate() calls
        self._hooks = []

        # Attention weights captured during forward pass
        # Reset at the start of each generate() call
        self.attention_maps = []

        # Forward hook — fires when self_attention produces its output
        # output is a tuple: (context, attention_weights)
        # output[1] contains attention weights [1, num_heads, 197, 197]
        def save_attention(module, input, output):
            self.attention_maps.append(output[1].detach())

        # Register hook on every transformer block's self_attention module
        # 12 blocks → 12 hooks → 12 attention matrices per forward pass
        for block in model.model.encoder.layers:
            handle = block.self_attention.register_forward_hook(save_attention)
            self._hooks.append(handle)

        # Store model so generate() and visualize() can access it via self
        self.model = model

    def generate(self, image_tensor):
        """
        Generate an attention rollout heatmap for a single image.

        Runs one forward pass, captures attention weights from all 12 blocks,
        and combines them via rollout to produce a single spatial heatmap.

        Args:
            image_tensor: FloatTensor [1, 3, 224, 224] — preprocessed image on device

        Returns:
            heatmap:   numpy array [224, 224] — values in [0, 1]
                       high values = patches the CLS token attended to most
            class_idx: int — predicted class index
        """
        # Reset attention maps — prevents maps from previous calls accumulating
        self.attention_maps = []

        # Eval mode — disables dropout, uses stable BatchNorm statistics
        self.model.eval()

        # Forward pass — hooks fire and fill self.attention_maps with 12 tensors
        # each tensor shape: [1, num_heads, 197, 197]
        output = self.model(image_tensor)

        # Predicted class — argmax of logits across 6 classes
        class_idx = output.argmax(dim=1).item()

        # Stack all 12 attention matrices into one tensor
        # list of 12 × [1, 12, 197, 197] → [12, 1, 12, 197, 197]
        attn = torch.stack(self.attention_maps)

        # Average across attention heads — collapses head dimension
        # [12, 1, 12, 197, 197] → [12, 197, 197]
        attn = attn.mean(dim=1)

        # Add identity matrix to account for residual connections
        # In each transformer block, tokens also attend to themselves
        # via the residual path — the identity matrix represents this
        attn = attn + torch.eye(attn.shape[-1]).to(attn.device)

        # Normalise each row so attention weights sum to 1
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Rollout — multiply attention matrices across all 12 layers
        # This propagates attention through the full depth of the network
        # rollout[i,j] = how much token i attends to token j across ALL layers
        rollout = attn[0]
        for i in range(1, len(attn)):
            rollout = rollout @ attn[i]

        # Extract CLS token attention to all 196 patch tokens
        # CLS token is index 0, patch tokens are indices 1:197
        # Result: [196] — one attention score per patch
        cls_attention = rollout[0, 1:]

        # Reshape flat patch sequence into 2D spatial grid
        # 196 patches → 14×14 grid (224/16 = 14 patches per side)
        heatmap = cls_attention.reshape(14, 14)

        # Upsample from [14, 14] → [224, 224] to match original image size
        # F.interpolate requires [batch, channel, h, w] so we add those dims
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)    # [14, 14] → [1, 1, 14, 14]
        heatmap = F.interpolate(
            heatmap,
            (224, 224),
            mode="bilinear",       # smooth interpolation, better than nearest
            align_corners=False
        )

        # Normalise to [0, 1] — 1e-8 prevents division by zero
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Remove batch and channel dims added for interpolation
        heatmap = heatmap.squeeze()                    # [1, 1, 224, 224] → [224, 224]

        # Move to CPU and convert to numpy — required for matplotlib
        return heatmap.cpu().numpy(), class_idx

    def visualize(self, sample, classes, boxes=None):
        """
        Generate and display an attention rollout heatmap alongside ground truth boxes.

        Shows two plots side by side:
            Left  — original grayscale image with ground truth bounding boxes
            Right — same image with attention rollout heatmap overlaid

        The comparison reveals whether the ViT's attention aligns with the
        annotated defect region. Comparing this against ResNet50 Grad-CAM
        shows how CNNs and Transformers differently localise defects —
        the core interpretability finding of this project.

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

        # Forward pass produces both heatmap and predicted class index
        heatmap, pred_idx = self.generate(image_tensor)
        pred_class        = classes[pred_idx]

        # Open original untransformed image for display
        # Resized to 224×224 to match heatmap and box coordinate space
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("L").resize((224, 224))

        # Two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left plot — original image with ground truth boxes
        ax1.imshow(image, cmap="gray")

        # Right plot — image with attention heatmap overlay
        # Two imshow calls on same axis: second renders on top with transparency
        ax2.imshow(image, cmap="gray")
        ax2.imshow(heatmap, alpha=0.5, cmap="jet")  # jet: blue=low, red=high attention

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

        # Clean up axes and add descriptive titles
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title("Original + Ground Truth Boxes")
        ax2.set_title(f"Attention Rollout: {pred_class}")

        plt.tight_layout()
        plt.show()