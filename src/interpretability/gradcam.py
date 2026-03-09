"""
gradcam.py
─────────────────────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class GradCAM():
    def __init__(self, model) -> None:
        self.activations = None
        self.gradients   = None

        def save_activation(module, input, output):
            self.activations = output

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        model.backbone.layer4.register_forward_hook(save_activation)
        model.backbone.layer4.register_full_backward_hook(save_gradient)
        self.model = model

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        activations = self.activations.detach()
        gradients   = self.gradients.detach()
        weights = gradients.mean(dim=[2, 3]).squeeze(0)
        weighted_sum = torch.einsum("bcd,b -> cd",activations, weights)
        heatmap = F.relu(weighted_sum)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)   # adds batch and channel dims
        heatmap = F.interpolate(heatmap,(224,224), mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze() 
        return heatmap.cpu().numpy()