# app/model.py
import torch
from huggingface_hub import hf_hub_download
from src.models.cnn.resnet import build_resnet
from src.interpretability.gradcam import GradCAM

def load_model(device):
    # Download weights from Hub at startup
    path = hf_hub_download(
        repo_id  = "lautivuelos/neu-det-defect-detection",
        filename = "resnet50_final.pth"
    )

    checkpoint = torch.load(path, map_location=device)

    model = build_resnet(num_classes=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    cam = GradCAM(model, target_layer=model.backbone.layer4)

    return model, checkpoint["classes"], cam