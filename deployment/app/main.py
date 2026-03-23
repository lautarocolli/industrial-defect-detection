"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application for NEU-DET surface defect classification.

Endpoints:
    GET  /health          — health check for Kubernetes readiness probes
    POST /predict         — classify a defect image, return JSON scores
    POST /predict/visual  — classify a defect image, return Grad-CAM PNG

Model:
    ResNet50 pretrained on ImageNet, partially fine-tuned on NEU-DET.
    Weights loaded from Hugging Face Hub at startup.
    Classifies 6 defect types: crazing, inclusion, patches,
    pitted_surface, rolled-in_scale, scratches.

Running locally:
    uvicorn deployment.app.main:app --reload

Interactive docs:
    http://localhost:8000/docs
─────────────────────────────────────────────────────────────────────────────
"""

import io
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from deployment.app.model import load_model
from src.data_operations.transforms import get_val_transforms


# ══════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: loads the ResNet50 model, class list, and GradCAM instance
    from Hugging Face Hub and stores them on app.state. The server only
    starts accepting requests after this completes — guaranteeing the
    model is ready before any request arrives.

    Shutdown: no cleanup required for a PyTorch inference model.
    """
    # ── Startup ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # load_model downloads weights from Hugging Face Hub on first run
    # subsequent startups use the local cache — no re-download needed
    app.state.model, app.state.classes, app.state.cam = load_model(device)
    app.state.device = device

    print(f"Model loaded. Classes: {app.state.classes}")

    yield  # server runs here — handling requests

    # ── Shutdown ───────────────────────────────────────────────────────────
    print("Shutting down.")


# ══════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    lifespan    = lifespan,
    title       = "NEU-DET Defect Detection API",
    description = (
        "ResNet50-based surface defect classifier trained on the NEU Surface "
        "Defect Database. Classifies 6 defect types with Grad-CAM interpretability."
    ),
    version     = "1.0.0",
)


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════

def preprocess(image_bytes: bytes, device: torch.device):
    """
    Preprocess raw image bytes into a model-ready tensor.

    Applies the same pipeline used during training:
        bytes → PIL Image (grayscale, 224×224) → val transforms → tensor

    Args:
        image_bytes: raw bytes from UploadFile.read()
        device:      torch.device to move the tensor to

    Returns:
        pil_image:    PIL Image resized to 224×224 — used for display
        image_tensor: FloatTensor [1, 3, 224, 224] — ready for model input
    """
    # Convert bytes to PIL Image — grayscale to match training data
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((224, 224))

    # Apply val transforms — resize, grayscale→RGB, normalise with ImageNet stats
    # No augmentation — deterministic transforms only for inference
    transform    = get_val_transforms()
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    return pil_image, image_tensor


# ══════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """
    Health check endpoint for Kubernetes readiness and liveness probes.

    Returns 200 OK with a status message when the server is running.
    Kubernetes uses this to decide whether to route traffic to this pod.

    Returns:
        JSON: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Classify a steel surface defect image.

    Upload a grayscale steel surface image and receive the predicted
    defect class with confidence scores for all 6 defect types.

    Args:
        file: image file upload (jpg or png recommended)

    Returns:
        JSON with:
            predicted_class: name of the predicted defect class
            confidence:      model confidence for the predicted class (0–1)
            all_scores:      probability scores for all 6 classes
    """
    model   = request.app.state.model
    classes = request.app.state.classes
    device  = request.app.state.device

    # Read uploaded file bytes and preprocess into model-ready tensor
    image_bytes       = await file.read()
    pil_image, tensor = preprocess(image_bytes, device)

    # Run inference — no gradients needed for classification only
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1).squeeze()

    # Extract prediction and build response
    pred_idx   = probs.argmax().item()
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx].item()

    # Build per-class score dict — sorted by confidence descending
    all_scores = {
        cls: round(probs[i].item(), 4)
        for i, cls in enumerate(classes)
    }
    all_scores = dict(
        sorted(all_scores.items(), key=lambda x: -x[1])
    )

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 4),
        "all_scores":      all_scores,
    }


@app.post("/predict/visual")
async def predict_visual(request: Request, file: UploadFile = File(...)):
    """
    Classify a steel surface defect image and return a Grad-CAM heatmap.

    Upload a grayscale steel surface image and receive a PNG showing
    the original image with a Grad-CAM activation heatmap overlaid,
    highlighting the regions the model focused on for its prediction.

    Useful for human review in quality control contexts — shows not just
    what defect was detected but where the model found it.

    Args:
        file: image file upload (jpg or png recommended)

    Returns:
        PNG image — original image with Grad-CAM heatmap overlay and
                    predicted class label as the plot title
    """
    cam     = request.app.state.cam
    classes = request.app.state.classes
    device  = request.app.state.device

    # Read uploaded file bytes and preprocess
    image_bytes       = await file.read()
    pil_image, tensor = preprocess(image_bytes, device)

    # Generate Grad-CAM heatmap as PNG bytes
    png_bytes = cam.to_bytes(tensor, pil_image, classes)

    # Stream PNG bytes back as image response
    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png"
    )