# Industrial Defect Detection
Benchmarking CNNs and Vision Transformers for industrial surface defect classification with Grad-CAM interpretability, deployed as a REST API on Kubernetes.

## Results

| Metric | ResNet50 | ViT-B/16 |
|---|---|---|
| Test Accuracy | 100% | 100% |
| Test Loss | 0.0017 | 0.0128 |
| Trainable Parameters | 22.1M | 4,614 |
| Total Parameters | 23.5M | 85.8M |
| Epochs to ~99% Val Acc | 5 | 4 |
| Interpretability | Grad-CAM ✓ | Not meaningful ✗ |

**ResNet50 selected for deployment** — same accuracy, faster convergence, and meaningful Grad-CAM spatial explanations.

---

## Project Structure

```
industrial-defect-detection/
│
├── data/                          # gitignored
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   ├── 00_eda.ipynb               # dataset exploration and analysis
│   ├── 01_cnn_experiments.ipynb   # ResNet50 training and Grad-CAM
│   ├── 02_vit_experiments.ipynb   # ViT-B/16 training and attention analysis
│   └── 03_comparison.ipynb        # side-by-side comparison and model selection
│
├── src/
│   ├── data_operations/
│   │   ├── dataset.py             # NEUDefectDataset and DataLoader factory
│   │   ├── transforms.py          # training and val/test transforms
│   │   └── split_dataset.py       # stratified train/val/test split script
│   ├── models/
│   │   ├── cnn/resnet.py          # ResNet50 with partial fine-tuning
│   │   └── transformer/vit.py     # ViT-B/16 with feature extraction
│   ├── training/
│   │   └── train.py               # model-agnostic training and evaluation
│   ├── interpretability/
│   │   ├── gradcam.py             # Grad-CAM for ResNet50 and ViT
│   │   └── attention_viz.py       # Attention Rollout for ViT
│   └── utils/
│       └── seed.py                # deterministic seeding
│
├── deployment/
│   ├── app/
│   │   ├── main.py                # FastAPI application
│   │   └── model.py               # model loading from Hugging Face Hub
│   └── pyproject.toml             # deployment-specific dependencies
│
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment manifest
│   └── service.yaml               # Kubernetes service manifest
│
├── experiments/                   # gitignored — model checkpoints
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Model Weights

Pretrained weights are available on Hugging Face Hub:
[lautivuelos/neu-det-defect-detection](https://huggingface.co/lautivuelos/neu-det-defect-detection)

---

## Dataset

This project uses the [NEU Surface Defect Database](https://www.kaggle.com/datasets/rdsunday/neu-urface-defect-database) — 1800 grayscale steel surface images across 6 defect classes, 300 images per class.

**Classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

**Split:** 70% train / 15% val / 15% test — stratified by class, zero overlap verified.

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/lautivuelos/industrial-defect-detection.git
cd industrial-defect-detection
```

### 2. Create the Virtual Environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
pip install uv
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv sync
uv sync --group dev
```

### 4. Register Jupyter Kernel

```bash
python -m ipykernel install --user \
    --name industrial-defect-detection \
    --display-name "Python (industrial-defect-detection)"
jupyter notebook
```

### 5. Download the Dataset

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

mkdir data && cd data
kaggle datasets download -d rdsunday/neu-urface-defect-database
unzip neu-urface-defect-database.zip
rm neu-urface-defect-database.zip
```

### 6. Split the Dataset

```bash
python src/data_operations/split_dataset.py
```

Splits into 70/15/15 with stratification — both images and annotations.

---

## 🚀 Deployment

### Run Locally

```bash
uvicorn deployment.app.main:app --reload
```

Interactive docs available at `http://localhost:8000/docs`.

### Run with Docker

```bash
docker build -t neu-det-api .
docker run -p 8000:8000 neu-det-api
```

### Run on Kubernetes (kind)

```bash
# Create cluster
kind create cluster --name neu-det

# Load image into cluster
kind load docker-image neu-det-api --name neu-det

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Forward port
kubectl port-forward service/neu-det-api 8000:8000
```

---

## 🔌 API Endpoints

### `GET /health`
Health check for Kubernetes readiness probes.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /predict`
Classify a defect image — returns JSON with predicted class and confidence scores.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
```

```json
{
    "predicted_class": "scratches",
    "confidence": 1.0,
    "all_scores": {
        "scratches": 1.0,
        "crazing": 0.0,
        "inclusion": 0.0,
        "patches": 0.0,
        "pitted_surface": 0.0,
        "rolled-in_scale": 0.0
    }
}
```

### `POST /predict/visual`
Classify a defect image — returns a PNG with Grad-CAM heatmap overlay showing where the model focused.

```bash
curl -X POST http://localhost:8000/predict/visual \
  -F "file=@image.jpg" \
  --output heatmap.png
```

---

## 🔬 Key Findings

**Both architectures achieve 100% test accuracy** on NEU-DET, demonstrating that pretrained ImageNet features transfer effectively to steel surface defect classification.

**ViT achieves identical accuracy with 4,614 trainable parameters vs ResNet's 22.1M** — training only the classifier head on a frozen backbone is sufficient for this dataset, demonstrating strong transfer from ImageNet pretraining.

**Grad-CAM works for ResNet50 but not for frozen ViT, at least in this case.** ResNet produces spatially coherent heatmaps that align with ground truth defect regions. For ViT trained with feature extraction, both Grad-CAM and Attention Rollout produce uninformative maps — consistent with published findings that frozen transformer backbones do not develop task-specific spatial representations. Meaningful ViT interpretability would require full fine-tuning or a self-supervised backbone such as DINO.

**Grad-CAM heatmap quality varies by defect type.** Localised defects (scratches, rolled-in scale) produce tight, focused heatmaps. Distributed defects (crazing, pitted surface) produce diffuse heatmaps — correctly reflecting that the defect pattern spans the full image rather than a specific region.

---

## ⚠️ Known limitations

**No "no-defect" class.** The model always predicts one of 6 defect classes — it cannot identify clean parts. Production deployment would require either a 7th class with clean surface images or an anomaly detection component.

**ViT interpretability requires further work.** Feature extraction ViT does not produce meaningful spatial explanations. Full fine-tuning or a DINO backbone would be needed for comparable interpretability to ResNet50. If you have the time and interest to add this, please send a PR!

---

## 🔒 Notes

- `.venv/`, `data/`, and `experiments/` are gitignored
- Only code and configuration files are tracked
- Model weights are hosted on Hugging Face Hub
