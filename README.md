# industrial-defect-detection
Benchmarking CNNs and Vision Transformers for industrial surface defect classification with Grad-CAM interpretability.

# ⚙️ Technical Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/industrial-defect-detection.git
cd industrial-defect-detection
```

---

## 2️⃣ Create the Virtual Environment (uv)

This project uses **uv** for dependency management.

If you don’t have uv installed:

```bash
pip install uv
```

Create the virtual environment:

```bash
uv venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
uv add torch torchvision
uv add jupyter ipykernel matplotlib pandas scikit-learn tqdm pillow
```

This automatically updates `pyproject.toml`.

---

## 4️⃣ Register Jupyter Kernel

Register the environment so it appears in Jupyter:

```bash
python -m ipykernel install --user \
    --name industrial-defect-detection \
    --display-name "Python (industrial-defect-detection)"
```

Launch Jupyter:

```bash
jupyter notebook
```

Then select:

```
Python (industrial-defect-detection)
```

---

## 5️⃣ Download the Dataset (Kaggle API)

This project uses the **NEU Surface Defect Database** from Kaggle.

### Install Kaggle CLI (if needed)

```bash
uv add kaggle
```

### Configure API Access

1. Go to Kaggle → Account Settings  
2. Create a new API token  
3. Move `kaggle.json` to:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download the Dataset

From the project root:

```bash
mkdir data
cd data

kaggle datasets download -d rdsunday/neu-urface-defect-database
unzip neu-urface-defect-database.zip
rm neu-urface-defect-database.zip
```

---

## 6️⃣ Split the Dataset (Train / Val / Test)

Before running the training notebook, you must split the dataset into:

- **70% Training**
- **15% Validation**
- **15% Test**

This project includes a script that automatically splits both **images and annotations**.

> **What are annotations?**  
> Annotations are XML label files associated with each image.  
> They contain the ground-truth defect class and bounding box coordinates (xmin, ymin, xmax, ymax) that define where the defect appears in the image.  
> This information is used as the training target for the model.

The script is located at:

```
src/data-operations/split_dataset.py
```

### ▶️ Run the split script

From the project root directory:

```bash
python src/split_dataset.py
```

---

## 📁 Project Structure

```
industrial-defect-detection/
│
├── data/                     # Dataset (ignored by git)
├── notebooks/                # Training + experiments
├── src/                      # Model & utility code
├── pyproject.toml            # Dependencies (managed by uv)
├── uv.lock                   # Dependency lock file
└── README.md
```

---

## 🔒 Notes

- `.venv/` is ignored  
- `data/` is ignored  
- Only code and configuration files are tracked in Git

## Model Weights

Pretrained model weights are available on Hugging Face Hub:
[lautivuelos/neu-det-defect-detection](https://huggingface.co/lautivuelos/neu-det-defect-detection)