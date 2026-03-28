"""
split_dataset.py
─────────────────────────────────────────────────────────────────────────────
Generate train, val and test splits for the NEU-DET dataset.

Destination:
    root/data/splits

You should end up with:
    root/data/splits/<train or val or test>/<IMAGES and ANNOTATIONS>/

Usage:
    From the project root:
    python -m src.data_operations.split_dataset
─────────────────────────────────────────────────────────────────────────────
"""

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed

set_seed(42)  # for reproducibility

# ── Paths ──────────────────────────────────────────────────────────────────
project_root    = Path(__file__).resolve().parent.parent.parent
base_dir        = project_root / "data" / "NEU-DET"
images_dir      = base_dir / "IMAGES"
annotations_dir = base_dir / "ANNOTATIONS"
output_dir      = project_root / "data" / "splits"

# ── Split ratios ───────────────────────────────────────────────────────────
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15

# Split names defined once — used in both the split loop and verification
splits = ["train", "val", "test"]

# ── Build class → file mapping ─────────────────────────────────────────────
class_files = defaultdict(list)

for img_path in images_dir.glob("*.jpg"):
    class_name = img_path.stem.split("_")[0]
    class_files[class_name].append(img_path)

# ── Split and copy files ───────────────────────────────────────────────────
for class_name, img_paths in class_files.items():

    train_imgs, temp_imgs = train_test_split(
        img_paths, test_size=(1 - train_ratio), random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=42
    )

    split_map = {
        "train": train_imgs,
        "val":   val_imgs,
        "test":  test_imgs,
    }

    for split_name, split_imgs in split_map.items():
        img_out_dir = output_dir / split_name / "IMAGES"
        ann_out_dir = output_dir / split_name / "ANNOTATIONS"

        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            # Copy image
            shutil.copy(img_path, img_out_dir / img_path.name)

            # Copy corresponding XML annotation
            xml_path = annotations_dir / (img_path.stem + ".xml")
            shutil.copy(xml_path, ann_out_dir / xml_path.name)

print("Dataset split completed.")

# ── Verification ───────────────────────────────────────────────────────────
print("\n===== DATASET SPLIT VERIFICATION =====\n")

split_dir    = project_root / "data" / "splits"
total_images = 0
split_counts = {}

for split in splits:
    images_path = split_dir / split / "IMAGES"
    ann_path    = split_dir / split / "ANNOTATIONS"

    images      = list(images_path.glob("*.jpg"))
    annotations = list(ann_path.glob("*.xml"))

    n_images      = len(images)
    n_annotations = len(annotations)

    split_counts[split] = n_images
    total_images       += n_images

    print(f"{split.upper()} SET")
    print(f"  Images:      {n_images}")
    print(f"  Annotations: {n_annotations}")

    if n_images != n_annotations:
        print("  ⚠️  Mismatch between images and annotations!")
    else:
        print("  ✓  Image and annotation counts match.")

    class_counter = Counter(img.stem.split("_")[0] for img in images)
    print(f"  Class distribution: {dict(class_counter)}")
    print("-" * 40)

print("\n===== RATIO CHECK =====")
for split in splits:
    ratio = split_counts[split] / total_images
    print(f"  {split:<8} {ratio:.2%}")

print(f"\nTotal images across all splits: {total_images}")