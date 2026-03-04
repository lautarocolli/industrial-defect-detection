"""
PyTorch Dataset for the NEU Surface Defect Database.

Directory structure expected:
    data/splits/{split}/IMAGES/      ← .jpg images
    data/splits/{split}/ANNOTATIONS/ ← .xml Pascal VOC annotations

Label source:
    Class label and bounding boxes are both read from the XML annotation file.
    The <object><name> tag is the ground truth class — filename is never parsed.

Each sample always returns:
    image       FloatTensor  [3, 224, 224]
    label       int          class index
    boxes       FloatTensor  [N, 4]  (xmin, ymin, xmax, ymax)
    image_path  str          original path — used for Grad-CAM / debugging

Usage:
    from src.data.dataset import NEUDefectDataset, build_dataloaders
    from src.data.transforms import get_train_transforms, get_val_transforms

    train_ds = NEUDefectDataset(
        root      = Path("data/splits"),
        split     = "train",
        transform = get_train_transforms(),
    )

    print(train_ds.classes)         # ["crazing", "inclusion", ...]
    print(train_ds.class_to_idx)    # {"crazing": 0, "inclusion": 1, ...}
    print(train_ds[0])              # {"image": ..., "label": 0, "boxes": ..., "image_path": ...}
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Types ──────────────────────────────────────────────────────────────────
SplitName = Literal["train", "val", "test"]

# ══════════════════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════════════════

def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for NEU-DET detection batches.

    Why this is needed:
        PyTorch's default collate stacks all sample values into tensors.
        This works for images [3, 224, 224] and labels (int), but fails
        for bounding boxes because different images contain different numbers
        of defect instances — shapes [1, 4], [3, 4], [2, 4] cannot be stacked.

    Solution:
        Stack images and labels normally.
        Keep boxes as a list of tensors — one per image, no stacking.

    Args:
        batch: list of sample dicts from NEUDefectDataset.__getitem__

    Returns:
        {
            "image":      FloatTensor [batch_size, 3, 224, 224],
            "label":      LongTensor  [batch_size],
            "boxes":      list of FloatTensor, each [N_i, 4],
            "image_path": list of str,
        }
    """
    return {
        "image":      torch.stack([s["image"] for s in batch]),
        "label":      torch.tensor([s["label"] for s in batch], dtype=torch.long),
        "boxes":      [s["boxes"] for s in batch],
        "image_path": [s["image_path"] for s in batch],
    }

# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class NEUDefectDataset(Dataset):
    """
    PyTorch Dataset for NEU-DET surface defect detection.

    Args:
        root:      Path to the splits root directory (e.g. Path("data/splits"))
        split:     One of "train", "val", "test"
        transform: torchvision transform applied to each PIL image
    """

    def __init__(
        self,
        root:      Path,
        split:     SplitName,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        self.image_dir      = self.root / split / "IMAGES"
        self.annotation_dir = self.root / split / "ANNOTATIONS"

        self._validate_directories()

        # Sorted for determinism — index order is always identical across runs
        self.image_paths: list[Path] = sorted(self.image_dir.glob("*.jpg"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No .jpg images found in {self.image_dir}. "
                "Check your dataset split path."
            )

        # Class mapping built entirely from XML — single source of truth
        self.classes: list[str] = self._discover_classes()
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_directories(self) -> None:
        """Fail early with a clear message if expected directories are missing."""
        for directory in (self.image_dir, self.annotation_dir):
            if not directory.exists():
                raise FileNotFoundError(
                    f"Directory not found: {directory}\n"
                    f"Expected structure: {{root}}/{self.split}/IMAGES/ "
                    f"and {{root}}/{self.split}/ANNOTATIONS/"
                )

    # ── Class discovery ────────────────────────────────────────────────────

    def _discover_classes(self) -> list[str]:
        """
        Build a sorted list of unique class names by scanning all XML files.

        Reads the <object><name> tag from every annotation in this split.
        Sorted for determinism — class_to_idx is always identical across runs
        as long as every split contains the same set of defect classes.
        """
        classes: set[str] = set()

        for image_path in self.image_paths:
            xml_path = self.annotation_dir / f"{image_path.stem}.xml"
            if not xml_path.exists():
                raise FileNotFoundError(
                    f"Annotation file missing during class discovery: {xml_path}\n"
                    "Every image must have a matching .xml annotation file."
                )
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_tag = obj.find("name")
                if name_tag is not None and name_tag.text:
                    classes.add(name_tag.text.strip())

        if not classes:
            raise ValueError(
                f"No class names found in annotations at {self.annotation_dir}. "
                "Check that your XML files contain <object><name> tags."
            )

        return sorted(classes)

    # ── XML parsing ────────────────────────────────────────────────────────

    def _parse_annotation(self, image_path: Path) -> tuple[list[str], list[list[int]]]:
        """
        Parse a Pascal VOC XML file in a single pass.

        Extracts both class names and bounding boxes together to avoid
        parsing the same file twice.

        Returns:
            class_names: list of class name strings, one per object instance
            boxes:       list of [xmin, ymin, xmax, ymax] per object instance

        Both lists are parallel — class_names[i] corresponds to boxes[i].

        Raises:
            FileNotFoundError: if the .xml file does not exist
            ValueError:        if the file contains no valid objects
        """
        xml_path = self.annotation_dir / f"{image_path.stem}.xml"

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {xml_path}"
            )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        class_names: list[str]       = []
        boxes:       list[list[int]] = []

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox   = obj.find("bndbox")

            # Skip malformed objects missing either required tag
            if name_tag is None or bndbox is None:
                continue

            class_names.append(name_tag.text.strip())
            boxes.append([
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            ])

        if not boxes:
            raise ValueError(
                f"No valid objects found in {xml_path}. "
                "Check annotation file integrity."
            )

        return class_names, boxes

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single sample.

        Returns:
            {
                "image":       FloatTensor [3, 224, 224],
                "label":       int         — class index of first object
                "boxes":       FloatTensor [N, 4]  (xmin, ymin, xmax, ymax)
                "image_path":  str         — useful for Grad-CAM and debugging
            }

        Note on "label":
            NEU-DET images contain one defect class per image — all objects
            in a single image share the same class. Label is taken from the
            first object. If your dataset ever contains mixed-class images
            this assumption will need revisiting.
        """
        image_path = self.image_paths[idx]

        # Load as grayscale — transform handles L → RGB channel repetition
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)

        class_names, boxes = self._parse_annotation(image_path)

        # NEU-DET guarantee: all objects in one image share the same class
        label = self.class_to_idx[class_names[0]]

        return {
            "image":      image,
            "label":      label,
            "boxes":      torch.tensor(boxes, dtype=torch.float32),
            "image_path": str(image_path),
        }

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NEUDefectDataset("
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"n_classes={len(self.classes)}, "
            f"classes={self.classes}"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    root:            Path,
    train_transform: Optional[Callable] = None,
    val_transform:   Optional[Callable] = None,
    batch_size:      int  = 32,
    num_workers:     int  = 4,
    pin_memory:      bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from the split directory.

    Args:
        root:            Path to splits root (e.g. Path("data/splits"))
        train_transform: Augmented transforms for training set
        val_transform:   Deterministic transforms for val and test sets
        batch_size:      Samples per batch
        num_workers:     Parallel data loading workers
        pin_memory:      Page-lock memory for faster GPU transfer

    Returns:
        (train_loader, val_loader, test_loader)

    Example:
        from src.data.transforms import get_train_transforms, get_val_transforms

        train_loader, val_loader, test_loader = build_dataloaders(
            root            = Path("data/splits"),
            train_transform = get_train_transforms(),
            val_transform   = get_val_transforms(),
            batch_size      = 32,
        )
    """
    train_ds = NEUDefectDataset(root, split="train", transform=train_transform)
    val_ds   = NEUDefectDataset(root, split="val",   transform=val_transform)
    test_ds  = NEUDefectDataset(root, split="test",  transform=val_transform)

    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, (
        "Class mapping mismatch across splits. "
        "Ensure all splits contain at least one image from every defect class."
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
    )

    return train_loader, val_loader, test_loader