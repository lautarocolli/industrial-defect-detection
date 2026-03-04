"""
dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset for the NEU Surface Defect Database.

Task: Classification — predict defect class from image.

Directory structure expected:
    data/splits/{split}/IMAGES/      ← .jpg images
    data/splits/{split}/ANNOTATIONS/ ← .xml Pascal VOC annotations

Label source:
    Class label is read from the <object><name> tag in the XML file.
    Filename is never parsed.

Each training batch contains:
    image       FloatTensor  [3, 224, 224]
    label       int          class index
    image_path  str          original path — used to look up boxes for Grad-CAM

Boxes are loaded separately at visualisation time:
    dataset.get_boxes(image_path) → FloatTensor [N, 4] (xmin, ymin, xmax, ymax)

Usage:
    from src.data_operations.dataset import NEUDefectDataset, build_dataloaders
    from src.data_operations.transforms import get_train_transforms, get_val_transforms

    train_loader, val_loader, test_loader = build_dataloaders(
        root            = Path("data/splits"),
        train_transform = get_train_transforms(),
        val_transform   = get_val_transforms(),
        batch_size      = 32,
    )

    # Training batch
    batch = next(iter(train_loader))
    batch["image"].shape      # [32, 3, 224, 224]
    batch["label"].shape      # [32]
    batch["image_path"][0]    # "data/splits/train/IMAGES/crazing_1.jpg"

    # Grad-CAM visualisation — load boxes for a specific image
    boxes = train_loader.dataset.get_boxes("data/splits/train/IMAGES/crazing_1.jpg")
    # FloatTensor [[xmin, ymin, xmax, ymax], ...]
─────────────────────────────────────────────────────────────────────────────
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
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class NEUDefectDataset(Dataset):
    """
    PyTorch Dataset for NEU-DET surface defect classification.

    Predicts defect class from image. Bounding boxes are available on demand
    via get_boxes() for Grad-CAM overlay.

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

    def _parse_annotation(self, image_path: Path) -> tuple[str, list[list[int]]]:
        """
        Parse a Pascal VOC XML file and return class name + bounding boxes.

        Both are extracted in a single pass to avoid reading the file twice.

        Returns:
            class_name: defect class of this image (first object's name)
            boxes:      list of [xmin, ymin, xmax, ymax] per defect instance

        Note:
            NEU-DET images contain one defect class per image — all objects
            share the same class. class_name is taken from the first object.
            If your dataset ever contains mixed-class images this will need
            revisiting.

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

        class_name: Optional[str]    = None
        boxes: list[list[int]]       = []

        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox   = obj.find("bndbox")

            # Skip malformed objects missing either required tag
            if name_tag is None or bndbox is None:
                continue

            # Take class name from the first valid object
            if class_name is None:
                class_name = name_tag.text.strip()

            boxes.append([
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            ])

        if class_name is None or not boxes:
            raise ValueError(
                f"No valid objects found in {xml_path}. "
                "Check annotation file integrity."
            )

        return class_name, boxes

    # ── Public box access for Grad-CAM ─────────────────────────────────────

    def get_boxes(self, image_path: str) -> torch.Tensor:
        """
        Load bounding boxes for a single image at visualisation time.

        Called by the Grad-CAM pipeline to overlay ground truth defect
        regions on top of activation heatmaps. Not used during training.

        Boxes are rescaled from original 200×200 image space to the
        224×224 input space used by the model.

        Args:
            image_path: path string as stored in batch["image_path"]

        Returns:
            FloatTensor [N, 4] — (xmin, ymin, xmax, ymax) in 224×224 space

        Example:
            boxes = dataset.get_boxes(batch["image_path"][0])
        """
        scale = 224 / 200   # original → model input space

        _, boxes = self._parse_annotation(Path(image_path))

        scaled = [
            [
                int(x * scale) for x in box
            ]
            for box in boxes
        ]

        return torch.tensor(scaled, dtype=torch.float32)

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single training sample.

        Returns:
            {
                "image":      FloatTensor [3, 224, 224],
                "label":      int — class index,
                "image_path": str — used to retrieve boxes for Grad-CAM,
            }
        """
        image_path = self.image_paths[idx]

        # Load as grayscale — transform handles L → RGB channel repetition
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)

        class_name, _ = self._parse_annotation(image_path)
        label         = self.class_to_idx[class_name]

        return {
            "image":      image,
            "label":      label,
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
        from src.data_operations.transforms import get_train_transforms, get_val_transforms

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

    # Guard against split integrity issues — all splits must share the same classes
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    return train_loader, val_loader, test_loader