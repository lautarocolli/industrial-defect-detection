"""
Image transforms for the NEU-DET defect detection pipeline.

Augmentation choices for NEU-DET specifically:
  - Horizontal + vertical flips: safe — defects are orientation-independent
  - Random rotation (±15°): safe — surface textures have no canonical orientation
  - Random affine shear: adds mild geometric diversity without removing defects
  - Colour jitter: intentionally OMITTED — images are greyscale steel scans,
    colour variation adds no meaningful signal
  - Heavy random crops: intentionally OMITTED — defect regions can be small,
    aggressive cropping risks removing the defect entirely
"""

from torchvision import transforms

# ── ImageNet statistics ────────────────────────────────────────────────────
# Used because ResNet50 weights are pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Target size ────────────────────────────────────────────────────────────
# ResNet50 was designed for 224×224 input.
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with augmentation.

    Pipeline:
        1. Resize  — upsample 200×200 → 224×224
        2. Grayscale → RGB — repeat single channel 3×
        3. Random horizontal flip — p=0.5
        4. Random vertical flip   — p=0.5
        5. Random rotation        — ±15 degrees
        6. Random affine shear    — mild geometric diversity
        7. ToTensor               — HWC uint8 → CHW float32 in [0, 1]
        8. Normalise              — ImageNet mean/std

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validation and test transforms — deterministic, no augmentation.

    Pipeline:
        1. Resize      — 200×200 → 224×224
        2. Grayscale → RGB
        3. ToTensor
        4. Normalise   — ImageNet mean/std

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Convenience alias ──────────────────────────────────────────────────────
# Val and test share the same transform — aliased here for clarity at the
get_test_transforms = get_val_transforms