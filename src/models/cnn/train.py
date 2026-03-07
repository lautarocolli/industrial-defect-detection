"""
train.py
─────────────────────────────────────────────────────────────────────────────
PyTorch epoch pass.

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

import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()