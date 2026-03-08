"""
train.py
─────────────────────────────────────────────────────────────────────────────
Training and evaluation functions for NEU-DET defect classification.

train()    — runs the full training loop across all epochs, printing
             loss and accuracy for both training and validation sets.

evaluate() — runs a single pass over any DataLoader with no weight updates.
             Used for final test set evaluation after training is complete.

Usage:
    from src.models.cnn.train import train, evaluate
    from src.models.cnn.resnet import build_resnet

    model = build_resnet(num_classes=6)

    train(
        model       = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = 20,
    )

    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, val_loader, num_epochs):
    """
    Train a model for a fixed number of epochs with validation after each epoch.

    Args:
        model:        nn.Module — the model to train (e.g. ResNet50Classifier)
        train_loader: DataLoader — training set
        val_loader:   DataLoader — validation set
        num_epochs:   int — number of full passes over the training data

    Prints loss and accuracy for both train and val after every epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # ── Training ───────────────────────────────────────────────────────
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

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # ── Epoch summary ──────────────────────────────────────────────────
        train_acc  = correct / total
        train_loss = running_loss / len(train_loader)
        val_acc    = val_correct / val_total
        val_loss   = val_running_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")


def evaluate(model, loader):
    """
    Evaluate a trained model on any DataLoader without updating weights.

    Used for final test set evaluation after training is complete.

    Args:
        model:  nn.Module  — trained model to evaluate
        loader: DataLoader — any split (typically test_loader)

    Returns:
        loss: float — average loss per batch
        acc:  float — accuracy across all samples

    Example:
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc  = correct / total
    loss = running_loss / len(loader)

    print(f"loss: {loss:.4f}  acc: {acc:.4f}")

    return loss, acc