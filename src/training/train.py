"""
train.py
─────────────────────────────────────────────────────────────────────────────
Training and evaluation functions for NEU-DET defect classification.

train()    — runs the full training loop across all epochs, printing
             loss and accuracy for both training and validation sets,
             and returning a history dict for convergence analysis.

evaluate() — runs a single pass over any DataLoader with no weight updates.
             Used for final test set evaluation after training is complete.

Both functions are model-agnostic — they work with any nn.Module that
accepts [batch, 3, 224, 224] input and returns [batch, num_classes] logits.
Used for both ResNet50 and ViT-B/16.

Usage:
    from src.training.train import train, evaluate
    from src.models.cnn.resnet import build_resnet

    model = build_resnet(num_classes=6)

    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = 20,
    )

    # history is a dict of lists — one value per epoch
    # keys: "epoch", "train_loss", "train_acc", "val_loss", "val_acc"

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
        model:        nn.Module — the model to train, works for ResNet50 and ViT
        train_loader: DataLoader — training set
        val_loader:   DataLoader — validation set
        num_epochs:   int — number of full passes over the training data

    Returns:
        history: dict of lists with keys:
            "epoch"      — epoch numbers [1, 2, ..., num_epochs]
            "train_loss" — average training loss per epoch
            "train_acc"  — training accuracy per epoch
            "val_loss"   — average validation loss per epoch
            "val_acc"    — validation accuracy per epoch

        Use history to plot convergence curves and compare architectures:

            import matplotlib.pyplot as plt
            plt.plot(history["epoch"], history["train_loss"], label="train")
            plt.plot(history["epoch"], history["val_loss"],   label="val")
            plt.legend()
            plt.show()

        Or convert to DataFrame for tabular inspection:

            import pandas as pd
            df = pd.DataFrame(history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Only pass trainable parameters to the optimiser
    # Frozen parameters have requires_grad=False — Adam skips them anyway,
    # but filtering explicitly makes the intent clear
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    # Initialise history dict — lists are populated inside the epoch loop
    # One value appended per epoch for each metric
    history = {
        "epoch":      [],
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    for epoch in range(num_epochs):
        # ── Training ───────────────────────────────────────────────────────
        # model.train() enables dropout and BatchNorm training behaviour
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Zero gradients — PyTorch accumulates by default
            optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()

            # Accumulate metrics — .item() converts tensor to Python float
            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        # ── Validation ─────────────────────────────────────────────────────
        # model.eval() disables dropout and uses stable BatchNorm statistics
        model.eval()
        val_running_loss = 0.0
        val_correct      = 0
        val_total        = 0

        # torch.no_grad() disables gradient computation — faster and
        # uses less memory since no computation graph is built
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(inputs)
                loss    = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted      = torch.max(outputs.data, 1)
                val_total        += labels.size(0)
                val_correct      += (predicted == labels).sum().item()

        # ── Epoch summary ──────────────────────────────────────────────────
        train_acc  = correct / total
        train_loss = running_loss / len(train_loader)
        val_acc    = val_correct / val_total
        val_loss   = val_running_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Append this epoch's metrics to history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def evaluate(model, loader):
    """
    Evaluate a trained model on any DataLoader without updating weights.

    Used for final test set evaluation after training is complete.
    No optimizer needed — this function never calls loss.backward().

    Args:
        model:  nn.Module  — trained model to evaluate, works for ResNet50 and ViT
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

    # Eval mode — disables dropout, uses stable BatchNorm statistics
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

    acc  = correct / total
    loss = running_loss / len(loader)

    print(f"loss: {loss:.4f}  acc: {acc:.4f}")

    return loss, acc