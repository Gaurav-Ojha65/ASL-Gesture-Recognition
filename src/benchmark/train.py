"""
Training script for the Sign Language MNIST benchmark CNN.

Usage:
    python -m src.benchmark.train          # from project root
    .venv/Scripts/python src/benchmark/train.py   # direct

Trains on the training split, validates on the validation split,
saves the best checkpoint to models/mnist_cnn_best.pt,
and plots training curves to reports/.
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── project imports ──
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.benchmark.cnn import SignLanguageCNN, count_parameters
from src.benchmark.data_loader import get_dataloaders

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
SEED = 42
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
VAL_SPLIT = 0.2

DATA_DIR = os.path.join("data", "raw", "sign_language_mnist")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")
CHECKPOINT_DIR = "models"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mnist_cnn_best.pt")
REPORTS_DIR = "reports"

# Expected label set from STEP 3.2 (original labels, not contiguous indices)
EXPECTED_ORIGINAL_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


# ══════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
# Data augmentation (training only)
# ══════════════════════════════════════════════════════════════════════
train_transform = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])


# ══════════════════════════════════════════════════════════════════════
# Train / validate one epoch
# ══════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Apply training augmentation on GPU tensors
        augmented = train_transform(images)

        optimizer.zero_grad()
        outputs = model(augmented)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Device ──
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Aborting — will not silently fall back to CPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Device selected : {device}")
    print()

    # ── Data ──
    print("Loading data...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, random_seed=SEED
    )
    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    print(f"Test samples  : {len(test_ds)}")
    print()

    # ── Model ──
    model = SignLanguageCNN().to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(model)
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ══════════════════════════════════════════════════════════════════
    # Sanity checks
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1 — Forward pass shape check
    sample_images, sample_labels = next(iter(train_loader))
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    with torch.no_grad():
        sample_out = model(sample_images)
    assert sample_out.shape == (sample_images.size(0), 24), (
        f"Expected output shape ({sample_images.size(0)}, 24), got {sample_out.shape}"
    )
    print(f"[PASS] Forward pass output shape: {sample_out.shape}")

    # 2 — Loss calculation
    loss_check = criterion(sample_out, sample_labels)
    assert torch.isfinite(loss_check), f"Loss is not finite: {loss_check.item()}"
    print(f"[PASS] Loss calculation works: {loss_check.item():.4f}")

    # 3 — Smoke-test: one mini-epoch
    model.train()
    optimizer.zero_grad()
    out = model(train_transform(sample_images))
    loss = criterion(out, sample_labels)
    loss.backward()
    optimizer.step()
    print(f"[PASS] Smoke-test training step completed, loss: {loss.item():.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # Full training loop
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"TRAINING — {EPOCHS} epochs")
    print("=" * 60)

    # Re-initialize for a clean run
    seed_everything(SEED)
    model = SignLanguageCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        # Save best checkpoint based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "label_to_index": train_ds.label_to_index,
                "index_to_label": train_ds.index_to_label,
                "num_classes": SignLanguageCNN.NUM_CLASSES,
                "input_shape": [1, 28, 28],
                "normalization": "divide_by_255",
                "random_seed": SEED,
                "batch_size": BATCH_SIZE,
                "lr": LR,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)

    total_time = time.time() - total_start
    avg_epoch_time = total_time / EPOCHS

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best epoch        : {best_epoch}")
    print(f"Best val accuracy : {best_val_acc:.4f}")
    print(f"Best val loss     : {best_val_loss:.4f}")
    print(f"Total time        : {total_time:.1f}s")
    print(f"Avg epoch time    : {avg_epoch_time:.1f}s")
    print(f"Checkpoint saved  : {CHECKPOINT_PATH}")

    # ── Plot training curves ──
    curves_path = os.path.join(REPORTS_DIR, "mnist_cnn_training_curves.png")
    plot_training_curves(history, curves_path)

    return best_val_acc, best_val_loss, best_epoch, total_time, avg_epoch_time, num_params


if __name__ == "__main__":
    main()
