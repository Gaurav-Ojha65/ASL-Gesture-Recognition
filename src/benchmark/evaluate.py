"""
Evaluation script for the Sign Language MNIST benchmark CNN.

Loads the best validation checkpoint and evaluates on the held-out test set.
Produces:
  - reports/mnist_cnn_classification_report.md
  - reports/mnist_cnn_confusion_matrix.png
"""

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.benchmark.cnn import SignLanguageCNN
from src.benchmark.data_loader import get_dataloaders

# ── Paths ──
DATA_DIR = os.path.join("data", "raw", "sign_language_mnist")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")
CHECKPOINT_PATH = os.path.join("models", "mnist_cnn_best.pt")
REPORTS_DIR = "reports"

BATCH_SIZE = 64
SEED = 42

# Original label → ASL letter mapping (A=0 … but 9=J and 25=Z are absent)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Device ──
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Aborting.")
        sys.exit(1)
    device = torch.device("cuda")

    # ── Load checkpoint ──
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val accuracy : {checkpoint['val_acc']:.4f}")
    print(f"  Val loss     : {checkpoint['val_loss']:.4f}")

    index_to_label = checkpoint["index_to_label"]
    # Build class names from the mapping: index → original label → letter
    class_names = [ALPHABET[index_to_label[i]] for i in range(24)]

    # ── Model ──
    model = SignLanguageCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Test data ──
    _, _, test_loader, _, _, test_ds = get_dataloaders(
        TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE, val_split=0.2, random_seed=SEED
    )
    print(f"Test samples: {len(test_ds)}")

    # ── Inference ──
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ──
    test_acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    per_class_report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    print(f"\nTest Accuracy   : {test_acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print()
    print(per_class_report)

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(24),
        yticks=np.arange(24),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Sign Language MNIST — Confusion Matrix (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(24):
        for j in range(24):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "mnist_cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ── Find most confused pairs ──
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    top_confused_indices = np.unravel_index(
        np.argsort(cm_no_diag, axis=None)[-5:][::-1], cm_no_diag.shape
    )
    confused_pairs = []
    for i, j in zip(top_confused_indices[0], top_confused_indices[1]):
        confused_pairs.append((class_names[i], class_names[j], int(cm_no_diag[i, j])))

    # ── Markdown report ──
    md = f"""# Sign Language MNIST CNN — Classification Report

## Model
- Architecture: 3-block Conv2D CNN (32->64->128 channels, BN, ReLU, MaxPool) + FC head
- Checkpoint: `{CHECKPOINT_PATH}`
- Trained epoch: {checkpoint['epoch']}

## Test Set Results
- **Test Accuracy:** {test_acc:.4f} ({test_acc*100:.2f}%)
- **Macro Precision:** {macro_p:.4f}
- **Macro Recall:** {macro_r:.4f}
- **Macro F1:** {macro_f1:.4f}

## Per-Class Report
```
{per_class_report}
```

## Most Confused Class Pairs
| True | Predicted | Count |
|------|-----------|-------|
"""
    for true_cls, pred_cls, count in confused_pairs:
        md += f"| {true_cls} | {pred_cls} | {count} |\n"

    md += f"""
## Confusion Matrix
![Confusion Matrix](mnist_cnn_confusion_matrix.png)
"""

    report_path = os.path.join(REPORTS_DIR, "mnist_cnn_classification_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Classification report saved to {report_path}")

    return test_acc, macro_p, macro_r, macro_f1


if __name__ == "__main__":
    main()
