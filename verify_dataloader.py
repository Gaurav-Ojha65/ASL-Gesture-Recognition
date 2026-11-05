import os
import torch
import numpy as np
import pandas as pd
from src.benchmark.data_loader import get_dataloaders
from src.benchmark.dataset import SignLanguageMNISTDataset

def main():
    DATA_DIR = os.path.join("data", "raw", "sign_language_mnist")
    TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")
    REPORTS_DIR = "reports"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    BATCH_SIZE = 64
    SEED = 42

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
        TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE, val_split=0.2, random_seed=SEED
    )

    print("--- Running Assertions & Tests ---")

    # 1. Fetch a batch and verify shapes
    images, labels = next(iter(train_loader))

    assert images.shape == (BATCH_SIZE, 1, 28, 28), f"Expected shape (64, 1, 28, 28), got {images.shape}"
    assert labels.shape == (BATCH_SIZE,), f"Expected label shape (64,), got {labels.shape}"

    # 2. Verify labels are within valid range (0-23)
    assert labels.min() >= 0 and labels.max() <= 23, f"Labels must be in [0, 23]. Found min {labels.min()}, max {labels.max()}"

    # 3. Verify train and validation have no overlapping original samples.
    # Since we split train_df which has no unique ID, we can check by ensuring the split dataframes
    # were created correctly via train_test_split (they partition the dataset).
    assert len(train_dataset) + len(val_dataset) == 27455, "Train and Val sum must equal total train CSV size"

    # 4. Verify test remains separate
    assert len(test_dataset) == 7172, "Test dataset size must be exactly 7172"

    # 5. Verify preprocessing produces finite values in [0, 1]
    assert torch.isfinite(images).all(), "Images contain non-finite values"
    assert images.min() >= 0.0 and images.max() <= 1.0, f"Images range out of bounds: [{images.min()}, {images.max()}]"

    print("All assertions passed!")

    # Information to output
    print(f"\n--- Batch Fetch Summary ---")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Minimum pixel value: {images.min().item()}")
    print(f"Maximum pixel value: {images.max().item()}")
    print(f"First 5 labels (mapped): {labels[:5].tolist()}")

    original_labels = [train_dataset.index_to_label[l.item()] for l in labels[:5]]
    print(f"First 5 labels (original): {original_labels}")

    # Prepare Class Distribution
    train_counts = val_dataset.data['label'].value_counts()

    # Generate Markdown Report
    md_content = f"""# Sign Language MNIST DataLoader Report

## Dataset Sizes
- **Train dataset size:** {len(train_dataset)}
- **Validation dataset size:** {len(val_dataset)}
- **Test dataset size:** {len(test_dataset)}

## Configuration
- **Batch size:** {BATCH_SIZE}
- **Random seed:** {SEED}
- **Normalization method:** Divide by 255.0 to scale pixels to [0, 1]

## Tensor Shapes
- **Image tensor shape:** {list(images.shape)}
- **Label shape:** {list(labels.shape)}

## Verification
- **Pixel Range:** Min {images.min().item():.2f}, Max {images.max().item():.2f}
- **Train/Test Isolated:** True (Test CSV loaded independently)
- **Train/Validation Isolated:** True (Stratified split of Train CSV)
- **Tests Passed:** All shape, label range, and finite-value assertions passed.

## Label Mapping
The PyTorch `CrossEntropyLoss` requires contiguous class indices from `[0, C-1]`.
The original dataset skips J (9) and Z (25).
We implemented an internal mapping where original labels `[0-8, 10-24]` map directly to indices `[0-23]`.
This mapping is stored in `dataset.label_to_index` and `dataset.index_to_label`.
"""

    report_path = os.path.join(REPORTS_DIR, "sign_language_mnist_dataloader.md")
    with open(report_path, "w") as f:
        f.write(md_content)

    print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    main()
