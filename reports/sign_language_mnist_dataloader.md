# Sign Language MNIST DataLoader Report

## Dataset Sizes
- **Train dataset size:** 21964
- **Validation dataset size:** 5491
- **Test dataset size:** 7172

## Configuration
- **Batch size:** 64
- **Random seed:** 42
- **Normalization method:** Divide by 255.0 to scale pixels to [0, 1]

## Tensor Shapes
- **Image tensor shape:** [64, 1, 28, 28]
- **Label shape:** [64]

## Verification
- **Pixel Range:** Min 0.00, Max 1.00
- **Train/Test Isolated:** True (Test CSV loaded independently)
- **Train/Validation Isolated:** True (Stratified split of Train CSV)
- **Tests Passed:** All shape, label range, and finite-value assertions passed.

## Label Mapping
The PyTorch `CrossEntropyLoss` requires contiguous class indices from `[0, C-1]`.
The original dataset skips J (9) and Z (25).
We implemented an internal mapping where original labels `[0-8, 10-24]` map directly to indices `[0-23]`.
This mapping is stored in `dataset.label_to_index` and `dataset.index_to_label`.
