# Sign Language MNIST Benchmark — Independent Audit Report

---

## TASK 1 — CHECKPOINT AUDIT

---
Checkpoint file: models\mnist_cnn_best.pt
Keys in checkpoint: ['batch_size', 'epoch', 'index_to_label', 'input_shape', 'label_to_index', 'lr', 'model_state_dict', 'normalization', 'num_classes', 'optimizer_state_dict', 'random_seed', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
Best epoch: 10
Val accuracy: 1.000000
Val loss: 0.000333
Num classes: 24
Input shape: [1, 28, 28]
Normalization: divide_by_255
Random seed: 42
Batch size: 64
Learning rate: 0.001
label_to_index count: 24
index_to_label count: 24
Model parameters: 394,456
Forward pass output shape: torch.Size([64, 24]) [PASS]

---

## TASK 2 — TEST SET RE-EVALUATION

---
Test samples: 7172
Test Accuracy:    0.997490 (99.75%)
Macro Precision:  0.997349
Macro Recall:     0.997831
Macro F1:         0.997568
Incorrect predictions: 18 out of 7172

---

## TASK 3 — CROSS-SPLIT DUPLICATE AUDIT

---
Computing row hashes (this takes a moment)...
Train vs Validation overlap: 0
Train vs Test overlap:       0
Validation vs Test overlap:  0

---

## TASK 4 — LABEL INTEGRITY

---
label_to_index matches expected: True
index_to_label matches expected: True
Full mapping verification:
  Index  0 -> Original  0 -> Letter A [OK]
  Index  1 -> Original  1 -> Letter B [OK]
  Index  2 -> Original  2 -> Letter C [OK]
  Index  3 -> Original  3 -> Letter D [OK]
  Index  4 -> Original  4 -> Letter E [OK]
  Index  5 -> Original  5 -> Letter F [OK]
  Index  6 -> Original  6 -> Letter G [OK]
  Index  7 -> Original  7 -> Letter H [OK]
  Index  8 -> Original  8 -> Letter I [OK]
  Index  9 -> Original 10 -> Letter K [OK]
  Index 10 -> Original 11 -> Letter L [OK]
  Index 11 -> Original 12 -> Letter M [OK]
  Index 12 -> Original 13 -> Letter N [OK]
  Index 13 -> Original 14 -> Letter O [OK]
  Index 14 -> Original 15 -> Letter P [OK]
  Index 15 -> Original 16 -> Letter Q [OK]
  Index 16 -> Original 17 -> Letter R [OK]
  Index 17 -> Original 18 -> Letter S [OK]
  Index 18 -> Original 19 -> Letter T [OK]
  Index 19 -> Original 20 -> Letter U [OK]
  Index 20 -> Original 21 -> Letter V [OK]
  Index 21 -> Original 22 -> Letter W [OK]
  Index 22 -> Original 23 -> Letter X [OK]
  Index 23 -> Original 24 -> Letter Y [OK]
All letters correct: True

---

## TASK 5 — TRANSFORM AUDIT

---
Dataset normalization (all splits): pixel / 255.0 inside __getitem__ (always applied)
Dataset transform: applied via self.transform if provided

data_loader.py creates datasets WITHOUT passing any transform argument.
  -> train_dataset  = SignLanguageMNISTDataset(dataframe=...) [no transform]
  -> val_dataset    = SignLanguageMNISTDataset(dataframe=...) [no transform]
  -> test_dataset   = SignLanguageMNISTDataset(csv_file=...)  [no transform]

train.py applies augmentation at BATCH level in train_one_epoch():
  train_transform = Compose([
    RandomRotation(degrees=10),
    RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
  ])
  This is applied ONLY to training batches inside train_one_epoch().
  validate() does NOT apply any augmentation.

SUMMARY:
  TRAIN:      normalization (/255) + augmentation (rotation, affine)
  VALIDATION: normalization (/255) only — no augmentation [CORRECT]
  TEST:       normalization (/255) only — no augmentation [CORRECT]

---

## TASK 6 — TEST RESULT CONSISTENCY

---
Reported accuracy: 99.75%
Expected incorrect for 99.75%: 18 (theoretical)
Actual incorrect count: 18
Actual accuracy: 99.7490%
Confusion matrix off-diagonal sum: 18
Matches incorrect count: True

All misclassified pairs:
  True B -> Predicted U: 1
  True B -> Predicted W: 6
  True N -> Predicted M: 4
  True T -> Predicted X: 1
  True Y -> Predicted K: 6

---

## TASK 7 — REPRODUCIBILITY CHECK

---
Run 1 accuracy: 0.997490
Run 2 accuracy: 0.997490
Predictions identical: True
Labels identical: True
Reproducible: True

---

## AUDIT SUMMARY

---
Benchmark result independently reproduced: 0.9975 (99.75%)
Cross-split duplicates found: Train/Val=0, Train/Test=0, Val/Test=0
Label mapping correct: True
Augmentation isolated to training: True
99.75% accuracy reproducible: True
Incorrect predictions: 18
Concerns: 0
  None
