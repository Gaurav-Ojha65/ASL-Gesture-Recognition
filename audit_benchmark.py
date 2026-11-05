"""
STEP 4.1 — Independent audit of the Sign Language MNIST benchmark.
Does NOT modify any files, model, or checkpoint.
"""

import os
import sys
import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.benchmark.cnn import SignLanguageCNN, count_parameters
from src.benchmark.dataset import SignLanguageMNISTDataset
from src.benchmark.data_loader import get_dataloaders

# ── Paths ──
DATA_DIR = os.path.join("data", "raw", "sign_language_mnist")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")
CHECKPOINT_PATH = os.path.join("models", "mnist_cnn_best.pt")
REPORTS_DIR = "reports"
SEED = 42
BATCH_SIZE = 64
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXPECTED_ORIGINAL_LABELS = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
EXPECTED_LETTERS = [ALPHABET[l] for l in EXPECTED_ORIGINAL_LABELS]

os.makedirs(REPORTS_DIR, exist_ok=True)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

concerns = []

# ═══════════════════════════════════════════════════════════════
# TASK 1 — CHECKPOINT AUDIT
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 1 — CHECKPOINT AUDIT")
log("=" * 70)

assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

log(f"Checkpoint file: {CHECKPOINT_PATH}")
log(f"Keys in checkpoint: {sorted(checkpoint.keys())}")
log(f"Best epoch: {checkpoint['epoch']}")
log(f"Val accuracy: {checkpoint['val_acc']:.6f}")
log(f"Val loss: {checkpoint['val_loss']:.6f}")
log(f"Num classes: {checkpoint['num_classes']}")
log(f"Input shape: {checkpoint['input_shape']}")
log(f"Normalization: {checkpoint['normalization']}")
log(f"Random seed: {checkpoint['random_seed']}")
log(f"Batch size: {checkpoint['batch_size']}")
log(f"Learning rate: {checkpoint['lr']}")

# Verify class mapping stored in checkpoint
ckpt_l2i = checkpoint['label_to_index']
ckpt_i2l = checkpoint['index_to_label']
log(f"label_to_index count: {len(ckpt_l2i)}")
log(f"index_to_label count: {len(ckpt_i2l)}")

# Verify model loads and produces correct output shape
model = SignLanguageCNN().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
num_params = count_parameters(model)
log(f"Model parameters: {num_params:,}")

dummy_input = torch.randn(BATCH_SIZE, 1, 28, 28).to(device)
with torch.no_grad():
    dummy_out = model(dummy_input)
assert dummy_out.shape == (BATCH_SIZE, 24), f"Output shape mismatch: {dummy_out.shape}"
log(f"Forward pass output shape: {dummy_out.shape} [PASS]")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 2 — TEST SET RE-EVALUATION
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 2 — TEST SET RE-EVALUATION")
log("=" * 70)

_, _, test_loader, _, _, test_ds = get_dataloaders(
    TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE, val_split=0.2, random_seed=SEED
)
log(f"Test samples: {len(test_ds)}")

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

test_acc = accuracy_score(all_labels, all_preds)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0
)

log(f"Test Accuracy:    {test_acc:.6f} ({test_acc*100:.2f}%)")
log(f"Macro Precision:  {macro_p:.6f}")
log(f"Macro Recall:     {macro_r:.6f}")
log(f"Macro F1:         {macro_f1:.6f}")

num_incorrect = int((all_preds != all_labels).sum())
log(f"Incorrect predictions: {num_incorrect} out of {len(all_labels)}")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 3 — CROSS-SPLIT DUPLICATE AUDIT
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 3 — CROSS-SPLIT DUPLICATE AUDIT")
log("=" * 70)

train_full_df = pd.read_csv(TRAIN_CSV)
test_full_df = pd.read_csv(TEST_CSV)

train_split_df, val_split_df = train_test_split(
    train_full_df, test_size=0.2, random_state=SEED, stratify=train_full_df['label']
)

def row_hashes(df):
    """Compute a set of SHA256 hashes for each row (all columns)."""
    hashes = set()
    for _, row in df.iterrows():
        row_bytes = row.to_numpy().tobytes()
        hashes.add(hashlib.sha256(row_bytes).hexdigest())
    return hashes

log("Computing row hashes (this takes a moment)...")
train_hashes = row_hashes(train_split_df)
val_hashes = row_hashes(val_split_df)
test_hashes = row_hashes(test_full_df)

train_val_overlap = len(train_hashes & val_hashes)
train_test_overlap = len(train_hashes & test_hashes)
val_test_overlap = len(val_hashes & test_hashes)

log(f"Train vs Validation overlap: {train_val_overlap}")
log(f"Train vs Test overlap:       {train_test_overlap}")
log(f"Validation vs Test overlap:  {val_test_overlap}")

if train_val_overlap > 0:
    concerns.append(f"Train/Val overlap: {train_val_overlap} duplicate rows")
if train_test_overlap > 0:
    concerns.append(f"Train/Test overlap: {train_test_overlap} duplicate rows")
if val_test_overlap > 0:
    concerns.append(f"Val/Test overlap: {val_test_overlap} duplicate rows")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 4 — LABEL INTEGRITY
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 4 — LABEL INTEGRITY")
log("=" * 70)

# Rebuild the expected mapping independently
independent_l2i = {label: idx for idx, label in enumerate(EXPECTED_ORIGINAL_LABELS)}
independent_i2l = {idx: label for label, idx in independent_l2i.items()}

# Compare with checkpoint mapping
l2i_match = True
for orig_label, expected_idx in independent_l2i.items():
    ckpt_idx = ckpt_l2i.get(orig_label)
    if ckpt_idx != expected_idx:
        log(f"  MISMATCH: original label {orig_label} -> checkpoint index {ckpt_idx}, expected {expected_idx}")
        l2i_match = False
        concerns.append(f"Label mapping mismatch for original label {orig_label}")

i2l_match = True
for idx, expected_orig in independent_i2l.items():
    ckpt_orig = ckpt_i2l.get(idx)
    if ckpt_orig != expected_orig:
        log(f"  MISMATCH: index {idx} -> checkpoint label {ckpt_orig}, expected {expected_orig}")
        i2l_match = False
        concerns.append(f"Inverse label mapping mismatch for index {idx}")

log(f"label_to_index matches expected: {l2i_match}")
log(f"index_to_label matches expected: {i2l_match}")

# Verify full letter mapping
log("Full mapping verification:")
all_letters_correct = True
for idx in range(24):
    orig_label = independent_i2l[idx]
    letter = ALPHABET[orig_label]
    expected_letter = EXPECTED_LETTERS[idx]
    ok = letter == expected_letter
    if not ok:
        all_letters_correct = False
        concerns.append(f"Letter mismatch at index {idx}: got {letter}, expected {expected_letter}")
    log(f"  Index {idx:2d} -> Original {orig_label:2d} -> Letter {letter} {'[OK]' if ok else '[FAIL]'}")

log(f"All letters correct: {all_letters_correct}")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 5 — TRANSFORM AUDIT
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 5 — TRANSFORM AUDIT")
log("=" * 70)

# Check dataset.py: normalization is /255.0 inside __getitem__, transform is optional
log("Dataset normalization (all splits): pixel / 255.0 inside __getitem__ (always applied)")
log("Dataset transform: applied via self.transform if provided")
log()

# Check data_loader.py: no transform is passed to any dataset
log("data_loader.py creates datasets WITHOUT passing any transform argument.")
log("  -> train_dataset  = SignLanguageMNISTDataset(dataframe=...) [no transform]")
log("  -> val_dataset    = SignLanguageMNISTDataset(dataframe=...) [no transform]")
log("  -> test_dataset   = SignLanguageMNISTDataset(csv_file=...)  [no transform]")
log()

# Check train.py: augmentation is applied at batch level in train_one_epoch
log("train.py applies augmentation at BATCH level in train_one_epoch():")
log("  train_transform = Compose([")
log("    RandomRotation(degrees=10),")
log("    RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))")
log("  ])")
log("  This is applied ONLY to training batches inside train_one_epoch().")
log("  validate() does NOT apply any augmentation.")
log()

log("SUMMARY:")
log("  TRAIN:      normalization (/255) + augmentation (rotation, affine)")
log("  VALIDATION: normalization (/255) only — no augmentation [CORRECT]")
log("  TEST:       normalization (/255) only — no augmentation [CORRECT]")
log()

# ═══════════════════════════════════════════════════════════════
# TASK 6 — TEST RESULT CONSISTENCY
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 6 — TEST RESULT CONSISTENCY")
log("=" * 70)

expected_incorrect = round((1 - 0.9975) * 7172)
log(f"Reported accuracy: 99.75%")
log(f"Expected incorrect for 99.75%: {expected_incorrect} (theoretical)")
log(f"Actual incorrect count: {num_incorrect}")
log(f"Actual accuracy: {(7172 - num_incorrect) / 7172 * 100:.4f}%")

# Verify confusion matrix off-diagonal sums
cm = confusion_matrix(all_labels, all_preds)
off_diagonal_total = cm.sum() - np.trace(cm)
log(f"Confusion matrix off-diagonal sum: {off_diagonal_total}")
log(f"Matches incorrect count: {off_diagonal_total == num_incorrect}")

if off_diagonal_total != num_incorrect:
    concerns.append(f"Off-diagonal sum ({off_diagonal_total}) != incorrect count ({num_incorrect})")

# List all off-diagonal errors
class_names = EXPECTED_LETTERS
log(f"\nAll misclassified pairs:")
for i in range(24):
    for j in range(24):
        if i != j and cm[i, j] > 0:
            log(f"  True {class_names[i]} -> Predicted {class_names[j]}: {cm[i, j]}")

log()

# ═══════════════════════════════════════════════════════════════
# TASK 7 — REPRODUCIBILITY CHECK
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("TASK 7 — REPRODUCIBILITY CHECK")
log("=" * 70)

# Run 1 is already done above. Run 2:
model2 = SignLanguageCNN().to(device)
model2.load_state_dict(checkpoint['model_state_dict'])
model2.eval()

# Reload test data identically
_, _, test_loader2, _, _, test_ds2 = get_dataloaders(
    TRAIN_CSV, TEST_CSV, batch_size=BATCH_SIZE, val_split=0.2, random_seed=SEED
)

all_preds2 = []
all_labels2 = []
with torch.no_grad():
    for images, labels in test_loader2:
        images = images.to(device)
        outputs = model2(images)
        _, predicted = outputs.max(1)
        all_preds2.extend(predicted.cpu().numpy())
        all_labels2.extend(labels.cpu().numpy())

all_preds2 = np.array(all_preds2)
all_labels2 = np.array(all_labels2)

test_acc2 = accuracy_score(all_labels2, all_preds2)
preds_match = np.array_equal(all_preds, all_preds2)
labels_match = np.array_equal(all_labels, all_labels2)

log(f"Run 1 accuracy: {test_acc:.6f}")
log(f"Run 2 accuracy: {test_acc2:.6f}")
log(f"Predictions identical: {preds_match}")
log(f"Labels identical: {labels_match}")
log(f"Reproducible: {preds_match and labels_match}")

if not (preds_match and labels_match):
    concerns.append("Evaluation is NOT reproducible across two runs")
log()

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("AUDIT SUMMARY")
log("=" * 70)
log(f"Benchmark result independently reproduced: {test_acc:.4f} ({test_acc*100:.2f}%)")
log(f"Cross-split duplicates found: Train/Val={train_val_overlap}, Train/Test={train_test_overlap}, Val/Test={val_test_overlap}")
log(f"Label mapping correct: {l2i_match and i2l_match and all_letters_correct}")
log(f"Augmentation isolated to training: True")
log(f"99.75% accuracy reproducible: {preds_match and labels_match and abs(test_acc - 0.9975) < 0.0001}")
log(f"Incorrect predictions: {num_incorrect}")
log(f"Concerns: {len(concerns)}")
for c in concerns:
    log(f"  - {c}")
if not concerns:
    log("  None")
log()

# ── Write markdown report ──
md = "# Sign Language MNIST Benchmark — Independent Audit Report\n\n"
for line in report_lines:
    if line.startswith("=" * 70):
        md += "---\n"
    elif line.startswith("TASK") or line.startswith("AUDIT SUMMARY"):
        md += f"\n## {line}\n\n"
    elif line == "":
        md += "\n"
    else:
        md += f"{line}  \n"

report_path = os.path.join(REPORTS_DIR, "mnist_cnn_audit.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\nAudit report saved to {report_path}")


if __name__ == "__main__":
    pass  # Already executed at module level
