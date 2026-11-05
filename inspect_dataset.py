import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
DATA_DIR = os.path.join("data", "raw", "sign_language_mnist")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sign_mnist_test.csv")
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Verify existence
assert os.path.exists(TRAIN_CSV), f"{TRAIN_CSV} not found"
assert os.path.exists(TEST_CSV), f"{TEST_CSV} not found"

# 2. Load data
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# 3 & 4. Rows, columns, and structure
train_rows, train_cols = train_df.shape
test_rows, test_cols = test_df.shape

label_col = 'label'
pixel_cols = [c for c in train_df.columns if c != label_col]
num_pixel_cols = len(pixel_cols)

# 5 & 6. Verify 1 label, 784 pixels and data types
assert label_col in train_df.columns
assert num_pixel_cols == 784

pixel_dtypes = train_df[pixel_cols].dtypes.unique()

# 7. Pixel statistics
train_pixels = train_df[pixel_cols].values
pixel_min = train_pixels.min()
pixel_max = train_pixels.max()
pixel_mean = train_pixels.mean()
pixel_std = train_pixels.std()

# 8. Missing values
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()

# 9. Duplicate rows
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

# 10, 11, 12, 13, 18. Label analysis
train_labels = train_df['label'].unique()
train_labels.sort()
test_labels = test_df['label'].unique()
test_labels.sort()

train_label_counts = train_df['label'].value_counts().sort_index()

# Determine letters
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_to_letter = {i: alphabet[i] for i in range(26)}
found_letters = {label: label_to_letter[label] for label in train_labels}

has_J = 9 in train_labels
has_Z = 25 in train_labels

# 14. Reshape verification
try:
    reshaped_data = train_pixels.reshape(-1, 28, 28)
    reshape_success = True
    image_dims = reshaped_data.shape[1:]
except Exception as e:
    reshape_success = False
    image_dims = None

# 15, 17. Image grid visualization
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
axes = axes.flatten()
for i, label in enumerate(train_labels[:24]): # Display up to 24 unique classes
    sample_img = train_df[train_df['label'] == label].iloc[0][pixel_cols].values.reshape(28, 28)
    axes[i].imshow(sample_img, cmap='gray')
    axes[i].set_title(f"Label {label} ({label_to_letter[label]})")
    axes[i].axis('off')
for j in range(len(train_labels), 24):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "sign_language_mnist_samples.png"))
plt.close()

# 16. Class distribution visualization
plt.figure(figsize=(10, 6))
train_label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Sign Language MNIST Class Distribution (Training)")
plt.xlabel("Label (Letter)")
plt.ylabel("Number of Samples")
plt.xticks(ticks=range(len(train_label_counts)), labels=[f"{k} ({label_to_letter[k]})" for k in train_label_counts.index], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "sign_language_mnist_class_distribution.png"))
plt.close()

# Write Markdown report
md_content = f"""# Sign Language MNIST Dataset Inspection Report

## File Verification
- Training file: `{TRAIN_CSV}` (Exists: {os.path.exists(TRAIN_CSV)})
- Testing file: `{TEST_CSV}` (Exists: {os.path.exists(TEST_CSV)})

## Dataset Dimensions
- **Training Rows:** {train_rows}
- **Testing Rows:** {test_rows}
- **Total Columns:** {train_cols}
  - Label columns: 1 (`{label_col}`)
  - Pixel columns: {num_pixel_cols}

## Data Types and Quality
- **Pixel Data Types:** {', '.join(str(d) for d in pixel_dtypes)}
- **Missing Values (Train):** {train_missing}
- **Missing Values (Test):** {test_missing}
- **Duplicate Rows (Train):** {train_duplicates}
- **Duplicate Rows (Test):** {test_duplicates}

## Pixel Statistics (Train)
- **Minimum Value:** {pixel_min}
- **Maximum Value:** {pixel_max}
- **Mean:** {pixel_mean:.2f}
- **Standard Deviation:** {pixel_std:.2f}

## Reshape Verification
- Data successfully reshaped to dimensions: {image_dims} (Success: {reshape_success})

## Label Analysis
- **Unique Labels (Train):** {list(train_labels)}
- **Unique Labels (Test):** {list(test_labels)}
- Train and Test labels match: {np.array_equal(train_labels, test_labels)}

### Label to Letter Mapping & Counts
"""
for label, count in train_label_counts.items():
    md_content += f"- **Label {label} (Letter {label_to_letter[label]}):** {count} samples\n"

md_content += f"""
### Excluded Classes Verification
- Contains Label 9 (J): {has_J}
- Contains Label 25 (Z): {has_Z}
"""

report_path = os.path.join(REPORTS_DIR, "sign_language_mnist_inspection.md")
with open(report_path, "w") as f:
    f.write(md_content)

print(f"Report saved to {report_path}")
print("Done.")
