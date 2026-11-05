# Sign Language MNIST Dataset Inspection Report

## File Verification
- Training file: `data\raw\sign_language_mnist\sign_mnist_train.csv` (Exists: True)
- Testing file: `data\raw\sign_language_mnist\sign_mnist_test.csv` (Exists: True)

## Dataset Dimensions
- **Training Rows:** 27455
- **Testing Rows:** 7172
- **Total Columns:** 785
  - Label columns: 1 (`label`)
  - Pixel columns: 784

## Data Types and Quality
- **Pixel Data Types:** int64
- **Missing Values (Train):** 0
- **Missing Values (Test):** 0
- **Duplicate Rows (Train):** 0
- **Duplicate Rows (Test):** 0

## Pixel Statistics (Train)
- **Minimum Value:** 0
- **Maximum Value:** 255
- **Mean:** 159.29
- **Standard Deviation:** 48.77

## Reshape Verification
- Data successfully reshaped to dimensions: (28, 28) (Success: True)

## Label Analysis
- **Unique Labels (Train):** [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(10), np.int64(11), np.int64(12), np.int64(13), np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22), np.int64(23), np.int64(24)]
- **Unique Labels (Test):** [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(10), np.int64(11), np.int64(12), np.int64(13), np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(18), np.int64(19), np.int64(20), np.int64(21), np.int64(22), np.int64(23), np.int64(24)]
- Train and Test labels match: True

### Label to Letter Mapping & Counts
- **Label 0 (Letter A):** 1126 samples
- **Label 1 (Letter B):** 1010 samples
- **Label 2 (Letter C):** 1144 samples
- **Label 3 (Letter D):** 1196 samples
- **Label 4 (Letter E):** 957 samples
- **Label 5 (Letter F):** 1204 samples
- **Label 6 (Letter G):** 1090 samples
- **Label 7 (Letter H):** 1013 samples
- **Label 8 (Letter I):** 1162 samples
- **Label 10 (Letter K):** 1114 samples
- **Label 11 (Letter L):** 1241 samples
- **Label 12 (Letter M):** 1055 samples
- **Label 13 (Letter N):** 1151 samples
- **Label 14 (Letter O):** 1196 samples
- **Label 15 (Letter P):** 1088 samples
- **Label 16 (Letter Q):** 1279 samples
- **Label 17 (Letter R):** 1294 samples
- **Label 18 (Letter S):** 1199 samples
- **Label 19 (Letter T):** 1186 samples
- **Label 20 (Letter U):** 1161 samples
- **Label 21 (Letter V):** 1082 samples
- **Label 22 (Letter W):** 1225 samples
- **Label 23 (Letter X):** 1164 samples
- **Label 24 (Letter Y):** 1118 samples

### Excluded Classes Verification
- Contains Label 9 (J): False
- Contains Label 25 (Z): False
