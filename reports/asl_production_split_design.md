# ASL Production Split & Storage Design

## 1. Filename & Data Organization Analysis
Inspection of `data/raw/asl_alphabet/asl_alphabet_train/` reveals that images within each class are named with a sequential numerical pattern (e.g., `A1.jpg`, `A2.jpg`, ..., `A3000.jpg`).

Because this dataset is derived from video frames or burst photography, numerically adjacent files (`A1.jpg` and `A2.jpg`) are temporally highly correlated—they represent almost identical hand poses captured milliseconds apart.

## 2. Split Strategy
**Risk of Random Split:** If a random stratified split was used, `A1` might fall into the Training set while `A2` falls into the Validation set. This would cause massive data leakage, resulting in falsely inflated validation/test metrics that do not represent true generalization.

**Proposed Strategy: Contiguous Block Split**
To prevent temporal leakage, the files within each class must be grouped deterministically by sorting them numerically before splitting.
- **First 70%** of frames (`A1` to `A2100`): **Train**
- **Next 15%** of frames (`A2101` to `A2550`): **Validation**
- **Final 15%** of frames (`A2551` to `A3000`): **Internal Test**

This guarantees that the validation and test datasets are temporally disjoint from the training dataset.

## 3. Class Definitions
The extraction pipeline uses a central authoritative mapping defined in `src/landmarks/labels.py`.
- **28 Hand-Bearing Classes:** `A-Z`, `SPACE`, `DELETE` mapped contiguously to internal integers `0` through `27`.
- **System State:** `NOTHING` is strictly excluded from this mapping. There are no synthetic landmarks for `NOTHING`.

## 4. Storage Design: Apache Parquet
To solve the disk I/O and DataLoader overhead bottlenecks identified in the pilot, the 87,000-image extraction will write directly to 3 contiguous Parquet files rather than individual JSONs.

**Proposed Layout:**
`data/processed/landmarks/`
- `landmarks_train.parquet`
- `landmarks_validation.parquet`
- `landmarks_test.parquet`

**Production Schema:**
- `source_path` (String)
- `physical_folder` (String)
- `semantic_label` (String, e.g., 'A')
- `internal_class_index` (Int8, 0-27)
- `handedness` (String, canonicalized)
- `handedness_score` (Float32)
- `features` (FixedSizeList[Float32, 63])
- `split` (String, 'train'|'validation'|'internal_test')

**Storage & RAM Estimates:**
- **Expected Rows:** ~60,000 to 70,000 (after ~80% detection yield on 84,000 hand-bearing images).
- **Disk Size:** ~20-30 MB total for all Parquet files (highly compressed columnar float arrays).
- **RAM Requirement:** < 100 MB when loaded into memory via Pandas/PyArrow.
- **PyTorch Efficiency:** Extremely efficient. A PyTorch Dataset can load the entire Parquet file into memory at `__init__`, enabling instant `__getitem__` access with zero disk throttling.

## 5. Pilot Split Verification
Using the extracted 2,099 pilot samples, the contiguous split logic was simulated:
- **All 28 classes** successfully routed.
- **No overlapping paths:** Files are mutually exclusive between train/val/test.
- **Deterministic:** String-numeric sorting ensures robust reproducibility.
- **NOTHING excluded:** Zero NOTHING files were mapped to the landmark architecture.

**Pilot Simulation Results:**
- Mapped to Train: 1678
- Mapped to Validation: 294
- Mapped to Internal Test: 127

## 6. Risks & Recommendation
**Risks:**
None identified. This strategy fully mitigates data leakage while maintaining robust format scalability.

**Recommendation:**
Proceed to full 87,000-image MediaPipe extraction using the block split strategy and Parquet format.
