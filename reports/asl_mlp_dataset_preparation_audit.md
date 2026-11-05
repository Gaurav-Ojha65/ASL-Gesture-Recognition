# ASL MLP Dataset Preparation Audit

## Objective
Independent verification of the PyTorch Dataset and DataLoader layers for the ASL 28-class MLP, consuming the validated Parquet files without altering stored features.

## Distribution Analysis
### Training Split
**Total samples:** 53259
- Minimum class count: 1241
- Maximum class count: 2083
- Class imbalance ratio (max/min): 1.68

| Semantic Label | Count | Percentage |
| --- | --- | --- |
| A | 1796 | 3.37% |
| B | 1809 | 3.40% |
| C | 1773 | 3.33% |
| D | 2018 | 3.79% |
| E | 1864 | 3.50% |
| F | 2031 | 3.81% |
| G | 1961 | 3.68% |
| H | 2029 | 3.81% |
| I | 2060 | 3.87% |
| J | 2031 | 3.81% |
| K | 2052 | 3.85% |
| L | 1997 | 3.75% |
| M | 1516 | 2.85% |
| N | 1241 | 2.33% |
| O | 1984 | 3.73% |
| P | 1866 | 3.50% |
| Q | 1931 | 3.63% |
| R | 2012 | 3.78% |
| S | 2025 | 3.80% |
| T | 2016 | 3.79% |
| U | 1961 | 3.68% |
| V | 1926 | 3.62% |
| W | 1927 | 3.62% |
| X | 1905 | 3.58% |
| Y | 2083 | 3.91% |
| Z | 1999 | 3.75% |
| DELETE | 1770 | 3.32% |
| SPACE | 1676 | 3.15% |

### Validation Split
**Total samples:** 9186
- Minimum class count: 170
- Maximum class count: 430
- Class imbalance ratio (max/min): 2.53

| Semantic Label | Count | Percentage |
| --- | --- | --- |
| A | 354 | 3.85% |
| B | 381 | 4.15% |
| C | 267 | 2.91% |
| D | 401 | 4.37% |
| E | 373 | 4.06% |
| F | 426 | 4.64% |
| G | 352 | 3.83% |
| H | 365 | 3.97% |
| I | 329 | 3.58% |
| J | 394 | 4.29% |
| K | 397 | 4.32% |
| L | 358 | 3.90% |
| M | 348 | 3.79% |
| N | 210 | 2.29% |
| O | 183 | 1.99% |
| P | 233 | 2.54% |
| Q | 190 | 2.07% |
| R | 379 | 4.13% |
| S | 350 | 3.81% |
| T | 316 | 3.44% |
| U | 418 | 4.55% |
| V | 430 | 4.68% |
| W | 353 | 3.84% |
| X | 286 | 3.11% |
| Y | 392 | 4.27% |
| Z | 311 | 3.39% |
| DELETE | 170 | 1.85% |
| SPACE | 220 | 2.39% |

### Internal Test Split
**Total samples:** 4399
- Minimum class count: 28
- Maximum class count: 450
- Class imbalance ratio (max/min): 16.07

| Semantic Label | Count | Percentage |
| --- | --- | --- |
| A | 145 | 3.30% |
| B | 60 | 1.36% |
| C | 79 | 1.80% |
| D | 164 | 3.73% |
| E | 107 | 2.43% |
| F | 450 | 10.23% |
| G | 249 | 5.66% |
| H | 78 | 1.77% |
| I | 84 | 1.91% |
| J | 222 | 5.05% |
| K | 293 | 6.66% |
| L | 244 | 5.55% |
| M | 93 | 2.11% |
| N | 78 | 1.77% |
| O | 182 | 4.14% |
| P | 28 | 0.64% |
| Q | 108 | 2.46% |
| R | 226 | 5.14% |
| S | 265 | 6.02% |
| T | 71 | 1.61% |
| U | 178 | 4.05% |
| V | 240 | 5.46% |
| W | 239 | 5.43% |
| X | 46 | 1.05% |
| Y | 162 | 3.68% |
| Z | 119 | 2.71% |
| DELETE | 102 | 2.32% |
| SPACE | 87 | 1.98% |

> [!WARNING]
> **Important Evaluation Warning:** The existing extraction audit shows very uneven successful counts in the internal-test split (e.g., P: 28, B: 60, X: 46, N: 78 vs F: 450). Per-class test metrics for classes with very few successful samples will have higher statistical uncertainty.

## Independent Verification
1. Train dataset length = 53,259: **PASS** (Found 53259)
2. Validation dataset length = 9,186: **PASS** (Found 9186)
3. Internal-test dataset length = 4,399: **PASS** (Found 4399)
4. Every feature tensor is exactly shape [63]: **PASS**
5. Every feature tensor is torch.float32: **PASS**
6. Every label is an integer in 0..27: **PASS**
7. No NaN or Inf exists in loaded feature tensors: **PASS**
8. Every semantic label agrees with its stored internal class index: **PASS**
9. The 28-class mapping exactly matches `src/landmarks/labels.py`: **PASS**
10. No sample is duplicated inside any dataset: **PASS**
11. No source path overlaps across train/validation/internal-test: **PASS**
12. DataLoader batch shapes are correct `features [64,63]`, `labels [64]`: **PASS**
13. The first few batches are reproducible when using the same seed: **PASS**
14. Validation/test loaders do not shuffle: **PASS**

## Independent Sample-Value Audit
- Source path exists: **PASS**
- Stored feature length = 63: **PASS**
- Tensor dtype = float32: **PASS**
- Feature values are finite: **PASS**
- Label matches semantic label and `labels.py`: **PASS**

## Final Verdict
All structural, overlap, typing, and DataLoader configuration checks have **PASSED**.

**“Can the three existing Parquet datasets be safely consumed by the future 28-class PyTorch MLP without changing the stored features, labels, or split boundaries?”**

**Yes.** The Parquet datasets load seamlessly into `torch.float32` tensors of shape `[63]` and integer labels `0..27`. The DataLoader configurations correctly handle deterministic shuffling for training and sequential loading for validation/test splits, without any data leakage between the partitions. The data is ready for MLP training.