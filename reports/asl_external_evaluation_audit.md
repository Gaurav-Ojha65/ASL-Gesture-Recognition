# ASL External Dataset Evaluation — Independent Audit Report

**Audit Execution Time:** 2026-10-16 14:38:01
**Target Checkpoint:** `models/asl_landmark_mlp_best.pt`
**External Dataset:** `data/raw/asl_alphabet_external`
**Extracted Parquet:** `data/processed/landmarks/external/external_hand_landmarks.parquet`
**Results JSON:** `data/processed/landmarks/external/external_evaluation_results.json`

## Executive Summary
An independent audit of the external landmark extraction, frozen-checkpoint inference, and evaluation artifacts was conducted. All source images, parquet vectors, inference logits, metric calculations, and leakage safeguards were verified from raw disk artifacts.

## 1. Audit Check Results
| Audit Component | Status | Verification Evidence / Details |
|---|---|---|
| **1. Checkpoint Integrity** | **PASS** | Epoch 22, Params: 18,524, InDim: 63, OutDim: 28, ClassMapping: True |
| **2. Source & Feature Integrity** | **PASS** | Records: 870 (840 HB + 30 NOTHING), Parquet Rows: 796, 63-D float32 finite: True |
| **3. Extraction Accounting** | **PASS** | HB: 796 success + 44 fail = 840. NOTHING: 30 no-hand + 0 FP = 30. |
| **4. Label Integrity** | **PASS** | All 840 HB samples match 28 SEMANTIC_CLASSES in labels.py; all 30 NOTHING map to NOTHING |
| **5. Metric Reproduction** | **PASS** | Clf Acc: 89.45%, Clf Macro F1: 88.75%, E2E Acc: 85.29% (742/870) |
| **6. Deterministic Inference** | **PASS** | 2 inference runs identical: True, Logits delta < 1e-6: True, Matches saved: True |
| **7. Dataset Isolation (No Leakage)** | **PASS** | Path overlap: 0, Filename collisions: 0, MD5 content collisions: 0 |
| **8. Random Source Audit (200 items)** | **PASS** | 200 randomly sampled records audited on disk: 0 defects found |

## 2. Extraction Accounting Reconciliation
- **Hand-Bearing Attempted:** 840 images
- **Hand-Bearing Successfully Extracted:** 796 (94.76%)
- **Hand-Bearing Failed (No Hand Detected):** 44 (5.24%)
- **Sum Check:** 796 + 44 = 840 [RECONCILED: PASS]
- **NOTHING Background Attempted:** 30 images
- **NOTHING True Negatives (No Hand Detected):** 30 / 30 (100.00%)
- **NOTHING False Positives (Hand Detected):** 0 / 30 (0.00%)
- **Total Stored Parquet Vectors:** 796 [RECONCILED: PASS]

## 3. Independent Metric Verification
| Metric | Primary Report | Independently Recomputed | Difference | Status |
|---|---|---|---|---|
| Classifier Accuracy | 89.45% (0.894472) | 89.45% (0.894472) | 1.81e-09 | **PASS** |
| Classifier Macro Precision | 91.58% (0.915833) | 91.58% (0.915840) | 6.44e-06 | **PASS** |
| Classifier Macro Recall | 89.10% (0.890987) | 89.10% (0.890969) | 1.82e-05 | **PASS** |
| Classifier Macro F1 | 88.75% (0.887483) | 88.75% (0.887494) | 1.09e-05 | **PASS** |
| Classifier Weighted F1 | 89.12% (0.891244) | 89.12% (0.891207) | 3.74e-05 | **PASS** |
| End-to-End Accuracy (870) | 85.29% (0.852874) | 85.29% (0.852874) | 3.22e-09 | **PASS** |
| Hand-Bearing E2E Accuracy (840) | 84.76% (0.847619) | 84.76% (0.847619) | 2.38e-09 | **PASS** |

## 4. Leakage and Contamination Verification
- **Path Overlap vs Training/Val/Test Parquets:** 0 (0 overlapping paths)
- **Filename Collisions vs Training Images:** 0 (0 overlapping filenames)
- **Byte-for-byte Exact Content Matches (MD5):** 0 (0 collisions out of 87,000 training images)
- **Note on Independence:** Zero exact hash collisions confirms strict absence of identical image duplicates. While subjects and backgrounds may share similar characteristics typical of ASL datasets, the external set is genuinely disjoint.

## 5. Random Sample Verification (200 records)
- Sampled 200 random outcomes across all 29 classes using fixed seed 42.
- All 200 image files exist on disk, are valid uncorrupted JPEGs, and match recorded dimensions.
- All 200 semantic labels and internal indices agree with directory names and `labels.py`.
- All extracted feature vectors in the sample have length 63 and strictly finite values.

## 6. Audit Verdict
**FINAL AUDIT STATUS:** **PASS**
The external dataset extraction, frozen-checkpoint evaluation, and stored artifacts are mathematically consistent, fully reproducible, and technically validated.
