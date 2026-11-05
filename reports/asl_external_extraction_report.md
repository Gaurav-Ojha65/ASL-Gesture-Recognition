# ASL External Dataset Landmark Extraction Report

## 1. Overview
- **External Dataset Path:** `data/raw/asl_alphabet_external`
- **Model Task Asset:** `hand_landmarker.task`
- **MediaPipe Thresholds:** `min_hand_detection_confidence=0.3`, `min_hand_presence_confidence=0.3`
- **Output Parquet:** `data/processed/landmarks/external/external_hand_landmarks.parquet`
- **Full Results JSON:** `data/processed/landmarks/external/external_evaluation_results.json`
- **Extraction Runtime:** 15.94 seconds (54.6 images/s)

## 2. Overall Extraction Statistics
- **Total Attempted Images:** 870 (840 hand-bearing + 30 NOTHING)
- **Hand-Bearing Images Attempted:** 840
- **Hand-Bearing Successful Detections:** 796
- **Hand-Bearing Failed Detections:** 44
- **Hand-Bearing Extraction Rate:** 94.76%
- **NOTHING Images Attempted:** 30
- **NOTHING False-Positive Detections:** 0 (0.00%)
- **Total Normalized Vectors Stored in Parquet:** 796

## 3. Failure Reasons Breakdown
| Failure Reason | Occurrences | Percentage (of 870) |
|---|---|---|
| No hand detected | 74 | 8.51% |

## 4. Image Dimensions Distribution
| Dimension (Width x Height) | Image Count | Percentage |
|---|---|---|
| 200x200 | 866 | 99.54% |
| 150x200 | 4 | 0.46% |

## 5. Per-Class Extraction Table
| Class | Class Type | Attempted | Success | Fail | Success Rate |
|---|---|---|---|---|---|
| `A` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `B` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `C` | Hand-Bearing | 30 | 29 | 1 | 96.7% |
| `D` | Hand-Bearing | 30 | 29 | 1 | 96.7% |
| `E` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `F` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `G` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `H` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `I` | Hand-Bearing | 30 | 21 | 9 | 70.0% |
| `J` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `K` | Hand-Bearing | 30 | 27 | 3 | 90.0% |
| `L` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `M` | Hand-Bearing | 30 | 28 | 2 | 93.3% |
| `N` | Hand-Bearing | 30 | 29 | 1 | 96.7% |
| `O` | Hand-Bearing | 30 | 26 | 4 | 86.7% |
| `P` | Hand-Bearing | 30 | 29 | 1 | 96.7% |
| `Q` | Hand-Bearing | 30 | 28 | 2 | 93.3% |
| `R` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `S` | Hand-Bearing | 30 | 28 | 2 | 93.3% |
| `T` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `U` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `V` | Hand-Bearing | 30 | 29 | 1 | 96.7% |
| `W` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `X` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `Y` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `Z` | Hand-Bearing | 30 | 30 | 0 | 100.0% |
| `DELETE` | Hand-Bearing | 30 | 23 | 7 | 76.7% |
| `SPACE` | Hand-Bearing | 30 | 20 | 10 | 66.7% |
| `NOTHING` | No-Hand | 30 | 0 | 30 | 0.0% |
