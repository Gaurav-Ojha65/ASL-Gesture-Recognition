# STEP 9.3 — Problem-Letter Recognition Diagnostics & Independent Audit Report

**Date:** 2026-10-16 20:46:43
**Workspace:** `.` (Project Root)
**Overall Audit Status:** **PASS** (5/5 checks passed)

---

## Executive Summary

This report diagnoses observed poor recognition of 11 target ASL letters and introduces an optional
runtime diagnostic subsystem. It is **diagnostic only**: no model weights, thresholds, or datasets
were modified.

> [!IMPORTANT]
> **Evidence distinction:** All metrics below are derived from the **external test dataset**
> (static images, 30 per class, different subjects and lighting). Webcam recognition can differ
> substantially due to background variation, dynamic motion blur, partial occlusion, and live
> MediaPipe tracking artefacts. Conclusions from static images may not fully explain live signing issues.
> Any claims about live webcam behaviour require a dedicated recording session.

---

## 1. Eleven Problem-Letter Analysis (from External Static Dataset)

### 1.1 Extraction (MediaPipe Detection) Stage

| Letter | Attempted | Detected | Failed | Detection Rate | Primary Issue |
| :--- | :---: | :---: | :---: | :---: | :--- |
| D | 30 | 29 | 1 | 96.7% | None |
| F | 30 | 30 | 0 | 100.0% | None |
| G | 30 | 30 | 0 | 100.0% | None |
| J | 30 | 30 | 0 | 100.0% | None |
| N | 30 | 29 | 1 | 96.7% | None |
| O | 30 | 26 | 4 | 86.7% | Moderate detection miss |
| **I** | **30** | **21** | **9** | **70.0%** | **⚠ HIGH detection failure** |
| S | 30 | 28 | 2 | 93.3% | None |
| T | 30 | 30 | 0 | 100.0% | None |
| X | 30 | 30 | 0 | 100.0% | None |
| Z | 30 | 30 | 0 | 100.0% | None |

> **Key finding for `I`:** 30 % of images had no hand detected. This is the dominant source of
> end-to-end failure for `I` in the external dataset. In live webcam use, a small or extended pinky
> at low resolution may also fall below MediaPipe's 0.3 confidence threshold.

### 1.2 MLP Classification Stage (on Successfully Extracted Images Only)

| Letter | MLP Eval | Correct | Precision | Recall | F1 | Top Confusions |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| D | 29 | 29 | 1.000 | 1.000 | 1.000 | (none) |
| F | 30 | 26 | 0.897 | 0.867 | 0.881 | E×2, B×2 |
| G | 30 | 29 | 0.906 | 0.967 | 0.935 | H×1 |
| J | 30 | 28 | 1.000 | 0.933 | 0.966 | Y×2 |
| **N** | **29** | **11** | **0.917** | **0.379** | **0.537** | **M×17, A×1** |
| O | 26 | 25 | 0.676 | 0.962 | 0.794 | DELETE×1 |
| I | 21 | 20 | 0.952 | 0.952 | 0.952 | Y×1 |
| S | 28 | 26 | 1.000 | 0.929 | 0.963 | E×1, I×1 |
| **T** | **30** | **19** | **1.000** | **0.633** | **0.776** | **M×7, O×2, SPACE×2** |
| X | 30 | 25 | 1.000 | 0.833 | 0.909 | G×2, M×2, R×1 |
| Z | 30 | 30 | 1.000 | 1.000 | 1.000 | (none) |

### 1.3 Confidence Distributions

| Letter | Avg Conf | Min Conf | Max Conf |
| :--- | :---: | :---: | :---: |
| D | 1.000 | 1.000 | 1.000 |
| F | 0.902 | 0.320 | 0.999 |
| G | 0.976 | 0.742 | 1.000 |
| J | 0.970 | 0.603 | 1.000 |
| N | 0.827 | 0.456 | 0.995 |
| O | 0.931 | 0.472 | 1.000 |
| I | 0.969 | 0.718 | 1.000 |
| S | 0.798 | 0.411 | 0.995 |
| T | 0.777 | 0.435 | 0.999 |
| X | 0.854 | 0.393 | 0.998 |
| Z | 0.999 | 0.978 | 1.000 |

> **Key finding for `N` and `T`:** The MLP produces confident-but-wrong predictions.
> `N` → `M` (17/18 errors): These signs differ by finger count; the MLP confuses
> 3-finger-under-thumb (`N`) with 4-finger-under-thumb (`M`).
> `T` → `M` (7 errors): Similar hand position with slight thumb placement difference.
> The model rarely hesitates — Min Conf for N = 0.456, T = 0.435 — meaning some
> confused predictions pass even the 0.70 confidence threshold.

### 1.4 Comparison with Internal Test Set

| Letter | Ext Dataset Recall | Int Test Recall | Ext F1 | Int F1 | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| D | 1.000 | 0.930 | 1.000 | 0.70 | Strong external, weaker internal (likely imbalance) |
| F | 0.867 | 1.000 | 0.881 | 0.99 | External harder than internal |
| G | 0.967 | 0.940 | 0.935 | 0.96 | Consistent |
| J | 0.933 | 0.990 | 0.966 | 0.99 | Consistent |
| N | 0.379 | 0.900 | 0.537 | 0.88 | **Major degradation on external; N→M is a consistent failure** |
| O | 0.962 | 0.370 | 0.794 | 0.49 | External much better; O is hard on internal (large support mismatch) |
| I | 0.952 | 0.760 | 0.952 | 0.85 | Ext classifier strong; detection miss is the real problem |
| S | 0.929 | 0.910 | 0.963 | 0.95 | Consistent |
| T | 0.633 | 0.630 | 0.776 | 0.73 | **Consistent across both datasets; M confusion is structural** |
| X | 0.833 | 0.870 | 0.909 | 0.90 | Consistent |
| Z | 1.000 | 0.960 | 1.000 | 0.96 | Strong |

---

## 2. Error Source Classification

| Category | Letters | Evidence |
| :--- | :--- | :--- |
| **Detection-Primary** | I | 30% miss rate on static external images; worse expected in webcam |
| **Classification-Primary (N→M, T→M)** | N, T | High detection rate but low recall; high-confidence wrong predictions |
| **Mixed (detection + classification)** | O | 13% detection miss + O/DELETE confusion |
| **Marginal classification issues** | F, J, S, X | Isolated 1-3 misclassifications; generally strong |
| **No evidence of issue** | D, G, Z | ≥ 97% classifier recall on external set |

---

## 3. Runtime Diagnostics Subsystem

### 3.1 Schema (22 CSV columns)

| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | int | Cumulative frame counter from cap.read() |
| `timestamp_sec` | float | Monotonic time.time() stamp |
| `camera_read_ok` | bool | False only on a failed cap.read() |
| `hand_detected` | bool | MediaPipe found ≥1 hand landmark set |
| `norm_success` | bool | LandmarkNormalizer returned valid 63-D features |
| `norm_failure_reason` | str | Reason string when norm fails |
| `predicted_label` | str | MLP top-1 class label |
| `confidence` | float | Softmax confidence of top-1 |
| `top1_label / top1_prob` | str/float | Top-1 prediction |
| `top2_label / top2_prob` | str/float | Top-2 prediction |
| `top3_label / top3_prob` | str/float | Top-3 prediction |
| `low_conf_rejected` | bool | Confidence below min_confidence threshold |
| `sm_candidate` | str | State-machine candidate label |
| `sm_is_dwelling` | bool | Whether candidate is accumulating dwell time |
| `sm_latched_label` | str | Currently latched label (hold debounce) |
| `commit_event_label` | str | Label emitted this frame (blank if none) |
| `text_buffer_length` | int | Characters in text buffer after this frame |
| `status` | str | Pipeline status string |
| `error_category` | str | CAMERA_READ_FAIL / DETECTOR_ERROR / INVALID_FEATURES / INFERENCE_ERROR |

### 3.2 Design Notes

- **Camera-read failures** are logged via `log_camera_failure()` with `camera_read_ok=False`
  and `text_buffer_length=""` (no pipeline state available). They are never treated as confirmed
  no-hand events.
- **Invalid normalisation** (`norm_success=False`) is recorded with `hand_detected=True` and
  `error_category=INVALID_FEATURES`. The state machine latch is preserved; the event is not
  treated as `NO_HAND`.
- **Low-confidence predictions** are recorded with `predicted_label` and `confidence` filled in,
  `low_conf_rejected=True`, and `commit_event_label=""`.
- **No video or image data** is saved to disk at any time.

### 3.3 Launch Commands

```bash
# Default session — no diagnostics (production mode)
.venv\Scripts\python.exe src/realtime/app.py

# Enable diagnostics (CSV written to reports/live_diagnostics/)
.venv\Scripts\python.exe src/realtime/app.py --diagnostics

# Custom diagnostics directory
.venv\Scripts\python.exe src/realtime/app.py --diagnostics --diagnostics-dir my_diag_sessions
```

---

## 4. Independent Audit Check Results

| Check | Description | Result | Details |
| :--- | :--- | :--- | :--- |
| `json_loaded` | 870 records in results JSON | **PASS** | Expected 870, got 870 |
| `class_mapping` | 11 target letters in SEMANTIC_CLASSES | **PASS** | All 11 letters mapped correctly |
| `per_letter_metrics` | Metrics recomputed from raw records | **PASS** | Metrics recomputed from raw records; see table in report |
| `csv_schema` | CSV columns, types, and logged rows correct | **PASS** | 22 columns present, 4 rows written |
| `artifact_integrity` | Checkpoint & Parquets SHA-256 unchanged | **PASS** | Verified zero mutation |

---

## 5. Test Results

- `tests/test_diagnostics.py`: **10 / 10 passed** (disabled=no file, enabled=valid CSV, predicted labels,
  detection failure, invalid norm, camera failure, low-conf, commit event, cleanup, deterministic replay)
- Full project test suite: **69 passed** (confirmed before audit)

---

## 6. Unresolved Questions Requiring Live Signing

The following cannot be resolved from static external images:

1. **`I` in motion:** Does the pinky-only hand pose cause MediaPipe tracking loss specifically
   during motion (e.g. `J` motion following `I`)? Requires recording a live signing session.
2. **`N` / `T` in user signing:** Are user hand positions closer to `M` than the training data?
   The external dataset shows 17/18 `N` images predicted as `M` — if training data had more
   `M`-leaning `N` examples this may be structural.
3. **Confidence threshold impact on `S` and `X`:** Min confidence for these letters reaches
   0.39–0.41 in the external set; in a live session some frames will be below 0.70 and rejected.
   Live recording would quantify how often dwell is broken by low-confidence frames.
4. **Dwell-time adequacy:** The 0.60s dwell may be too long for fast signers on some letters.
   Adjusting `--dwell` can be evaluated with `--diagnostics` enabled during a live session.

---

## 7. Artifact Integrity

```
asl_landmark_mlp_best.pt: size=235896, sha=dde94ee67f62..., unchanged=True
landmarks_train.parquet: size=20741872, sha=cbcb94a7b365..., unchanged=True
landmarks_validation.parquet: size=3455721, sha=d53ca09d26ed..., unchanged=True
landmarks_internal_test.parquet: size=1649972, sha=31cba0cf25fa..., unchanged=True
external_hand_landmarks.parquet: size=308352, sha=e25b1ae9823f..., unchanged=True
```
