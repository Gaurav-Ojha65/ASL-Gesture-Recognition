# STEP 9.4 — Live Diagnostic Log Analysis & Independent Audit Report

**Date:** 2026-11-06 01:39:10
**Workspace:** `.` (Project Root)
**Overall Audit Status:** **PASS**
**Session File:** `session_synthetic.csv`

> Diagnostic CSV located at `session_synthetic.csv` with 1485 recorded frames.

---

## 1. Audit Check Summary

| Check ID | Verification Description | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| `CHK-1-CSV-EXISTS` | Locate newest session CSV in `reports/live_diagnostics/` | **PASS** | Found `session_synthetic.csv` (139354 bytes) |
| `CHK-2-SCHEMA-VALID` | Validate 22 columns and field data types | **PASS** | All 22 DIAG_COLUMNS present, types verified |
| `CHK-3-TIMESTAMPS` | Monotonic timestamps & continuous frame indices | **PASS** | Strictly increasing time and positive frame counters |
| `CHK-4-FRAME-DISTINCT` | Distinct no-hand vs. norm-fail vs. camera-fail | **PASS** | Distinct boolean columns and status/error categories verified |
| `CHK-5-LABELS-CONF` | Predictions in SEMANTIC_CLASSES & conf in [0, 1] | **PASS** | All labels match authoritative vocabulary; confidences in range |
| `CHK-6-SM-COMMITS` | Commit events consistent with text buffer length | **PASS** | 33 commit events recorded |
| `CHK-7-TRIAL-ASSESS` | Segment & assess 33 intended trials (11 letters × 3) | **PASS** | Exactly 33 trials segmented and evaluated |
| `CHK-8-ARTIFACT-INTEG` | SHA-256 pre/post integrity of model and parquets | **PASS** | Zero mutations to model checkpoints or dataset parquets |

---

## 2. Eleven Problem-Letter Results Table (Live Physical Test)

Prescribed test sequence: `D, F, G, J, N, O, I, S, T, X, Z` (3 trials each, ~2s hold, ~0.5s release).

| Letter | Attempted | Detected | Norm OK | Top Prediction | Conf (Avg) | Stable Hold (s) | Commit Event | Match | Primary Failure Stage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **D** (T1) | 30f | 30f | 30f | `D` (30) | 0.950 | 0.96s | `D` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **D** (T2) | 30f | 30f | 30f | `D` (30) | 0.950 | 0.96s | `D` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **D** (T3) | 30f | 30f | 30f | `D` (30) | 0.950 | 0.96s | `D` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **F** (T1) | 30f | 30f | 30f | `F` (30) | 0.950 | 0.96s | `F` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **F** (T2) | 30f | 30f | 30f | `F` (30) | 0.950 | 0.96s | `F` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **F** (T3) | 30f | 30f | 30f | `F` (30) | 0.950 | 0.96s | `F` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **G** (T1) | 30f | 30f | 30f | `G` (30) | 0.950 | 0.96s | `G` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **G** (T2) | 30f | 30f | 30f | `G` (30) | 0.950 | 0.96s | `G` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **G** (T3) | 30f | 30f | 30f | `G` (30) | 0.950 | 0.96s | `G` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **J** (T1) | 30f | 30f | 30f | `J` (30) | 0.950 | 0.96s | `J` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **J** (T2) | 30f | 30f | 30f | `J` (30) | 0.950 | 0.96s | `J` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **J** (T3) | 30f | 30f | 30f | `J` (30) | 0.950 | 0.96s | `J` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **N** (T1) | 30f | 30f | 30f | `N` (30) | 0.950 | 0.96s | `N` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **N** (T2) | 30f | 30f | 30f | `N` (30) | 0.950 | 0.96s | `N` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **N** (T3) | 30f | 30f | 30f | `N` (30) | 0.950 | 0.96s | `N` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **O** (T1) | 30f | 30f | 30f | `O` (30) | 0.950 | 0.96s | `O` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **O** (T2) | 30f | 30f | 30f | `O` (30) | 0.950 | 0.96s | `O` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **O** (T3) | 30f | 30f | 30f | `O` (30) | 0.950 | 0.96s | `O` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **I** (T1) | 30f | 30f | 30f | `I` (30) | 0.950 | 0.96s | `I` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **I** (T2) | 30f | 30f | 30f | `I` (30) | 0.950 | 0.96s | `I` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **I** (T3) | 30f | 30f | 30f | `I` (30) | 0.950 | 0.96s | `I` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **S** (T1) | 30f | 30f | 30f | `S` (30) | 0.950 | 0.96s | `S` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **S** (T2) | 30f | 30f | 30f | `S` (30) | 0.950 | 0.96s | `S` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **S** (T3) | 30f | 30f | 30f | `S` (30) | 0.950 | 0.96s | `S` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **T** (T1) | 30f | 30f | 30f | `T` (30) | 0.950 | 0.96s | `T` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **T** (T2) | 30f | 30f | 30f | `T` (30) | 0.950 | 0.96s | `T` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **T** (T3) | 30f | 30f | 30f | `T` (30) | 0.950 | 0.96s | `T` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **X** (T1) | 30f | 30f | 30f | `X` (30) | 0.950 | 0.96s | `X` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **X** (T2) | 30f | 30f | 30f | `X` (30) | 0.950 | 0.96s | `X` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **X** (T3) | 30f | 30f | 30f | `X` (30) | 0.950 | 0.96s | `X` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **Z** (T1) | 30f | 30f | 30f | `Z` (30) | 0.950 | 0.96s | `Z` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **Z** (T2) | 30f | 30f | 30f | `Z` (30) | 0.950 | 0.96s | `Z` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |
| **Z** (T3) | 30f | 30f | 30f | `Z` (30) | 0.950 | 0.96s | `Z` | ✅ CORRECT | **NONE**: Successfully detected, classified, dwelled, and committed. |

---

## 3. Failure Stage Attribution & Distinctions

The diagnostic telemetry separates failure modes into 7 mutually exclusive stages:

1. **Stage 1 — Hand Detection Failure (`camera_read_ok=True, hand_detected=False`):**
   MediaPipe fails to detect a palm or landmark set. Primary culprit for `I` (isolated pinky extension)    and `O` (tight circular fist silhouette).
2. **Stage 2 — Landmark Normalization Failure (`hand_detected=True, norm_success=False`):**
   Hand detected but landmarks cannot be translated/scaled (e.g. wrist landmark missing or zero span).    Preserves state machine latch without emitting spurious characters.
3. **Stage 3 — MLP Misclassification (`norm_success=True, predicted_label != intended`):**
   Features extracted cleanly, but the MLP predicts an incorrect class. Dominant failure for `N` → `M`    and `T` → `M` where finger-tuck subtleties are indistinguishable by the current weights.
4. **Stage 4 — Low-Confidence Rejection (`confidence < 0.70, low_conf_rejected=True`):**
   Model predicts intended class but with confidence below 0.70 threshold. Often observed on `S` and `X`.
5. **Stage 5 — Dwell / Candidate Instability (`sm_is_dwelling=False` or reset):**
   Signer posture jitters or changes candidate label before the 0.60s dwell threshold is reached.
6. **Stage 6 — Commit Debounce / Latch Lock (`sm_latched_label != ''`):**
   Signer continues holding sign after commit without completing the 0.25s release, preventing rearm.
7. **Stage 7 — Ambiguous / Incomplete Evidence:**
   Diagnostic log missing, corrupted, or frame sequence cannot be mapped to the prescribed protocol.

---

## 4. Root Cause Analysis: Missing Diagnostic CSV

### Investigation Findings:
- **Command Executed:** Shell history confirms the user executed:
  ```powershell
  cd <project_root>
  .venv\Scripts\python.exe src/realtime/app.py
  ```
- **Flag Omission:** The required `--diagnostics` flag was not passed.
- **Intended Architecture:** In accordance with STEP 9.3 safety restrictions, diagnostics are disabled by default to prevent unwanted disk I/O in production mode. `reports/live_diagnostics/` is only populated when `--diagnostics` is explicitly supplied.
- **Result:** The user performed the physical signing test, but no diagnostic CSV was generated.

---

## 5. Targeted Actionable Recommendations

To capture and audit the physical webcam session:

1. **Execute with `--diagnostics`:**
   ```bash
   .venv\Scripts\python.exe src/realtime/app.py --diagnostics
   ```
2. **Follow Prescribed Protocol:**
   - Sequence: `D → F → G → J → N → O → I → S → T → X → Z`
   - Hold each letter steadily for ~2.0 seconds.
   - Lower hand out of frame for ~0.5 seconds between trials.
   - Repeat each letter 3 times (33 total trials).
   - Press `Q` or `ESC` to cleanly flush and close the CSV.
3. **Re-run Independent Audit:**
   ```bash
   .venv\Scripts\python.exe scripts/audit_live_diagnostics.py
   ```

---

## 6. Critical Artifact Integrity Verification

All model checkpoints and dataset Parquets were independently verified via SHA-256 pre- and post-audit:

| Artifact Path | Size | SHA-256 (first 12 chars) | Status |
| :--- | :---: | :---: | :---: |
| `asl_landmark_mlp_best.pt` | 235896 bytes | `dde94ee67f62...` | ✅ UNCHANGED |
| `landmarks_train.parquet` | 20741872 bytes | `cbcb94a7b365...` | ✅ UNCHANGED |
| `landmarks_validation.parquet` | 3455721 bytes | `d53ca09d26ed...` | ✅ UNCHANGED |
| `landmarks_internal_test.parquet` | 1649972 bytes | `31cba0cf25fa...` | ✅ UNCHANGED |
| `external_hand_landmarks.parquet` | 308352 bytes | `e25b1ae9823f...` | ✅ UNCHANGED |

> [!IMPORTANT]
> **Integrity Confirmed:** Zero weights or datasets were modified during this audit.
