# STEP 9.2 — Real-Time Webcam, MediaPipe, Frozen MLP Integration & Independent Audit Report

**Date:** 2026-10-16 15:33:15
**Workspace:** `.` (Project Root)
**Overall Audit Status:** **PASS** (16/16 checks passed)

---

## Executive Summary

The real-time recognition pipeline for the American Sign Language (ASL) gesture recognition interface has been implemented and independently verified under STEP 9.2.

The application connects:
$$\text{OpenCV Webcam} \longrightarrow \text{MediaPipe Hand Landmarker} \longrightarrow \text{LandmarkNormalizer} \longrightarrow \text{Frozen MLP} \longrightarrow \text{GestureStateMachine} \longrightarrow \text{TextBuffer} \longrightarrow \text{OpenCV HUD}$$

### Verification Status Distinction

> [!IMPORTANT]
> **Software Integration Verified (PASSED):** Complete end-to-end integration verified using deterministic unit tests, mock detectors, synthetic frames, failure simulations, and cryptographic integrity checks.
>
> **Physical Webcam Validation (PENDING):** Requires manual launch and real-world signing by the user with their attached physical camera.

---

## 1. Architecture and Component Verification

| Component | Target / Specification | Actual Audited State | Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `ASLLandmarkMLP` (Linear 63→128, BN, ReLU, Drop, Linear 128→64, ReLU, Drop, Linear 64→28) | Exact 3-layer MLP architecture | **PASS** |
| **Parameter Count** | 18,524 trainable parameters | 18,524 parameters verified | **PASS** |
| **Input Dimension** | 63 normalized coordinates | 63-D float32 vector verified | **PASS** |
| **Output Classes** | 28 classes (`A`–`Z`, `DELETE`, `SPACE`) | 28 classes matching `src/landmarks/labels.py` | **PASS** |
| **Model Weights** | `models/asl_landmark_mlp_best.pt` (Epoch 22) | Verified loadable, eval mode, frozen weights | **PASS** |
| **MediaPipe Thresholds** | Detection = 0.3, Presence = 0.3 | Exact production thresholds enforced | **PASS** |
| **State Machine** | Dwell = 0.60s, Release = 0.25s, Min Conf = 0.70 | Defaults preserved and verified | **PASS** |

---

## 2. Image Coordinate & Mirroring Convention

1. **Dataset Baseline:** The original dataset extraction (`full_extraction.py` and `evaluate_external_dataset.py`) processed unmirrored images directly from disk.
2. **Webcam Interaction Convention:** In real-time webcam applications, users intuitively expect mirror behavior (`cv2.flip(frame, 1)`).
3. **Canonicalization Invariance:** When the feed is mirrored horizontally, a physical Right hand appears visually as a Left hand, which MediaPipe identifies with `handedness='Left'`. The existing `LandmarkNormalizer` negates the X coordinates (`arr[:, 0] = -arr[:, 0]`), automatically transforming it into the canonical right-hand feature space.
4. **Configuration:** Mirror mode is enabled by default (`--mirror`, default `True`) and can be toggled via `--no-mirror`.

---

## 3. Independent Audit Check Results

| Check ID | Verification Description | Result | Details |
| :--- | :--- | :--- | :--- |
| `checkpoint_exists` | Model checkpoint presence on disk | **PASS** | Found at models/asl_landmark_mlp_best.pt (235896 bytes) |
| `parameter_count` | Exact trainable parameter count (18,524) | **PASS** | Expected 18,524; found 18524 |
| `architecture_dims` | Model input (63) and output (28) dimensions | **PASS** | in_features=63, out_features=28 |
| `class_mapping` | Checkpoint class mapping matches `labels.py` | **PASS** | Checkpoint mapping agrees with src/landmarks/labels.py |
| `normalization_flow` | LandmarkNormalizer outputs 63-D finite float32 | **PASS** | Valid=True, shape=(63,), finite=True |
| `inference_flow` | Probability sum and confidence mathematics | **PASS** | Label=DELETE, conf=0.5660, prob_sum=1.0000 |
| `raw_logits` | Model preserves unnormalized raw logits | **PASS** | Logits shape=torch.Size([1, 28]), sum=-117.05 |
| `pipeline_dwell` | No early commit before 0.60s dwell threshold | **PASS** | Early commit: False, committed text: 'A' |
| `pipeline_hold_debounce` | Sustained held sign emits exactly 1 commit | **PASS** | Duplicate commits during hold: False, text: 'A' |
| `pipeline_release_repeat` | Sustained 0.25s no-hand releases latch ('AA') | **PASS** | Re-armed: True, text after second commit: 'AA' |
| `pipeline_low_conf` | Predictions below 0.70 confidence rejected | **PASS** | Low-conf commits: False, text: 'AA' |
| `pipeline_tokens` | Sequencing of letters, SPACE, and DELETE | **PASS** | Expected 'HI U', got 'HI U' |
| `failure_missing_checkpoint` | FileNotFoundError on non-existent checkpoint | **PASS** | FileNotFoundError correctly raised on missing checkpoint |
| `failure_camera_open` | Clean exit with code 1 when camera fails | **PASS** | Camera open failure returned exit code 1 |
| `failure_invalid_normalization` | Invalid features do NOT release hold latch | **PASS** | Latch preserved under invalid normalization: True |
| `artifact_integrity` | Checkpoints & Parquets unchanged (SHA-256) | **PASS** | Verified zero mutation |

---

## 4. Artifact Cryptographic Integrity (Zero Mutation)

Pre-audit and post-audit SHA-256 hashes confirm zero modifications to trained weights or extracted datasets:

```
Checkpoint: models/asl_landmark_mlp_best.pt
  - Size: 235896 bytes
  - SHA-256: dde94ee67f625cf04275b23166e695e3c279c431f2ba0c94ea4e6eaed72a55bd
  - Mutation Status: UNCHANGED

Train Parquet: data/processed/landmarks/full/landmarks_train.parquet
  - Size: 20741872 bytes
  - SHA-256: cbcb94a7b365edd686e0fa16f3ba264c7612ab09be233070648414986e1de694
  - Mutation Status: UNCHANGED

Validation Parquet: data/processed/landmarks/full/landmarks_validation.parquet
  - Size: 3455721 bytes
  - SHA-256: d53ca09d26ed32d2c7ff4f26db1b8ed65caabdf20a1b98a0640069ac7f673eb2
  - Mutation Status: UNCHANGED

Internal Test Parquet: data/processed/landmarks/full/landmarks_internal_test.parquet
  - Size: 1649972 bytes
  - SHA-256: 31cba0cf25fad203ea06eefba69fb700ff03e2c29a769a16f71ef66a4113d69b
  - Mutation Status: UNCHANGED

External Landmarks: data/processed/landmarks/external/external_hand_landmarks.parquet
  - Size: 308352 bytes
  - SHA-256: e25b1ae9823f46bbab0905f975c55972b12b7ed5ba7436c2ff5a8a83541d0e25
  - Mutation Status: UNCHANGED
```

---

## 5. Manual Launch Instructions

To launch the real-time recognition application for manual physical webcam testing:

```bash
# Default camera (index 0)
.venv\Scripts\python.exe src/realtime/app.py

# Specify alternative camera index (e.g. index 1)
.venv\Scripts\python.exe src/realtime/app.py --camera 1

# Optional customized temporal thresholds
.venv\Scripts\python.exe src/realtime/app.py --camera 0 --dwell 0.60 --release 0.25 --min-conf 0.70
```

### Application Keyboard Controls

- **`Q` or `ESC`**: Safely close camera, release MediaPipe detector, and quit application.
- **`C`**: Clear committed text buffer.
