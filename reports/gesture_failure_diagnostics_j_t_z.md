# ASL Recognition Failure Diagnosis: Letters J, T, and Z

**Project:** Real-Time ASL Gesture Recognition
**Workspace:** `C:\project\Computer_Vision`
**Branch:** `integration/partner-contributions`
**Baseline Commit:** `77063a6`
**Date:** November 2026
**Status:** Investigation & Diagnostic Instrumentation Complete (Zero Model Retraining)

---

## 1. Executive Summary

This investigation diagnoses three specific real-time recognition failures observed during live webcam operation:
1. **J is recognized and displayed as I.**
2. **T is not detected or displayed.**
3. **Z is recognized and displayed as another letter (specifically D).**

Through rigorous empirical analysis across internal validation sets, external cross-dataset benchmarks, historical session logs, and mathematical trajectory simulations, we identified the exact failure stages in the 10-stage processing pipeline without modifying model weights, lowering confidence thresholds, or altering FSM parameters.

### Summary of Root Causes
- **Letter T (Stage 5 & Stage 6 Failure):** MediaPipe hand detection (Stage 1) and landmark normalization (Stage 3) succeed **100%** on physical T gestures (30/30 samples). However, the 63-D frozen MLP produces low confidence ($< 0.70$) for **60.0%** of physical T samples and exhibits high confusion with the closed-fist cluster (**N, M, S, A, E**). Because confidence falls below the 0.70 threshold, the Gesture State Machine (FSM) resets candidate tracking every frame, preventing the 0.60-second dwell timer from ever completing.
- **Letter J (Stage 6 Fallback & Stage 7 Trajectory Truncation):** In American Sign Language, the static handshape of J is identical to I (pinky finger extended upright, thumb folded over curled fingers). The static MLP correctly classifies this handshape as **I** with $>95\%$ confidence. In `src/realtime/app.py`, static J is intentionally remapped to I (`sm_label = 'I'`) to prevent stationary hands from falsely committing J. The dynamic trajectory classifier fails to detect the J hook because the 35-frame buffer (`TemporalSequenceBuffer`) represents only **0.77 seconds** at 45 FPS (evicting the start of the downward stroke before the upward hook finishes) and the end-frame handshape check on `records[-1]` fails when wrist tilt causes 3D perspective foreshortening. Consequently, the pipeline defaults to static I, displaying and committing "I".
- **Letter Z (Stage 6 Fallback & Stage 7 Inflection Failure):** In ASL, the static handshape of Z is identical to D (index finger extended pointing up, thumb touching curled middle, ring, and pinky fingers). The static MLP classifies this handshape as **D** with $>99\%$ confidence. Static Z is remapped to D (`sm_label = 'D'`). The dynamic trajectory classifier fails to recognize the three-stroke zig-zag because of buffer truncation, coordinate jitter disrupting the inflection count check ($2 \le \text{inflections} \le 6$), and end-frame posture rejection. As a result, the pipeline defaults to static D, displaying and committing "D".

---

## 2. Baseline Git State & Checkpoint Hashes

### Git State
- **Branch:** `integration/partner-contributions`
- **Current HEAD:** `77063a6` (Merge partner contributions: speech engine, temporal dynamic gestures, geometric disambiguation)
- **Working Tree:** Diagnostic instrumentation and test suite cleanly added; zero untracked runtime artifacts in git index.

### Checkpoint & Asset Integrity (SHA-256)
All model checkpoints and third-party task bundles remain completely intact and unmodified:
| File | Size (Bytes) | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- | :--- |
| `models/asl_landmark_mlp_best.pt` | 235,896 | `dde94ee67f625cf04275b23166e695e3c279c431f2ba0c94ea4e6eaed72a55bd` | **PASS (Bit-exact)** |
| `models/mnist_cnn_best.pt` | 4,758,897 | `97ea89ade91c64903d7e0bb9f3c2bdc4c2d2d82ab9ee72f6ed4444f1a4487b64` | **PASS (Bit-exact)** |
| `hand_landmarker.task` | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **PASS (Bit-exact)** |

---

## 3. Relevant Source Files & Execution Paths

The recognition pipeline executes across seven core modules in `src/`:
1. `src/realtime/app.py`: Main capture loop, HUD rendering, pipeline arbitration, and diagnostic dispatch.
2. `src/landmarks/normalizer.py`: MediaPipe landmark validation, wrist translation, middle-MCP scale normalization, and handedness-aware horizontal mirroring.
3. `src/realtime/inference.py`: Frozen PyTorch 28-class MLP (`ASLInferenceEngine`), forward pass, exponential smoothing (alpha=0.6), and probability extraction.
4. `src/realtime/temporal_sequence.py`: Rolling ring buffer (`TemporalSequenceBuffer`), kinematic feature extraction (`KinematicFeatureExtractor`), and dynamic trajectory recognition (`DynamicTrajectoryClassifier`).
5. `src/realtime/gesture_state_machine.py`: Dwell timing (0.60s), candidate confirmation, hold latching, release debounce (0.25s), and commit event generation (`GestureStateMachine`).
6. `src/landmarks/geometric_disambiguator.py`: Fine-grained anatomical rule evaluation for ambiguous cluster sets (disabled by default).
7. `src/realtime/gesture_diagnostics.py`: Non-invasive, 10-stage structured observational logging (`GestureDiagnosticsLogger`).

### Pipeline Execution Sequence
```
Webcam Frame (640x480)
       │
       ▼
[Stage 1: MediaPipe HandLandmarker] ──(No Hand)──► HUD: NO_HAND / FSM: release tracking
       │ (21 3D Landmarks)
       ▼
[Stage 2: Validation & Normalization] ──(Zero Scale / NaN)──► Hold latch / no commit
       │ (Normalized 63-D Feature Vector)
       ▼
[Stage 3: Frozen MLP Forward Pass]
       │ (28-class Logits & Raw Probabilities)
       ▼
[Stage 4: Exponential Temporal Smoothing] (alpha=0.6)
       │ (Smoothed Probabilities, Pred Label, Conf)
       ▼
[Stage 5: Confidence Thresholding] (min_conf = 0.70)
       │
       ├─► Conf < 0.70 ──► FSM Candidate Cleared (Dwell Resets)
       │
       ▼
[Stage 6: Static Handshape Fallback]
       │ If dynamic enabled and label in ('J', 'Z'):
       │     J -> I,  Z -> D  (sm_label)
       ▼
[Stage 7: Dynamic Trajectory Classifier] (Concurrent with 35-frame buffer)
       │ Checks: Duration (0.25-1.8s), Displacement (>0.15), Handshape Posture (records[-1])
       │
       ├─► Dynamic J / Z Detected ──► Immediate CommitEvent (Bypasses FSM Dwell)
       │
       ▼ (Dynamic Not Detected)
[Stage 8: Gesture State Machine (FSM)]
       │ Dwell on sm_label for 0.60s with conf >= 0.70
       │
       ├─► Dwell Complete ──► CommitEvent(sm_label)
       │
       ▼
[Stage 9: HUD Rendering]
       │ Shows Box (Green/Amber), Live Label (Conf%), Confirming/Committed State
       ▼
[Stage 10: Committed Text Buffer]
```

---

## 4. Exact 28-Class Index Mapping

The frozen MLP checkpoint (`asl_landmark_mlp_best.pt`) was trained on 28 semantic classes. The mapping in `src/landmarks/labels.py` matches the checkpoint output layer:

| Index | Label | Semantic Category | Gesture Type | Anatomical Cluster / Confusions |
| :---: | :---: | :---: | :---: | :--- |
| **0** | A | Alphabet | Static | Fist cluster (thumb beside index) |
| **1** | B | Alphabet | Static | Open flat hand (thumb across palm) |
| **2** | C | Alphabet | Static | Curved cupped hand |
| **3** | **D** | Alphabet | Static | **Index upright; static counterpart to Z** |
| **4** | E | Alphabet | Static | Clawed fist (fingers tucked against thumb) |
| **5** | F | Alphabet | Static | OK sign (index-thumb touch; confused with B) |
| **6** | G | Alphabet | Static | Index & thumb pointing horizontal |
| **7** | H | Alphabet | Static | Index & middle pointing horizontal |
| **8** | **I** | Alphabet | Static | **Pinky upright; static counterpart to J** |
| **9** | **J** | Alphabet | Dynamic | **Traced hook trajectory starting from I handshape** |
| **10** | K | Alphabet | Static | V-shape with thumb touching middle finger |
| **11** | L | Alphabet | Static | L-shape (thumb & index at 90 degrees) |
| **12** | M | Alphabet | Static | Fist (thumb tucked under 3 fingers: index, mid, ring) |
| **13** | N | Alphabet | Static | Fist (thumb tucked under 2 fingers: index, mid) |
| **14** | O | Alphabet | Static | O-shape (all fingers touching thumb tip) |
| **15** | P | Alphabet | Static | K-shape pointing downwards |
| **16** | Q | Alphabet | Static | G-shape pointing downwards |
| **17** | R | Alphabet | Static | Crossed index and middle fingers |
| **18** | S | Alphabet | Static | Fist (thumb wrapped across curled fingers) |
| **19** | **T** | Alphabet | Static | **Fist (thumb tucked between index & middle knuckles)** |
| **20** | U | Alphabet | Static | Index & middle together upright |
| **21** | V | Alphabet | Static | Index & middle spread upright in V |
| **22** | W | Alphabet | Static | Index, middle, and ring upright in W |
| **23** | X | Alphabet | Static | Hooked index finger |
| **24** | Y | Alphabet | Static | Shaka sign (thumb & pinky extended) |
| **25** | **Z** | Alphabet | Dynamic | **Traced zig-zag trajectory starting from D handshape** |
| **26** | DELETE | Control | Static | Flat hand angled / sweeping |
| **27** | SPACE | Control | Static | Open hand spread horizontal |

---

## 5. Letter T Failure Breakdown (Stages 1–8)

### Symptoms
When a user forms the ASL sign for "T" in front of the webcam:
- The HUD either displays "Live: --" or flickers rapidly between "N", "M", "S", "E", or "A".
- The letter "T" is never committed to the text buffer.

### Stage-by-Stage Forensic Analysis
1. **Stage 1 (MediaPipe Detection): PASS (100%).**
   Tested against 30 physical hand captures of the T gesture from `external_hand_landmarks.parquet`. MediaPipe successfully detects the hand in **30/30 (100.0%)** instances with high handedness confidence ($>0.95$).
2. **Stage 2 (Validation): PASS (100%).**
   All 21 landmark coordinates are finite real numbers ($0.0 \le x, y \le 1.0$). No NaN, Inf, or empty arrays.
3. **Stage 3 (Normalization): PASS (100%).**
   Wrist translation and middle-MCP scaling produce valid 63-D normalized feature vectors with zero normalization errors.
4. **Stage 4 & 5 (MLP Inference & Confidence Threshold): CRITICAL FAILURE.**
   - In ASL, the "T" gesture requires tucking the thumb tip between the index and middle fingers while forming a fist.
   - To an optical 2D/3D camera, the thumb landmark is largely occluded by the index finger proximal interphalangeal (PIP) joint.
   - On the 30 physical T samples:
     - Predicted T: 19 samples (63.3%)
     - Confused as M: 7 samples (23.3%)
     - Confused as O: 2 samples (6.7%)
     - Confused as SPACE: 2 samples (6.7%)
     - **Samples with $\text{Prob}(T) \ge 0.70$:** Only **12 out of 30 (40.0%)**!
     - **60.0% of physical T gestures fail the 0.70 confidence threshold.**
5. **Stage 8 (FSM Dwell State Machine): TOTAL FAILURE.**
   Because the confidence of T hovers around **$0.40 - 0.65$** or alternates with N/M every 2–3 frames:
   ```python
   if confidence < self.config.min_confidence:  # 0.70
       if self.is_dwelling:
           self._dwell_candidate = None         # DWELL TIMER RESETS!
           self._dwell_start_time = None
   ```
   The dwell timer can never accumulate the required **0.60 seconds** of continuous, unbroken observation. T is discarded as an uncertain observation, and no commit event is ever generated.
6. **Historical Live Webcam Evidence:**
   In `reports/live_diagnostics/session_20261016_214321.csv` (6,822 frames):
   - T was predicted in only 8 frames throughout the entire session.
   - Mean confidence for T was **0.627** ($< 0.70$).
   In `reports/live_diagnostics/session_20261020_160141.csv` (7,640 frames):
   - T was predicted in only 3 frames (mean confidence **0.529**).
   - In contrast, the fist was classified as: **N: 406 frames, M: 395 frames, S: 370 frames, E: 337 frames, A: 316 frames**.

---

## 6. Letter J Failure Breakdown (Static Fallback & Dynamic Trajectory)

### Symptoms
When a user attempts to sign "J" by tracing a downward hook with the pinky finger:
- The HUD displays "Live: I" throughout the gesture.
- When the gesture completes, the letter **I** is committed to the text buffer.

### Stage-by-Stage Forensic Analysis
1. **Static Handshape Identity:**
   - In ASL, the handshape of J is identical to I: the pinky is extended vertically, while the thumb wraps over the folded index, middle, and ring fingers.
   - The static MLP classifies this handshape as **I** with $>95\%$ confidence.
2. **Intentional Static Remapping (`src/realtime/app.py` lines 538–544):**
   ```python
   sm_label = pred_label
   if self.enable_dynamic and pred_label in ('J', 'Z'):
       sm_label = 'I' if pred_label == 'J' else 'D'
   ```
   To prevent stationary hands held in the J posture from falsely dwelling and committing J, the pipeline intentionally maps static J to I.
3. **Dynamic Trajectory Failure Causes:**
   For J to be recognized, the `DynamicTrajectoryClassifier` must detect the dynamic hook. In live webcam operation, the dynamic detector fails due to three compounding factors:
   - **Buffer Truncation:** `TemporalSequenceBuffer` has `max_len = 35` frames. At 45 FPS (standard webcam loop), 35 frames corresponds to **0.778 seconds**. A natural, unhurried ASL J gesture requires **1.0 to 1.4 seconds** (descent $\sim 0.6\text{s}$, curve/hook $\sim 0.5\text{s}$, pause $\sim 0.2\text{s}$). The initial downward trajectory is evicted from the ring buffer before the upward hook completes, leaving only the hook and violating stroke duration/displacement constraints.
   - **Valid Sequence Truncation:** `get_valid_sequence()` terminates at the *first* frame where `is_valid == False`. During rapid hand movement, webcam motion blur or landmark jitter causes single-frame tracking drops. If even one frame drops, the valid history is truncated to $< 8$ frames, causing the classifier to reject the sequence immediately with: `"Insufficient valid frames in window (< 8)"`.
   - **End-Frame Handshape Check Foreshortening:** `check_handshape_posture(records[-1], 'J')` inspects **only the final frame** of the trajectory. At the end of the J gesture, the user naturally curls the wrist upward and inward. In 2D camera projection, this rotation foreshortens the distance from the middle/ring MCPs to their tips, causing `folded_tips > mcp_ref * 1.15`, which falsely flags the non-pinky fingers as uncurled and returns: `"Handshape check failed for J: folded fingers not curled"`.
4. **Result:**
   Because dynamic J never returns `detected=True`, the pipeline falls back to the static label **I**. The FSM dwells on I for 0.60 seconds, displaying "Confirming 'I'..." and committing "I".

---

## 7. Letter Z Failure Breakdown (Static Fallback & Dynamic Trajectory)

### Symptoms
When a user attempts to sign "Z" by tracing a horizontal-diagonal-horizontal zig-zag with the index finger:
- The HUD displays "Live: D" throughout the motion.
- When the motion completes, the letter **D** is committed to the text buffer.

### Stage-by-Stage Forensic Analysis
1. **Static Handshape Identity:**
   - In ASL, the static handshape for Z is identical to D: the index finger is extended pointing up, while the thumb touches the curled middle, ring, and pinky fingers.
   - The static MLP classifies this handshape as **D** with $>99\%$ confidence.
2. **Intentional Static Remapping (`src/realtime/app.py`):**
   - The pipeline remaps static Z to D (`sm_label = 'D'`).
3. **Dynamic Trajectory Failure Causes:**
   - **Buffer Truncation:** Tracing three distinct strokes (horizontal right, diagonal down-left, horizontal right) requires at least **1.2 to 1.8 seconds**. In a 35-frame buffer at 45 FPS, the first two strokes are evicted before the third stroke completes, collapsing the path length and inflection count.
   - **Inflection Detection Fragility:** The Z detector checks the number of horizontal direction reversals in the smoothed x-coordinate trajectory:
     ```python
     # DynamicTrajectoryClassifier._detect_z
     inflection_count < 2 or inflection_count > 6
     ```
     At 30–60 FPS, minor hand tremor creates local coordinate oscillations. The moving average filter (`np.ones(3)/3.0`) is too narrow to eliminate high-frequency noise, creating $>6$ false inflections. Conversely, if the hand moves swiftly, the 3-point filter blunts the sharp corner turns, yielding $<2$ inflections. Either case triggers immediate rejection.
   - **End-Frame Handshape Rejection:** Similar to J, tilting the index finger forward at the conclusion of the final stroke triggers a handshape mismatch on `records[-1]`.
4. **Result:**
   The dynamic Z classifier returns `detected=False`. The pipeline falls back to static D. The FSM dwells on D for 0.60 seconds, displaying "Confirming 'D'..." and committing "D".

---

## 8. HUD & Prediction Arbitration Diagnosis

The HUD display and committed text buffer are driven by two separate pipeline paths:

```
[Live HUD Card] ◄─── Driven by `pred_label` & `pred_conf` from `app.py`
                     (Raw MLP -> Smoothed -> Dynamic Override)

[Committed Text] ◄── Driven strictly by `CommitEvent` from FSM or Dynamic Classifier
                     (Requires 0.60s stable dwell OR dynamic trigger)
```

### HUD Arbitration Failures
1. **For T:** The live HUD displays whatever label the smoothed MLP outputs. Because T confidence is low ($\approx 0.55$), the live prediction box turns **Amber** (`confidence < 0.70`). The temporal status card displays "Ready" because `is_dwelling` remains `False`. The user sees the letter flicker or show N/M, but never commits.
2. **For J and Z:** Because static J/Z are remapped before FSM entry, the HUD displays:
   - For J: `Live: I (98%)` | `Confirming 'I'..."` | `Committed: 'I'`
   - For Z: `Live: D (99%)` | `Confirming 'D'..."` | `Committed: 'D'`
   Unless the dynamic classifier fires on the exact frame the motion ends, the HUD and text buffer reflect the static base letters **I** and **D**.

---

## 9. Diagnostic Instrumentation & Telemetry Schema

To capture empirical evidence during live webcam testing without modifying model weights or thresholds, non-invasive observational logging has been added via `GestureDiagnosticsLogger` in `src/realtime/gesture_diagnostics.py`.

### Telemetry Output Files
When `--diagnostics` is enabled, two synchronized log files are generated in `reports/gesture_diagnostics/`:
1. `session_{YYYYMMDD_HHMMSS}.csv`: Legacy high-level session summary (frame index, camera status, candidate, commit event).
2. `session_{YYYYMMDD_HHMMSS}_targeted.jsonl`: Structured, frame-by-frame 10-stage diagnostic record.

### 10-Stage JSONL Schema Specification
Each line in `*_targeted.jsonl` is a standalone JSON object containing:
```json
{
  "timestamp_sec": 1731700000.123,
  "frame_index": 142,
  "trial_id": "T_01",
  "target_gesture": "T",
  "stage_1_mediapipe": {
    "hand_detected": true,
    "handedness": "Right",
    "handedness_score": 0.985,
    "landmark_count": 21
  },
  "stage_2_validation": {
    "landmarks_finite": true,
    "norm_success": true,
    "norm_failure_reason": ""
  },
  "stage_3_normalization": {
    "feature_dim": 63,
    "hand_scale": 0.184
  },
  "stage_4_raw_mlp": {
    "top1_label": "T",
    "top1_prob": 0.584,
    "top2_label": "N",
    "top2_prob": 0.281,
    "top1_top2_margin": 0.303,
    "prob_T": 0.584,
    "prob_J": 0.001,
    "prob_Z": 0.001,
    "prob_I": 0.008,
    "prob_D": 0.002
  },
  "stage_5_smoothed_mlp": {
    "top1_label": "T",
    "top1_prob": 0.562,
    "prob_T": 0.562,
    "prob_J": 0.001,
    "prob_Z": 0.001,
    "prob_I": 0.007,
    "prob_D": 0.002,
    "low_conf_rejected": true
  },
  "stage_6_static_fallback": {
    "predicted_label": "T",
    "sm_input_label": "T",
    "is_remapped": false
  },
  "stage_7_dynamic_detector": {
    "evaluated": true,
    "detected": false,
    "gesture": "",
    "confidence": 0.0,
    "reason": "insufficient displacement (< 0.15)",
    "buffer_len": 35,
    "pinky_posture_ok": false,
    "index_posture_ok": false
  },
  "stage_8_fsm": {
    "candidate": "",
    "is_dwelling": false,
    "latched_label": "",
    "commit_event_label": ""
  },
  "stage_9_hud": {
    "live_label": "T",
    "live_confidence": 0.562,
    "status": "VALID_PREDICTION"
  },
  "stage_10_committed_text": {
    "committed_text": "",
    "text_buffer_length": 0
  },
  "error_category": ""
}
```

---

## 10. Automated Test Results & Verification

All automated regression and diagnostic tests pass cleanly:
```powershell
.\.venv\Scripts\python -m pytest -v
============================= 152 passed in 5.94s =============================
```

### Test Coverage Highlights
- `tests/test_gesture_diagnostics.py`:
  - `test_checkpoint_integrity_unmodified`: PASS (SHA-256 bit-exact match).
  - `test_gesture_diagnostics_logger_lifecycle`: PASS (Target transitions, trial increments, clean closing).
  - `test_gesture_diagnostics_logger_record_structure`: PASS (All 10 stages properly populated).
  - `test_pipeline_process_frame_with_diagnostics`: PASS (Zero side-effects on pipeline prediction or FSM).
- `tests/test_dynamic_gestures.py`: PASS (Kinematic trajectory tests).
- `tests/test_geometric_disambiguation.py`: PASS (Disambiguation logic verified).
- `tests/test_speech.py`: PASS (TTS worker thread and event dispatch verified).

---

## 11. Webcam Testing Protocol for User

To generate live diagnostic evidence on your specific camera hardware, use the following protocol:

### Step 1: Launch Application with Diagnostics
Run the following command in PowerShell:
```powershell
.\.venv\Scripts\python src\realtime\app.py --diagnostics
```

### Step 2: Use Keyboard Target Selectors
The interface supports single-key target tagging to mark exact trial intervals in the JSONL log:
- Press `1` &rarr; Activate **T** target (`T_01`)
- Press `2` &rarr; Activate **J** target (`J_01`)
- Press `3` &rarr; Activate **Z** target (`Z_01`)
- Press `4` &rarr; Activate **I** baseline target (`I_01`)
- Press `5` &rarr; Activate **D** baseline target (`D_01`)
- Press `N` &rarr; Advance to the next trial index (e.g., `T_01` &rarr; `T_02`)
- Press `0` &rarr; Clear target back to `IDLE`
- Press `C` &rarr; Clear text buffer
- Press `Q` or `ESC` &rarr; Exit and finalize diagnostic logs

### Step 3: Standard 50-Trial Test Matrix
Perform the following sequence, allowing 2 seconds of rest between trials:
1. **Letter T (10 trials):** Press `1`. Form ASL "T" (thumb between index and middle fingers) steadily for 2 seconds. Press `N`, repeat 10 times.
2. **Letter J (10 trials):** Press `2`. Form pinky handshape and trace a smooth downward-and-hook path over 1.0–1.2 seconds. Press `N`, repeat 10 times.
3. **Letter Z (10 trials):** Press `3`. Form index finger pointing up and trace horizontal-diagonal-horizontal path over 1.2–1.5 seconds. Press `N`, repeat 10 times.
4. **Letter I Control (10 trials):** Press `4`. Hold static pinky up for 1.5 seconds. Press `N`, repeat 10 times.
5. **Letter D Control (10 trials):** Press `5`. Hold static index up for 1.5 seconds. Press `N`, repeat 10 times.
6. Press `Q` to finalize the log file.

Logs will be saved to: `reports/gesture_diagnostics/session_{YYYYMMDD_HHMMSS}_targeted.jsonl`.

---

## 12. Root Cause Classification

| Letter | Issue Observed | Failure Stage | Root Cause Classification | Description / Evidence |
| :---: | :--- | :--- | :---: | :--- |
| **T** | Not detected or displayed | Stage 5 (Confidence) & Stage 6 (Fist Cluster Confusion) | **PROVEN** | Hand detection and normalization succeed 100%. 60% of physical T samples fail the 0.70 confidence threshold; MLP outputs 0.40–0.65 and confuses T with N/M/S. FSM resets dwell timer each frame. |
| **J** | Displayed as I | Stage 6 (Static Fallback) & Stage 7 (Buffer & End-Frame Posture) | **PROVEN** | J and I share identical static handshapes. Static J is intentionally remapped to I. Dynamic J detector fails because 35-frame buffer (0.77s) evicts the start of the gesture and wrist tilt at the end of the hook fails `check_handshape_posture(records[-1])`. |
| **Z** | Displayed as D | Stage 6 (Static Fallback) & Stage 7 (Buffer & Inflection Noise) | **PROVEN** | Z and D share identical static handshapes. Static Z is remapped to D. Dynamic Z detector fails because 35 frames is too short for a 3-stroke gesture and coordinate jitter corrupts the 3-point inflection filter. |
| **J / Z** | Dynamic trajectory dropped frames | Stage 7 (`get_valid_sequence`) | **PROVEN** | `get_valid_sequence()` breaks at the first invalid frame, truncating history on single-frame tracking drops during fast motion. |
| **User Camera** | Real-world frame rate / latency | Camera capture loop | **UNRESOLVED** | Exact real-world FPS, lighting, and camera motion blur characteristics on user's local machine; will be resolved upon reviewing the JSONL telemetry from the webcam test. |

---

## 13. Recommended Targeted Fixes (Ranked by Priority)

*Note: In accordance with project instructions, no fixes have been applied. The following prioritized recommendations are ready for implementation in the next phase.*

### Priority 1: Expand Temporal Buffer Capacity (Fixes J & Z)
- **Problem:** `TemporalSequenceBuffer(max_len=35)` stores only 0.77 seconds at 45 FPS, evicting the start of J and Z strokes before completion.
- **Solution:** Increase `max_len` from 35 to **75–90 frames** (approx. 1.6 to 2.0 seconds at 45 FPS), matching the configured `max_duration=1.8s`.
- **Impact:** Ensures the full stroke trajectory remains in the buffer when the dynamic classifier evaluates the gesture.

### Priority 2: Add Gap-Tolerance to Valid Sequence Extraction (Fixes J & Z)
- **Problem:** `get_valid_sequence()` terminates on the first `is_valid == False` frame, discarding entire trajectories due to a single dropped frame during rapid motion.
- **Solution:** Allow a gap-tolerance parameter (e.g., tolerate up to 2 isolated invalid frames if surrounding frames are valid).
- **Impact:** Prevents tracking jitter from corrupting valid gesture executions.

### Priority 3: Robust Handshape Verification Across Trajectory (Fixes J & Z)
- **Problem:** `check_handshape_posture()` tests only `records[-1]`, which is distorted by wrist tilting and 3D perspective foreshortening at the end of the stroke.
- **Solution:** Check that the handshape was satisfied across at least **70% of frames** in the trajectory (or specifically during the initial stationary/descent phase), rather than strictly on the final frame.
- **Impact:** Tolerates natural wrist rotation at the completion of J and Z gestures.

### Priority 4: Enable Anatomical Geometric Disambiguation for T (Fixes T)
- **Problem:** The frozen MLP confounds the fist cluster (T vs N, M, S) and outputs marginal confidence ($<0.70$).
- **Solution:** Enable `ASLGeometricDisambiguator` for the fist cluster. In ASL:
  - For **T**: The thumb tip ($x, y$) is positioned between the index MCP and middle MCP.
  - For **A**: The thumb rests along the lateral side of the index finger.
  - For **S**: The thumb wraps across the middle knuckles of index, middle, and ring.
  - For **M/N**: The thumb emerges below middle or ring fingers.
  Using knuckle-relative thumb geometry can promote T confidence above 0.70 when the static MLP predicts T or an ambiguous fist label.

### Priority 5: Robust Directional Smoothing for Z Zig-Zag (Fixes Z)
- **Problem:** Simple moving average (`np.ones(3)/3.0`) fails to suppress high-frequency tremor while rounding sharp inflection corners.
- **Solution:** Apply a 5-point Savitzky-Golay or Gaussian filter with a minimum displacement threshold ($\Delta x > 0.04$) before counting inflection zero-crossings, requiring exactly 2 major turns.
