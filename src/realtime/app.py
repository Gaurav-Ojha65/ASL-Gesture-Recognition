"""
Real-Time ASL Gesture Recognition Application.

Connects:
OpenCV Webcam Capture -> MediaPipe Hand Landmarker -> LandmarkNormalizer ->
Frozen 28-Class MLP (ASLInferenceEngine) -> Temporal GestureStateMachine -> TextBuffer ->
OpenCV Visual HUD Display.

Keyboard Controls:
- 'Q' or ESC: Quit safely
- 'C': Clear text buffer

Optional diagnostics (--diagnostics flag):
- Writes a per-session CSV to reports/live_diagnostics/
- Disabled by default; observational only.
- Does not capture or save webcam images or video.
"""

import csv
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.landmarks.normalizer import LandmarkNormalizer
from src.realtime.inference import ASLInferenceEngine, InferenceResult
from src.realtime.gesture_state_machine import (
    GestureStateMachine,
    StateMachineConfig,
    CommitEvent
)
from src.realtime.text_buffer import TextBuffer


# ─── Diagnostic CSV schema ────────────────────────────────────────────────────
DIAG_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "camera_read_ok",          # bool: True if frame was read successfully
    "hand_detected",           # bool: True if MediaPipe found a hand
    "norm_success",            # bool: True if normalisation produced 63-D features
    "norm_failure_reason",     # str | ""
    "predicted_label",         # str | ""
    "confidence",              # float | ""
    "top1_label",              # str | ""
    "top1_prob",               # float | ""
    "top2_label",              # str | ""
    "top2_prob",               # float | ""
    "top3_label",              # str | ""
    "top3_prob",               # float | ""
    "low_conf_rejected",       # bool: True if hand detected but confidence < threshold
    "sm_candidate",            # str | "" – state-machine current candidate
    "sm_is_dwelling",          # bool
    "sm_latched_label",        # str | "" – currently latched (hold-debounce) label
    "commit_event_label",      # str | "" – label if a commit was emitted this frame
    "text_buffer_length",      # int
    "status",                  # str – pipeline status string (NO_HAND, VALID_PREDICTION, …)
    "error_category",          # str | "" – CAMERA_READ_FAIL, DETECTOR_ERROR, INFERENCE_ERROR, INVALID_FEATURES, etc.
]


class DiagnosticsLogger:
    """
    Writes one CSV row per successfully-read frame.
    Camera-read failures are recorded as a distinct event (camera_read_ok=False).

    Notes
    -----
    - Observational only: does not alter predictions, state-machine, or text-buffer.
    - No webcam images or video are saved.
    - The file is flushed every `flush_every` rows so partial sessions are recoverable.
    - Thread-safety is not required; the camera loop is single-threaded.
    """

    def __init__(self, output_path: str, flush_every: int = 20):
        self._output_path = output_path
        self._flush_every = flush_every
        self._row_count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._fh = open(output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=DIAG_COLUMNS)
        self._writer.writeheader()
        self._fh.flush()

    # ── public interface ──────────────────────────────────────────────────────

    def log_camera_failure(self, frame_index: int, timestamp_sec: float) -> None:
        """Records a frame where cap.read() returned False / None."""
        self._write_row({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "camera_read_ok": False,
            "hand_detected": "",
            "norm_success": "",
            "norm_failure_reason": "",
            "predicted_label": "",
            "confidence": "",
            "top1_label": "", "top1_prob": "",
            "top2_label": "", "top2_prob": "",
            "top3_label": "", "top3_prob": "",
            "low_conf_rejected": "",
            "sm_candidate": "",
            "sm_is_dwelling": "",
            "sm_latched_label": "",
            "commit_event_label": "",
            "text_buffer_length": "",
            "status": "CAMERA_READ_FAIL",
            "error_category": "CAMERA_READ_FAIL",
        })

    def log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        hand_detected: bool,
        norm_success: Optional[bool],
        norm_failure_reason: str,
        predicted_label: Optional[str],
        confidence: Optional[float],
        probabilities: Optional[np.ndarray],
        label_names: Optional[List[str]],   # ordered list of class names matching probabilities index
        min_confidence: float,
        sm_candidate: Optional[str],
        sm_is_dwelling: bool,
        sm_latched_label: Optional[str],
        commit_event: Optional[CommitEvent],
        text_buffer_length: int,
        status: str,
        error_category: str,
    ) -> None:
        """Records a normal processed frame (camera read succeeded)."""

        # Top-3 from probabilities
        top_labels = ["", "", ""]
        top_probs: List[Any] = ["", "", ""]
        if probabilities is not None and label_names is not None and len(probabilities) >= 3:
            top_idx = np.argsort(probabilities)[::-1][:3]
            for rank, idx in enumerate(top_idx):
                top_labels[rank] = label_names[idx]
                top_probs[rank] = float(probabilities[idx])

        # Low-confidence rejection: hand was present, MLP ran, but confidence < threshold
        low_conf_rejected: Any = ""
        if hand_detected and norm_success is True and predicted_label is not None:
            low_conf_rejected = bool(
                confidence is not None and confidence < min_confidence
            )

        self._write_row({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "camera_read_ok": True,
            "hand_detected": hand_detected,
            "norm_success": norm_success if norm_success is not None else "",
            "norm_failure_reason": norm_failure_reason,
            "predicted_label": predicted_label if predicted_label is not None else "",
            "confidence": confidence if confidence is not None else "",
            "top1_label": top_labels[0], "top1_prob": top_probs[0],
            "top2_label": top_labels[1], "top2_prob": top_probs[1],
            "top3_label": top_labels[2], "top3_prob": top_probs[2],
            "low_conf_rejected": low_conf_rejected,
            "sm_candidate": sm_candidate if sm_candidate is not None else "",
            "sm_is_dwelling": sm_is_dwelling,
            "sm_latched_label": sm_latched_label if sm_latched_label is not None else "",
            "commit_event_label": commit_event.label if commit_event is not None else "",
            "text_buffer_length": text_buffer_length,
            "status": status,
            "error_category": error_category,
        })

    def close(self) -> None:
        """Flushes and closes the diagnostic file."""
        try:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    # ── private ───────────────────────────────────────────────────────────────

    def _write_row(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % self._flush_every == 0:
            self._fh.flush()


# ─── Main data types ──────────────────────────────────────────────────────────

@dataclass
class FrameProcessingResult:
    """Structured result of processing a single video frame."""
    frame: np.ndarray
    status: str
    prediction_label: Optional[str]
    confidence: float
    hand_detected: bool
    landmarks: Optional[List[Dict[str, float]]]
    handedness: Optional[str]
    commit_event: Optional[CommitEvent]
    committed_text: str
    fps: float
    # diagnostics extras (always populated; top_probs is None when no MLP ran)
    probabilities: Optional[np.ndarray] = None
    norm_success: Optional[bool] = None
    norm_failure_reason: str = ""
    error_category: str = ""


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ASLRealTimePipeline:
    """
    Manages the integrated real-time ASL recognition pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/asl_landmark_mlp_best.pt",
        task_model_path: str = "hand_landmarker.task",
        dwell_duration: float = 0.60,
        release_duration: float = 0.25,
        min_confidence: float = 0.70,
        detector: Optional[vision.HandLandmarker] = None,
        inference_engine: Optional[ASLInferenceEngine] = None,
        mirror_feed: bool = True
    ):
        """
        Args:
            checkpoint_path: Path to PyTorch model checkpoint.
            task_model_path: Path to MediaPipe hand_landmarker.task.
            dwell_duration: Dwell confirmation duration in seconds (default: 0.60).
            release_duration: Sustained no-hand duration before re-arming in seconds (default: 0.25).
            min_confidence: Minimum prediction probability threshold (default: 0.70).
            detector: Optional pre-initialized MediaPipe HandLandmarker (useful for testing/mocking).
            inference_engine: Optional pre-initialized ASLInferenceEngine.
            mirror_feed: Whether to mirror webcam feed horizontally (default: True).
        """
        self.checkpoint_path = checkpoint_path
        self.task_model_path = task_model_path
        self.mirror_feed = mirror_feed

        # 1. Initialize Normalizer
        self.normalizer = LandmarkNormalizer()

        # 2. Initialize Inference Engine
        if inference_engine is not None:
            self.inference_engine = inference_engine
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            self.inference_engine = ASLInferenceEngine(checkpoint_path=checkpoint_path)

        # 3. Initialize MediaPipe Detector
        if detector is not None:
            self.detector = detector
            self._owns_detector = False
        else:
            if not os.path.exists(task_model_path):
                raise FileNotFoundError(f"MediaPipe task model not found at: {task_model_path}")
            base_options = python.BaseOptions(model_asset_path=task_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self._owns_detector = True

        # 4. Initialize State Machine
        self.sm_config = StateMachineConfig(
            dwell_duration=dwell_duration,
            release_duration=release_duration,
            min_confidence=min_confidence
        )
        self.state_machine = GestureStateMachine(config=self.sm_config)

        # 5. Initialize Text Buffer
        self.text_buffer = TextBuffer()

        # State tracking
        self._last_timestamp_ms: int = -1
        self._prev_frame_time: Optional[float] = None
        self._fps_filtered: float = 0.0

    def process_frame(self, frame_bgr: np.ndarray, timestamp_sec: float) -> FrameProcessingResult:
        """
        Processes a single input camera frame.

        Args:
            frame_bgr: BGR frame from OpenCV (H, W, 3).
            timestamp_sec: Monotonic timestamp in seconds.

        Returns:
            FrameProcessingResult containing annotations, predictions, and status.
        """
        # Calculate instantaneous FPS with exponential moving average
        if self._prev_frame_time is not None and timestamp_sec > self._prev_frame_time:
            dt = timestamp_sec - self._prev_frame_time
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            if self._fps_filtered <= 0.0:
                self._fps_filtered = instant_fps
            else:
                self._fps_filtered = 0.9 * self._fps_filtered + 0.1 * instant_fps
        self._prev_frame_time = timestamp_sec

        # 1. Flip frame horizontally if mirror mode is enabled
        if self.mirror_feed:
            frame_processed = cv2.flip(frame_bgr, 1)
        else:
            frame_processed = frame_bgr.copy()

        h, w = frame_processed.shape[:2]

        # 2. Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 3. Monotonic millisecond timestamp for MediaPipe VIDEO mode
        timestamp_ms = int(timestamp_sec * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # 4. MediaPipe Detection
        try:
            if hasattr(self.detector, 'detect_for_video'):
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            else:
                detection_result = self.detector.detect(mp_image)
        except Exception as e:
            # Detector exception: treat as uncertain observation without crashing
            return FrameProcessingResult(
                frame=frame_processed,
                status=f"DETECTOR_ERROR: {str(e)}",
                prediction_label=None,
                confidence=0.0,
                hand_detected=False,
                landmarks=None,
                handedness=None,
                commit_event=None,
                committed_text=self.text_buffer.get_text(),
                fps=self._fps_filtered,
                probabilities=None,
                norm_success=None,
                norm_failure_reason="",
                error_category="DETECTOR_ERROR",
            )

        # 5. Analyze Hand Presence
        has_hand = bool(
            detection_result
            and detection_result.hand_landmarks
            and len(detection_result.hand_landmarks) > 0
            and detection_result.handedness
            and len(detection_result.handedness) > 0
        )

        commit_event: Optional[CommitEvent] = None
        pred_label: Optional[str] = None
        pred_conf: float = 0.0
        landmarks_dict: Optional[List[Dict[str, float]]] = None
        handedness_cat: Optional[str] = None
        probabilities: Optional[np.ndarray] = None
        norm_success: Optional[bool] = None
        norm_failure_reason: str = ""
        error_category: str = ""

        if not has_hand:
            # Case A: No hand detected (Genuine absence)
            status = "NO_HAND"
            commit_event = self.state_machine.process_observation(
                label=None,
                confidence=0.0,
                hand_detected=False,
                timestamp=timestamp_sec
            )
        else:
            # Hand is present
            raw_landmarks = detection_result.hand_landmarks[0]
            landmarks_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in raw_landmarks]
            handedness_cat = detection_result.handedness[0][0].category_name

            # 6. Normalize landmarks
            is_valid, features, norm_reason = self.normalizer.normalize(landmarks_dict, handedness_cat)
            norm_success = is_valid
            norm_failure_reason = norm_reason if not is_valid else ""

            if not is_valid:
                # Case B: Hand detected, but features invalid (e.g. zero-scale)
                # Treat as uncertain observation without releasing the hold latch!
                status = f"INVALID_FEATURES: {norm_reason}"
                error_category = "INVALID_FEATURES"
                safe_label = self.state_machine.latched_label or self.state_machine.current_candidate or 'A'
                commit_event = self.state_machine.process_observation(
                    label=safe_label,
                    confidence=0.0,
                    hand_detected=True,
                    timestamp=timestamp_sec
                )
            else:
                # Case C: Valid hand & valid normalized 63-D features
                # Run frozen MLP inference
                try:
                    inf_result: InferenceResult = self.inference_engine.predict(features)
                    pred_label = inf_result.label
                    pred_conf = inf_result.confidence
                    probabilities = inf_result.probabilities

                    status = "VALID_PREDICTION"
                    commit_event = self.state_machine.process_observation(
                        label=pred_label,
                        confidence=pred_conf,
                        hand_detected=True,
                        timestamp=timestamp_sec
                    )
                except Exception as e:
                    status = f"INFERENCE_ERROR: {str(e)}"
                    error_category = "INFERENCE_ERROR"
                    safe_label = self.state_machine.latched_label or self.state_machine.current_candidate or 'A'
                    commit_event = self.state_machine.process_observation(
                        label=safe_label,
                        confidence=0.0,
                        hand_detected=True,
                        timestamp=timestamp_sec
                    )

        # 7. Consume commit event into text buffer
        if commit_event is not None:
            self.text_buffer.apply(commit_event)

        # 8. Render HUD
        annotated_frame = self.render_hud(
            frame_processed,
            status=status,
            prediction_label=pred_label,
            confidence=pred_conf,
            hand_detected=has_hand,
            landmarks=landmarks_dict,
            fps=self._fps_filtered
        )

        return FrameProcessingResult(
            frame=annotated_frame,
            status=status,
            prediction_label=pred_label,
            confidence=pred_conf,
            hand_detected=has_hand,
            landmarks=landmarks_dict,
            handedness=handedness_cat,
            commit_event=commit_event,
            committed_text=self.text_buffer.get_text(),
            fps=self._fps_filtered,
            probabilities=probabilities,
            norm_success=norm_success,
            norm_failure_reason=norm_failure_reason,
            error_category=error_category,
        )

    def render_hud(
        self,
        frame: np.ndarray,
        status: str,
        prediction_label: Optional[str],
        confidence: float,
        hand_detected: bool,
        landmarks: Optional[List[Dict[str, float]]],
        fps: float
    ) -> np.ndarray:
        """
        Renders HUD overlay onto the frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw hand bounding box & landmarks if present
        if hand_detected and landmarks:
            xs = [int(lm['x'] * w) for lm in landmarks]
            ys = [int(lm['y'] * h) for lm in landmarks]
            min_x, max_x = max(0, min(xs) - 15), min(w - 1, max(xs) + 15)
            min_y, max_y = max(0, min(ys) - 15), min(h - 1, max(ys) + 15)

            # Box color: Green if prediction is confident, else Amber
            box_color = (0, 220, 0) if (prediction_label and confidence >= self.sm_config.min_confidence) else (0, 165, 255)
            cv2.rectangle(out, (min_x, min_y), (max_x, max_y), box_color, 2)

            # Draw key landmarks (wrist=0, middle_mcp=9)
            wrist_pt = (xs[0], ys[0])
            mcp_pt = (xs[9], ys[9])
            cv2.circle(out, wrist_pt, 5, (255, 0, 0), -1)
            cv2.circle(out, mcp_pt, 5, (0, 0, 255), -1)
            cv2.line(out, wrist_pt, mcp_pt, (255, 255, 0), 2)

        # 2. Top Header HUD Bar (Glassmorphic semi-transparent overlay)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # Title
        cv2.putText(out, "ASL Gesture Recognition", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        # FPS Display
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        cv2.putText(out, fps_text, (w - 140, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        # 3. Live Prediction & State Card (Left side)
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (15, 65), (280, 175), (25, 30, 38), -1)
        cv2.addWeighted(overlay2, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (15, 65), (280, 175), (60, 70, 85), 1)

        # Detection Status
        status_color = (0, 220, 0) if hand_detected else (140, 140, 140)
        status_str = "HAND DETECTED" if hand_detected else "NO HAND"
        cv2.putText(out, f"Status: {status_str}", (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

        # Live Prediction Label & Confidence
        if hand_detected and prediction_label:
            conf_pct = int(confidence * 100)
            pred_color = (0, 255, 128) if confidence >= self.sm_config.min_confidence else (0, 180, 255)
            cv2.putText(out, f"Live: {prediction_label} ({conf_pct}%)", (25, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, pred_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "Live: --", (25, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 1, cv2.LINE_AA)

        # Temporal State (Dwelling vs Latched)
        if self.state_machine.is_dwelling:
            cand = self.state_machine.current_candidate
            state_text = f"Confirming '{cand}'..."
            state_col = (0, 215, 255)
        elif self.state_machine.is_latched:
            latch = self.state_machine.latched_label
            state_text = f"Committed: '{latch}' (Hold)"
            state_col = (100, 240, 100)
        else:
            state_text = "Ready"
            state_col = (180, 180, 180)
        cv2.putText(out, state_text, (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_col, 1, cv2.LINE_AA)

        # 4. Committed Text Buffer (Bottom prominent bar)
        overlay3 = out.copy()
        cv2.rectangle(overlay3, (15, h - 110), (w - 15, h - 45), (15, 20, 28), -1)
        cv2.addWeighted(overlay3, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (15, h - 110), (w - 15, h - 45), (0, 200, 255), 2)

        # Text label
        cv2.putText(out, "COMMITTED TEXT:", (25, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

        # Actual text content (with blinking cursor)
        curr_text = self.text_buffer.get_text()
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        display_str = curr_text if curr_text else "(buffer empty)"
        text_col = (255, 255, 255) if curr_text else (130, 130, 130)

        # Clip if too long for screen
        if len(display_str) > 45:
            display_str = "..." + display_str[-42:]
        display_str += cursor if curr_text else ""

        cv2.putText(out, display_str, (175, h - 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_col, 2, cv2.LINE_AA)

        # 5. Instructions footer
        cv2.putText(out, "Controls: [Q] / [ESC] Quit  |  [C] Clear Buffer", (25, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        return out

    def close(self) -> None:
        """Releases underlying MediaPipe detector if owned."""
        if hasattr(self, '_owns_detector') and self._owns_detector and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass


def run_app(
    camera_index: int = 0,
    checkpoint_path: str = "models/asl_landmark_mlp_best.pt",
    task_model_path: str = "hand_landmarker.task",
    dwell_duration: float = 0.60,
    release_duration: float = 0.25,
    min_confidence: float = 0.70,
    mirror_feed: bool = True,
    max_frames: Optional[int] = None,
    enable_diagnostics: bool = False,
    diagnostics_dir: str = "reports/live_diagnostics",
) -> int:
    """
    Main application runner.

    Opens camera, runs recognition loop, renders HUD, and handles controls.
    Returns 0 on clean exit, non-zero on startup or camera failure.

    Args:
        enable_diagnostics: When True, writes a per-session CSV under diagnostics_dir.
                            Disabled by default. Does NOT capture or store video/images.
        diagnostics_dir:    Directory for diagnostic CSVs (default: reports/live_diagnostics).
    """
    from src.landmarks.labels import SEMANTIC_CLASSES

    print("=" * 70)
    print("REAL-TIME ASL GESTURE RECOGNITION APPLICATION")
    print("=" * 70)
    print(f"Camera index:       {camera_index}")
    print(f"Checkpoint path:    {checkpoint_path}")
    print(f"MediaPipe task:     {task_model_path}")
    print(f"Dwell duration:     {dwell_duration}s")
    print(f"Release duration:   {release_duration}s")
    print(f"Min confidence:     {min_confidence}")
    print(f"Mirror feed:        {mirror_feed}")
    print(f"Diagnostics:        {'ENABLED' if enable_diagnostics else 'disabled'}")
    print("=" * 70)

    # 1. Verify and initialize pipeline
    try:
        pipeline = ASLRealTimePipeline(
            checkpoint_path=checkpoint_path,
            task_model_path=task_model_path,
            dwell_duration=dwell_duration,
            release_duration=release_duration,
            min_confidence=min_confidence,
            mirror_feed=mirror_feed
        )
    except FileNotFoundError as e:
        print(f"Startup Error: Missing required file: {e}")
        return 1
    except Exception as e:
        print(f"Startup Error: Failed to initialize pipeline: {e}")
        return 1

    # 2. Open camera capture
    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera Initialization Error: Could not open camera at index {camera_index}.")
        print("Please check that your webcam is connected, not in use by another application,")
        print("and that the specified camera index is correct.")
        pipeline.close()
        return 1

    # Set camera resolution (standard 640x480 for responsive real-time inference)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "ASL Real-Time Recognition Interface"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nApplication started successfully.")
    print("Keyboard Controls: Press [Q] or [ESC] to quit, [C] to clear buffer.\n")

    # 3. Optionally open diagnostics logger
    diag_logger: Optional[DiagnosticsLogger] = None
    if enable_diagnostics:
        session_ts = time.strftime("%Y%m%d_%H%M%S")
        diag_csv_path = os.path.join(diagnostics_dir, f"session_{session_ts}.csv")
        diag_logger = DiagnosticsLogger(diag_csv_path)
        print(f"Diagnostics: writing to {diag_csv_path}")

    frame_count = 0
    processed_frame_index = 0  # count only frames that passed cap.read()
    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1
            now = time.time()

            if not ret or frame is None:
                # An occasional failed frame read provides no reliable evidence of hand presence.
                # Log a brief warning and continue without corrupting the state machine.
                print("Warning: Failed to read frame from camera. Skipping frame...")
                if diag_logger is not None:
                    diag_logger.log_camera_failure(frame_index=frame_count, timestamp_sec=now)
                time.sleep(0.01)
                continue

            processed_frame_index += 1
            result = pipeline.process_frame(frame, now)

            # Diagnostics logging (observational; after process_frame so state is settled)
            if diag_logger is not None:
                diag_logger.log_frame(
                    frame_index=frame_count,
                    timestamp_sec=now,
                    hand_detected=result.hand_detected,
                    norm_success=result.norm_success,
                    norm_failure_reason=result.norm_failure_reason,
                    predicted_label=result.prediction_label,
                    confidence=result.confidence if result.prediction_label is not None else None,
                    probabilities=result.probabilities,
                    label_names=SEMANTIC_CLASSES,
                    min_confidence=pipeline.sm_config.min_confidence,
                    sm_candidate=pipeline.state_machine.current_candidate,
                    sm_is_dwelling=pipeline.state_machine.is_dwelling,
                    sm_latched_label=pipeline.state_machine.latched_label,
                    commit_event=result.commit_event,
                    text_buffer_length=len(pipeline.text_buffer),
                    status=result.status,
                    error_category=result.error_category,
                )

            cv2.imshow(window_name, result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("\nQuit requested by user.")
                break
            elif key in (ord('c'), ord('C')):
                pipeline.text_buffer.clear()
                print("Text buffer cleared.")

            if max_frames is not None and processed_frame_index >= max_frames:
                print(f"Reached maximum frame limit ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nRuntime Error in processing loop: {e}")
    finally:
        # Guaranteed cleanup path
        print("Releasing camera and OpenCV resources...")
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
        if diag_logger is not None:
            diag_logger.close()
            print(f"Diagnostics file closed.")
        print("Cleanup complete.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Real-Time ASL Gesture Recognition Interface")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--checkpoint", type=str, default="models/asl_landmark_mlp_best.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--task-model", type=str, default="hand_landmarker.task",
                        help="Path to MediaPipe hand_landmarker.task")
    parser.add_argument("--dwell", type=float, default=0.60,
                        help="Dwell duration in seconds (default: 0.60)")
    parser.add_argument("--release", type=float, default=0.25,
                        help="Release duration in seconds (default: 0.25)")
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Minimum prediction confidence (default: 0.70)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable horizontal camera feed mirroring")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional maximum frames to process (useful for headless testing)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Enable per-session diagnostic CSV logging (disabled by default)")
    parser.add_argument("--diagnostics-dir", type=str, default="reports/live_diagnostics",
                        help="Directory for diagnostic CSV files (default: reports/live_diagnostics)")

    args = parser.parse_args()

    exit_code = run_app(
        camera_index=args.camera,
        checkpoint_path=args.checkpoint,
        task_model_path=args.task_model,
        dwell_duration=args.dwell,
        release_duration=args.release,
        min_confidence=args.min_conf,
        mirror_feed=not args.no_mirror,
        max_frames=args.max_frames,
        enable_diagnostics=args.diagnostics,
        diagnostics_dir=args.diagnostics_dir,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
