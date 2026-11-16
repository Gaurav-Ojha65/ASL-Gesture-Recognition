"""
Targeted ASL Gesture Diagnostics Logger for J, T, Z, I, and D.

Provides structured, observational telemetry capturing every stage of the real-time pipeline:
MediaPipe detection -> landmark validation -> normalization -> raw MLP output ->
smoothed output -> static fallback -> dynamic detector -> FSM -> HUD -> committed text.

Designed to diagnose targeted recognition failures without modifying recognition behavior,
altering model probabilities, or saving raw images/video.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np


class GestureDiagnosticsLogger:
    """
    Writes machine-readable JSONL diagnostic logs for targeted gesture trials.
    Completely observational: never alters predictions, confidence, or state machine.
    """

    SUPPORTED_TARGETS = ["T", "J", "Z", "I", "D", "NONE"]

    def __init__(self, output_path: str, flush_every: int = 1):
        """
        Args:
            output_path: Path to the .jsonl file to write.
            flush_every: Flush file after every N writes (default: 1 for immediate telemetry).
        """
        self.output_path = output_path
        self.flush_every = flush_every
        self._write_count = 0

        self.current_target: str = "NONE"
        self.trial_counts: Dict[str, int] = {k: 0 for k in self.SUPPORTED_TARGETS if k != "NONE"}
        self.current_trial_id: str = "IDLE"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._fh = open(output_path, "w", encoding="utf-8")

    def set_target(self, target: str) -> str:
        """
        Sets the active target gesture and starts the next trial index for it.
        Returns the new trial_id.
        """
        target = target.strip().upper()
        if target not in self.SUPPORTED_TARGETS:
            target = "NONE"

        self.current_target = target
        if target == "NONE":
            self.current_trial_id = "IDLE"
        else:
            self.trial_counts[target] = self.trial_counts.get(target, 0) + 1
            idx = self.trial_counts[target]
            self.current_trial_id = f"{target}_{idx:02d}"

        return self.current_trial_id

    def next_trial(self) -> str:
        """Increments the trial counter for the current active target."""
        if self.current_target != "NONE":
            self.trial_counts[self.current_target] = self.trial_counts.get(self.current_target, 0) + 1
            idx = self.trial_counts[self.current_target]
            self.current_trial_id = f"{self.current_target}_{idx:02d}"
        return self.current_trial_id

    def clear_target(self) -> str:
        """Resets active target to NONE / IDLE."""
        self.current_target = "NONE"
        self.current_trial_id = "IDLE"
        return self.current_trial_id

    def log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        # Stage 1: MediaPipe Detection
        hand_detected: bool,
        handedness: Optional[str] = None,
        handedness_score: Optional[float] = None,
        landmark_count: int = 0,
        # Stage 2: Validation
        landmarks_finite: bool = True,
        norm_success: bool = False,
        norm_failure_reason: str = "",
        # Stage 3: Normalization
        feature_dim: int = 0,
        hand_scale: Optional[float] = None,
        # Stage 4: Raw MLP
        raw_top1_label: Optional[str] = None,
        raw_top1_prob: Optional[float] = None,
        raw_top2_label: Optional[str] = None,
        raw_top2_prob: Optional[float] = None,
        raw_top1_top2_margin: Optional[float] = None,
        raw_prob_t: Optional[float] = None,
        raw_prob_j: Optional[float] = None,
        raw_prob_z: Optional[float] = None,
        raw_prob_i: Optional[float] = None,
        raw_prob_d: Optional[float] = None,
        # Stage 5: Smoothed MLP
        smoothed_top1_label: Optional[str] = None,
        smoothed_top1_prob: Optional[float] = None,
        smoothed_prob_t: Optional[float] = None,
        smoothed_prob_j: Optional[float] = None,
        smoothed_prob_z: Optional[float] = None,
        smoothed_prob_i: Optional[float] = None,
        smoothed_prob_d: Optional[float] = None,
        low_conf_rejected: bool = False,
        # Stage 6: Static Fallback
        static_fallback_label: Optional[str] = None,
        sm_input_label: Optional[str] = None,
        # Stage 7: Dynamic Detector
        dyn_evaluated: bool = False,
        dyn_detected: bool = False,
        dyn_gesture: Optional[str] = None,
        dyn_confidence: float = 0.0,
        dyn_reason: str = "",
        dyn_buffer_len: int = 0,
        pinky_posture_ok: Optional[bool] = None,
        index_posture_ok: Optional[bool] = None,
        # Stage 8: State Machine (FSM)
        sm_candidate: Optional[str] = None,
        sm_is_dwelling: bool = False,
        sm_latched_label: Optional[str] = None,
        commit_event_label: Optional[str] = None,
        # Stage 9: HUD Display
        hud_live_label: Optional[str] = None,
        hud_live_confidence: float = 0.0,
        hud_status: str = "",
        # Stage 10: Committed Text
        committed_text: str = "",
        text_buffer_length: int = 0,
        # Error metadata
        error_category: str = "",
    ) -> None:
        """Writes one structured JSONL record for the current frame."""
        record = {
            "timestamp_sec": float(timestamp_sec),
            "frame_index": int(frame_index),
            "trial_id": self.current_trial_id,
            "target_gesture": self.current_target,
            "stage_1_mediapipe": {
                "hand_detected": bool(hand_detected),
                "handedness": handedness or "",
                "handedness_score": float(handedness_score) if handedness_score is not None else None,
                "landmark_count": int(landmark_count),
            },
            "stage_2_validation": {
                "landmarks_finite": bool(landmarks_finite),
                "norm_success": bool(norm_success),
                "norm_failure_reason": str(norm_failure_reason),
            },
            "stage_3_normalization": {
                "feature_dim": int(feature_dim),
                "hand_scale": float(hand_scale) if hand_scale is not None else None,
            },
            "stage_4_raw_mlp": {
                "top1_label": raw_top1_label or "",
                "top1_prob": float(raw_top1_prob) if raw_top1_prob is not None else None,
                "top2_label": raw_top2_label or "",
                "top2_prob": float(raw_top2_prob) if raw_top2_prob is not None else None,
                "top1_top2_margin": float(raw_top1_top2_margin) if raw_top1_top2_margin is not None else None,
                "prob_T": float(raw_prob_t) if raw_prob_t is not None else None,
                "prob_J": float(raw_prob_j) if raw_prob_j is not None else None,
                "prob_Z": float(raw_prob_z) if raw_prob_z is not None else None,
                "prob_I": float(raw_prob_i) if raw_prob_i is not None else None,
                "prob_D": float(raw_prob_d) if raw_prob_d is not None else None,
            },
            "stage_5_smoothed_mlp": {
                "top1_label": smoothed_top1_label or "",
                "top1_prob": float(smoothed_top1_prob) if smoothed_top1_prob is not None else None,
                "prob_T": float(smoothed_prob_t) if smoothed_prob_t is not None else None,
                "prob_J": float(smoothed_prob_j) if smoothed_prob_j is not None else None,
                "prob_Z": float(smoothed_prob_z) if smoothed_prob_z is not None else None,
                "prob_I": float(smoothed_prob_i) if smoothed_prob_i is not None else None,
                "prob_D": float(smoothed_prob_d) if smoothed_prob_d is not None else None,
                "low_conf_rejected": bool(low_conf_rejected),
            },
            "stage_6_static_fallback": {
                "predicted_label": static_fallback_label or "",
                "sm_input_label": sm_input_label or "",
                "is_remapped": bool(static_fallback_label != sm_input_label),
            },
            "stage_7_dynamic_detector": {
                "evaluated": bool(dyn_evaluated),
                "detected": bool(dyn_detected),
                "gesture": dyn_gesture or "",
                "confidence": float(dyn_confidence),
                "reason": str(dyn_reason),
                "buffer_len": int(dyn_buffer_len),
                "pinky_posture_ok": pinky_posture_ok,
                "index_posture_ok": index_posture_ok,
            },
            "stage_8_fsm": {
                "candidate": sm_candidate or "",
                "is_dwelling": bool(sm_is_dwelling),
                "latched_label": sm_latched_label or "",
                "commit_event_label": commit_event_label or "",
            },
            "stage_9_hud": {
                "live_label": hud_live_label or "",
                "live_confidence": float(hud_live_confidence),
                "status": str(hud_status),
            },
            "stage_10_committed_text": {
                "committed_text": str(committed_text),
                "text_buffer_length": int(text_buffer_length),
            },
            "error_category": str(error_category),
        }

        self._fh.write(json.dumps(record) + "\n")
        self._write_count += 1
        if self._write_count % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        """Flushes and cleanly closes the diagnostic file handle."""
        try:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
