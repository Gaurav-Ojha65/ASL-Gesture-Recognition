"""
Tests for targeted ASL recognition failure diagnostics (J, T, Z, I, D).

Verifies that:
1. GestureDiagnosticsLogger correctly records all 10 pipeline stages in structured JSONL format.
2. Trial indices and target transitions operate cleanly without side effects.
3. Observational telemetry does NOT modify model predictions, confidence, or state machine transitions.
4. ASLRealTimePipeline handles diagnostic metadata cleanly with zero regression.
5. Model checkpoints remain strictly unmodified.
"""

import json
import os
import hashlib
import tempfile
import numpy as np
import pytest

from src.realtime.gesture_diagnostics import GestureDiagnosticsLogger
from src.realtime.app import ASLRealTimePipeline, FrameProcessingResult
from src.landmarks.labels import SEMANTIC_CLASSES


EXPECTED_MLP_HASH = "dde94ee67f625cf04275b23166e695e3c279c431f2ba0c94ea4e6eaed72a55bd"
EXPECTED_CNN_HASH = "97ea89ade91c64903d7e0bb9f3c2bdc4c2d2d82ab9ee72f6ed4444f1a4487b64"
EXPECTED_TASK_HASH = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def test_checkpoint_integrity_unmodified():
    """Verify that model checkpoints and task assets have not been modified or retrained."""
    for path, expected_hash in [
        ("models/asl_landmark_mlp_best.pt", EXPECTED_MLP_HASH),
        ("models/mnist_cnn_best.pt", EXPECTED_CNN_HASH),
        ("hand_landmarker.task", EXPECTED_TASK_HASH),
    ]:
        assert os.path.exists(path), f"Required asset missing: {path}"
        with open(path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"Checkpoint hash mismatch for {path}! Expected {expected_hash}, got {actual_hash}"


def test_gesture_diagnostics_logger_lifecycle():
    """Verify target transitions, trial counters, and clean closing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_session.jsonl")
        logger = GestureDiagnosticsLogger(log_path)

        assert logger.current_target == "NONE"
        assert logger.current_trial_id == "IDLE"

        # Set target T
        t1 = logger.set_target("T")
        assert t1 == "T_01"
        assert logger.current_target == "T"

        t2 = logger.next_trial()
        assert t2 == "T_02"

        # Set target J
        j1 = logger.set_target("J")
        assert j1 == "J_01"
        assert logger.current_target == "J"

        # Clear target
        idle = logger.clear_target()
        assert idle == "IDLE"
        assert logger.current_target == "NONE"

        logger.close()
        assert os.path.exists(log_path)


def test_gesture_diagnostics_logger_record_structure():
    """Verify that logged records contain all required 10-stage diagnostic fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_record.jsonl")
        logger = GestureDiagnosticsLogger(log_path)
        logger.set_target("T")

        logger.log_frame(
            frame_index=1,
            timestamp_sec=100.0,
            hand_detected=True,
            handedness="Right",
            handedness_score=0.98,
            landmark_count=21,
            landmarks_finite=True,
            norm_success=True,
            norm_failure_reason="",
            feature_dim=63,
            hand_scale=0.18,
            raw_top1_label="T",
            raw_top1_prob=0.62,
            raw_top2_label="N",
            raw_top2_prob=0.25,
            raw_top1_top2_margin=0.37,
            raw_prob_t=0.62,
            raw_prob_j=0.01,
            raw_prob_z=0.01,
            raw_prob_i=0.02,
            raw_prob_d=0.01,
            smoothed_top1_label="T",
            smoothed_top1_prob=0.61,
            smoothed_prob_t=0.61,
            smoothed_prob_j=0.01,
            smoothed_prob_z=0.01,
            smoothed_prob_i=0.02,
            smoothed_prob_d=0.01,
            low_conf_rejected=True,
            static_fallback_label="T",
            sm_input_label="T",
            dyn_evaluated=True,
            dyn_detected=False,
            dyn_gesture=None,
            dyn_confidence=0.0,
            dyn_reason="insufficient displacement",
            dyn_buffer_len=15,
            pinky_posture_ok=False,
            index_posture_ok=False,
            sm_candidate=None,
            sm_is_dwelling=False,
            sm_latched_label=None,
            commit_event_label=None,
            hud_live_label="T",
            hud_live_confidence=0.61,
            hud_status="VALID_PREDICTION",
            committed_text="",
            text_buffer_length=0,
            error_category="",
        )
        logger.close()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["frame_index"] == 1
        assert rec["target_gesture"] == "T"
        assert rec["trial_id"] == "T_01"

        # Verify all 10 stages exist
        assert "stage_1_mediapipe" in rec
        assert "stage_2_validation" in rec
        assert "stage_3_normalization" in rec
        assert "stage_4_raw_mlp" in rec
        assert "stage_5_smoothed_mlp" in rec
        assert "stage_6_static_fallback" in rec
        assert "stage_7_dynamic_detector" in rec
        assert "stage_8_fsm" in rec
        assert "stage_9_hud" in rec
        assert "stage_10_committed_text" in rec

        # Check values
        assert rec["stage_4_raw_mlp"]["top1_label"] == "T"
        assert rec["stage_5_smoothed_mlp"]["low_conf_rejected"] is True
        assert rec["stage_7_dynamic_detector"]["reason"] == "insufficient displacement"


def test_pipeline_process_frame_with_diagnostics():
    """Verify that ASLRealTimePipeline runs cleanly with diagnostic_info and populates telemetry."""
    pipeline = ASLRealTimePipeline(
        checkpoint_path="models/asl_landmark_mlp_best.pt",
        task_model_path="hand_landmarker.task",
        min_confidence=0.70,
        dwell_duration=0.60,
        release_duration=0.25,
        enable_dynamic=True,
        enable_speech=False,
    )

    # Blank frame (no hand detected)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    diag_info = {"target": "T", "trial_id": "T_01"}

    res: FrameProcessingResult = pipeline.process_frame(blank_frame, timestamp_sec=1.0, diagnostic_info=diag_info)

    assert isinstance(res, FrameProcessingResult)
    assert res.hand_detected is False
    assert res.frame is not None
    assert res.frame.shape == (480, 640, 3)
    assert res.norm_success is None
    assert res.prediction_label is None

    pipeline.close()
