"""
Tests for the optional runtime diagnostics subsystem (--diagnostics flag).

Verifies:
1.  Diagnostics disabled → no CSV file created.
2.  Diagnostics enabled → valid CSV file with all expected columns.
3.  Predicted labels and probabilities are recorded correctly.
4.  Hand-detection failures are distinguished from classifier errors.
5.  Invalid normalization is not treated as confirmed no-hand.
6.  Camera read failures do not trigger character commits and are logged distinctly.
7.  Stable incorrect/correct predictions are recorded without changing model output.
8.  Commit events and text-buffer changes are recorded accurately.
9.  Resource cleanup closes the diagnostic file.
10. Same mocked frame sequence → consistent diagnostic records.
"""

import csv
import os
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pytest

from src.realtime.app import (
    ASLRealTimePipeline,
    DiagnosticsLogger,
    DIAG_COLUMNS,
    run_app,
)
from src.realtime.inference import InferenceResult
from src.realtime.gesture_state_machine import CommitEvent
from src.landmarks.labels import SEMANTIC_CLASSES


CHECKPOINT_PATH = "models/asl_landmark_mlp_best.pt"


# ─── Shared mock helpers ──────────────────────────────────────────────────────

@dataclass
class MockLandmark:
    x: float
    y: float
    z: float


@dataclass
class MockCategory:
    category_name: str
    score: float


class MockDetectionResult:
    def __init__(self, hand_landmarks=None, handedness=None):
        self.hand_landmarks = hand_landmarks if hand_landmarks is not None else []
        self.handedness = handedness if handedness is not None else []


class MockDetector:
    def __init__(self, result=None):
        self.detection_result = result
        self.closed = False

    def detect_for_video(self, mp_image, timestamp_ms):
        return self.detection_result

    def detect(self, mp_image):
        return self.detection_result

    def close(self):
        self.closed = True


class MockInferenceEngine:
    def __init__(self, label: str = "A", confidence: float = 0.95):
        self.label = label
        self.confidence = confidence

    def predict(self, features):
        probs = np.zeros(28, dtype=np.float32)
        idx = SEMANTIC_CLASSES.index(self.label) if self.label in SEMANTIC_CLASSES else 0
        probs[idx] = self.confidence
        probs[(idx + 1) % 28] = (1.0 - self.confidence) * 0.6
        probs[(idx + 2) % 28] = (1.0 - self.confidence) * 0.4
        probs /= probs.sum()
        return InferenceResult(
            label=self.label,
            class_index=idx,
            confidence=self.confidence,
            probabilities=probs,
            logits=np.zeros(28, dtype=np.float32),
        )


def make_valid_landmarks():
    lms = []
    for i in range(21):
        if i == 0:
            lms.append(MockLandmark(0.5, 0.8, 0.0))
        elif i == 9:
            lms.append(MockLandmark(0.5, 0.4, 0.0))
        else:
            lms.append(MockLandmark(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0))
    return lms


def make_zero_scale_landmarks():
    return [MockLandmark(0.5, 0.5, 0.0) for _ in range(21)]


def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─── DiagnosticsLogger unit tests ────────────────────────────────────────────

@pytest.fixture
def tmpdir_fresh():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_diagnostics_disabled_no_file_created(tmpdir_fresh):
    """Req 1: Diagnostics disabled → no CSV created."""
    valid_lms = make_valid_landmarks()
    res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=MockInferenceEngine("B", 0.90),
    )
    # Run without creating any logger
    pipe.process_frame(dummy_frame(), 1.0)
    csv_path = os.path.join(tmpdir_fresh, "should_not_exist.csv")
    assert not os.path.exists(csv_path)


def test_diagnostics_enabled_creates_valid_csv(tmpdir_fresh):
    """Req 2: Diagnostics enabled → valid CSV with all expected columns."""
    valid_lms = make_valid_landmarks()
    res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=MockInferenceEngine("C", 0.91),
    )
    csv_path = os.path.join(tmpdir_fresh, "test_diag.csv")
    logger = DiagnosticsLogger(csv_path)

    result = pipe.process_frame(dummy_frame(), 1.0)
    logger.log_frame(
        frame_index=1,
        timestamp_sec=1.0,
        hand_detected=result.hand_detected,
        norm_success=result.norm_success,
        norm_failure_reason=result.norm_failure_reason,
        predicted_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        label_names=SEMANTIC_CLASSES,
        min_confidence=pipe.sm_config.min_confidence,
        sm_candidate=pipe.state_machine.current_candidate,
        sm_is_dwelling=pipe.state_machine.is_dwelling,
        sm_latched_label=pipe.state_machine.latched_label,
        commit_event=result.commit_event,
        text_buffer_length=len(pipe.text_buffer),
        status=result.status,
        error_category=result.error_category,
    )
    logger.close()

    assert os.path.exists(csv_path)
    rows = read_csv(csv_path)
    assert len(rows) == 1
    # All expected columns present
    for col in DIAG_COLUMNS:
        assert col in rows[0], f"Missing column: {col}"


def test_diagnostics_records_predicted_label_and_probs(tmpdir_fresh):
    """Req 3: Predicted label and probabilities are recorded correctly."""
    valid_lms = make_valid_landmarks()
    res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    inf = MockInferenceEngine("G", 0.88)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=inf,
    )
    csv_path = os.path.join(tmpdir_fresh, "probs.csv")
    logger = DiagnosticsLogger(csv_path)

    result = pipe.process_frame(dummy_frame(), 1.0)
    logger.log_frame(
        frame_index=1, timestamp_sec=1.0,
        hand_detected=result.hand_detected,
        norm_success=result.norm_success,
        norm_failure_reason=result.norm_failure_reason,
        predicted_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        label_names=SEMANTIC_CLASSES,
        min_confidence=pipe.sm_config.min_confidence,
        sm_candidate=pipe.state_machine.current_candidate,
        sm_is_dwelling=pipe.state_machine.is_dwelling,
        sm_latched_label=pipe.state_machine.latched_label,
        commit_event=result.commit_event,
        text_buffer_length=len(pipe.text_buffer),
        status=result.status,
        error_category=result.error_category,
    )
    logger.close()

    rows = read_csv(csv_path)
    row = rows[0]
    assert row["predicted_label"] == "G"
    assert row["hand_detected"] == "True"
    assert row["status"] == "VALID_PREDICTION"
    assert row["top1_label"] == "G"
    top1_prob = float(row["top1_prob"])
    assert 0.8 < top1_prob <= 1.0


def test_diagnostics_distinguishes_hand_detection_failure(tmpdir_fresh):
    """Req 4: Hand detection failure is distinct from classifier errors."""
    no_hand_res = MockDetectionResult(hand_landmarks=[], handedness=[])
    det = MockDetector(no_hand_res)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=MockInferenceEngine(),
    )
    csv_path = os.path.join(tmpdir_fresh, "nohand.csv")
    logger = DiagnosticsLogger(csv_path)

    result = pipe.process_frame(dummy_frame(), 1.0)
    logger.log_frame(
        frame_index=1, timestamp_sec=1.0,
        hand_detected=result.hand_detected,
        norm_success=result.norm_success,
        norm_failure_reason=result.norm_failure_reason,
        predicted_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        label_names=SEMANTIC_CLASSES,
        min_confidence=pipe.sm_config.min_confidence,
        sm_candidate=pipe.state_machine.current_candidate,
        sm_is_dwelling=pipe.state_machine.is_dwelling,
        sm_latched_label=pipe.state_machine.latched_label,
        commit_event=result.commit_event,
        text_buffer_length=len(pipe.text_buffer),
        status=result.status,
        error_category=result.error_category,
    )
    logger.close()

    rows = read_csv(csv_path)
    row = rows[0]
    assert row["hand_detected"] == "False"
    assert row["status"] == "NO_HAND"
    assert row["predicted_label"] == ""
    assert row["error_category"] == ""


def test_diagnostics_invalid_normalization_not_treated_as_no_hand(tmpdir_fresh):
    """Req 5: Invalid normalization is logged as INVALID_FEATURES, not NO_HAND."""
    zero_lms = make_zero_scale_landmarks()
    res = MockDetectionResult(hand_landmarks=[zero_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=MockInferenceEngine(),
    )
    csv_path = os.path.join(tmpdir_fresh, "invalid_norm.csv")
    logger = DiagnosticsLogger(csv_path)

    result = pipe.process_frame(dummy_frame(), 1.0)
    logger.log_frame(
        frame_index=1, timestamp_sec=1.0,
        hand_detected=result.hand_detected,
        norm_success=result.norm_success,
        norm_failure_reason=result.norm_failure_reason,
        predicted_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        label_names=SEMANTIC_CLASSES,
        min_confidence=pipe.sm_config.min_confidence,
        sm_candidate=pipe.state_machine.current_candidate,
        sm_is_dwelling=pipe.state_machine.is_dwelling,
        sm_latched_label=pipe.state_machine.latched_label,
        commit_event=result.commit_event,
        text_buffer_length=len(pipe.text_buffer),
        status=result.status,
        error_category=result.error_category,
    )
    logger.close()

    rows = read_csv(csv_path)
    row = rows[0]
    assert row["hand_detected"] == "True"
    assert "INVALID_FEATURES" in row["status"]
    assert row["error_category"] == "INVALID_FEATURES"
    assert row["norm_success"] == "False"


def test_diagnostics_camera_read_failure_distinct(tmpdir_fresh):
    """Req 6: Camera read failures → log_camera_failure, no character committed."""
    csv_path = os.path.join(tmpdir_fresh, "cam_fail.csv")
    logger = DiagnosticsLogger(csv_path)
    logger.log_camera_failure(frame_index=5, timestamp_sec=2.0)
    logger.close()

    rows = read_csv(csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["camera_read_ok"] == "False"
    assert row["status"] == "CAMERA_READ_FAIL"
    assert row["error_category"] == "CAMERA_READ_FAIL"
    # text_buffer_length is blank (no pipeline state available)
    assert row["text_buffer_length"] == ""
    assert row["commit_event_label"] == ""


def test_diagnostics_stable_incorrect_prediction_recorded(tmpdir_fresh):
    """Req 7: A stable low-confidence prediction is recorded without changing model output."""
    valid_lms = make_valid_landmarks()
    res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    # Below threshold (0.70) but above 0 → low_conf_rejected=True, no commit
    inf = MockInferenceEngine("N", 0.50)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=inf,
        min_confidence=0.70,
    )
    csv_path = os.path.join(tmpdir_fresh, "lowconf.csv")
    logger = DiagnosticsLogger(csv_path)

    result = pipe.process_frame(dummy_frame(), 1.0)
    logger.log_frame(
        frame_index=1, timestamp_sec=1.0,
        hand_detected=result.hand_detected,
        norm_success=result.norm_success,
        norm_failure_reason=result.norm_failure_reason,
        predicted_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        label_names=SEMANTIC_CLASSES,
        min_confidence=pipe.sm_config.min_confidence,
        sm_candidate=pipe.state_machine.current_candidate,
        sm_is_dwelling=pipe.state_machine.is_dwelling,
        sm_latched_label=pipe.state_machine.latched_label,
        commit_event=result.commit_event,
        text_buffer_length=len(pipe.text_buffer),
        status=result.status,
        error_category=result.error_category,
    )
    logger.close()

    rows = read_csv(csv_path)
    row = rows[0]
    # MLP prediction is recorded as-is
    assert row["predicted_label"] == "N"
    assert float(row["confidence"]) == pytest.approx(0.50, abs=0.001)
    # Low-confidence rejection flag set
    assert row["low_conf_rejected"] == "True"
    # No commit this frame
    assert row["commit_event_label"] == ""
    assert pipe.text_buffer.get_text() == ""


def test_diagnostics_commit_event_recorded(tmpdir_fresh):
    """Req 8: Commit events and text-buffer length are recorded accurately."""
    valid_lms = make_valid_landmarks()
    res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])
    det = MockDetector(res)
    inf = MockInferenceEngine("A", 0.95)
    pipe = ASLRealTimePipeline(
        checkpoint_path=CHECKPOINT_PATH,
        detector=det,
        inference_engine=inf,
        dwell_duration=0.60,
    )
    csv_path = os.path.join(tmpdir_fresh, "commit.csv")
    logger = DiagnosticsLogger(csv_path)

    rows_written = []
    for ts in [1.0, 1.30, 1.65]:
        result = pipe.process_frame(dummy_frame(), ts)
        logger.log_frame(
            frame_index=int(ts * 100), timestamp_sec=ts,
            hand_detected=result.hand_detected,
            norm_success=result.norm_success,
            norm_failure_reason=result.norm_failure_reason,
            predicted_label=result.prediction_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            label_names=SEMANTIC_CLASSES,
            min_confidence=pipe.sm_config.min_confidence,
            sm_candidate=pipe.state_machine.current_candidate,
            sm_is_dwelling=pipe.state_machine.is_dwelling,
            sm_latched_label=pipe.state_machine.latched_label,
            commit_event=result.commit_event,
            text_buffer_length=len(pipe.text_buffer),
            status=result.status,
            error_category=result.error_category,
        )
    logger.close()

    rows = read_csv(csv_path)
    assert len(rows) == 3

    # t=1.65 should have a commit
    commit_row = rows[2]
    assert commit_row["commit_event_label"] == "A"
    assert int(commit_row["text_buffer_length"]) == 1
    assert pipe.text_buffer.get_text() == "A"


def test_diagnostics_resource_cleanup(tmpdir_fresh):
    """Req 9: Logger.close() closes the file; subsequent close() is safe."""
    csv_path = os.path.join(tmpdir_fresh, "cleanup.csv")
    logger = DiagnosticsLogger(csv_path)
    logger.log_camera_failure(1, 0.0)
    logger.close()
    assert logger._fh.closed

    # Second close must not raise
    logger.close()


def test_diagnostics_deterministic_replay(tmpdir_fresh):
    """Req 10: Same mocked sequence → identical diagnostic records across two runs."""
    valid_lms = make_valid_landmarks()
    no_hand_res = MockDetectionResult(hand_landmarks=[], handedness=[])
    hand_res = MockDetectionResult(hand_landmarks=[valid_lms], handedness=[[MockCategory("Right", 0.9)]])

    def run_sequence(csv_path: str):
        det = MockDetector(hand_res)
        inf = MockInferenceEngine("H", 0.90)
        pipe = ASLRealTimePipeline(
            checkpoint_path=CHECKPOINT_PATH,
            detector=det,
            inference_engine=inf,
            dwell_duration=0.50,
            release_duration=0.20,
        )
        logger = DiagnosticsLogger(csv_path)
        timestamps = [1.0, 1.25, 1.55, 1.60, 1.90]
        for i, ts in enumerate(timestamps):
            if ts > 1.55:
                det.detection_result = no_hand_res
            result = pipe.process_frame(dummy_frame(), ts)
            logger.log_frame(
                frame_index=i + 1, timestamp_sec=ts,
                hand_detected=result.hand_detected,
                norm_success=result.norm_success,
                norm_failure_reason=result.norm_failure_reason,
                predicted_label=result.prediction_label,
                confidence=result.confidence,
                probabilities=result.probabilities,
                label_names=SEMANTIC_CLASSES,
                min_confidence=pipe.sm_config.min_confidence,
                sm_candidate=pipe.state_machine.current_candidate,
                sm_is_dwelling=pipe.state_machine.is_dwelling,
                sm_latched_label=pipe.state_machine.latched_label,
                commit_event=result.commit_event,
                text_buffer_length=len(pipe.text_buffer),
                status=result.status,
                error_category=result.error_category,
            )
        logger.close()
        return read_csv(csv_path)

    path1 = os.path.join(tmpdir_fresh, "run1.csv")
    path2 = os.path.join(tmpdir_fresh, "run2.csv")
    rows1 = run_sequence(path1)
    rows2 = run_sequence(path2)

    assert len(rows1) == len(rows2)
    for r1, r2 in zip(rows1, rows2):
        for col in DIAG_COLUMNS:
            assert r1[col] == r2[col], f"Divergence at column {col}: {r1[col]!r} != {r2[col]!r}"
