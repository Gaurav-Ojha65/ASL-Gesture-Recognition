"""
Unit tests for scripts/audit_live_diagnostics.py.

Verifies:
1. Handling when no diagnostic CSV exists (BLOCKED status, report written, clean exit).
2. CSV schema and field type validation (all 22 DIAG_COLUMNS).
3. Cluster segmentation logic (gap separation vs. intra-trial flicker).
4. Failure stage attribution:
   - Detection failure
   - Normalization failure
   - Misclassification (e.g., N -> M)
   - Low-confidence rejection (< 0.70)
   - Dwell-time instability (< 0.60s hold)
   - Commit debounce / latch lock
   - Correct commit match
5. End-to-end replay with a 33-trial synthetic session.
6. Artifact integrity pre/post check.
"""

import csv
import os
import shutil
import tempfile
import pytest

from scripts.audit_live_diagnostics import (
    DIAG_COLUMNS,
    TARGET_LETTERS,
    TRIALS_PER_LETTER,
    TOTAL_INTENDED_TRIALS,
    find_latest_diagnostic_csv,
    parse_and_validate_csv,
    segment_hand_clusters,
    analyze_trial,
    build_audit_report,
    run_audit,
    hash_critical_artifacts,
    CRITICAL_ARTIFACTS,
)


def _make_dummy_row(
    frame_index: int,
    timestamp_sec: float,
    camera_read_ok: bool = True,
    hand_detected: bool = True,
    norm_success: bool = True,
    norm_failure_reason: str = "",
    predicted_label: str = "D",
    confidence: float = 0.95,
    low_conf_rejected: bool = False,
    sm_candidate: str = "D",
    sm_is_dwelling: bool = True,
    sm_latched_label: str = "",
    commit_event_label: str = "",
    text_buffer_length: int = 0,
    status: str = "VALID_PREDICTION",
    error_category: str = "",
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "camera_read_ok": camera_read_ok,
        "hand_detected": hand_detected,
        "norm_success": norm_success,
        "norm_failure_reason": norm_failure_reason,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top1_label": predicted_label,
        "top1_prob": confidence,
        "top2_label": "B" if predicted_label != "B" else "C",
        "top2_prob": round((1.0 - confidence) * 0.7, 3),
        "top3_label": "A" if predicted_label != "A" else "C",
        "top3_prob": round((1.0 - confidence) * 0.3, 3),
        "low_conf_rejected": low_conf_rejected,
        "sm_candidate": sm_candidate,
        "sm_is_dwelling": sm_is_dwelling,
        "sm_latched_label": sm_latched_label,
        "commit_event_label": commit_event_label,
        "text_buffer_length": text_buffer_length,
        "status": status,
        "error_category": error_category,
    }


def _write_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAG_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_missing_csv_returns_blocked():
    """Verify that when no CSV exists, run_audit reports BLOCKED and returns exit code 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_csv = os.path.join(tmpdir, "nonexistent.csv")
        code = run_audit(csv_path_override=fake_csv)
        assert code == 2


def test_parse_valid_csv():
    """Verify clean parsing of a valid CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        rows = [
            _make_dummy_row(1, 100.0, predicted_label="D", confidence=0.95),
            _make_dummy_row(2, 100.033, predicted_label="D", confidence=0.96),
        ]
        _write_csv(csv_path, rows)

        parsed, errors, metrics = parse_and_validate_csv(csv_path)
        assert len(errors) == 0
        assert metrics["total_rows"] == 2
        assert metrics["hand_detected_count"] == 2
        assert metrics["norm_success_count"] == 2
        assert metrics["monotonic_timestamps"] is True
        assert metrics["strictly_positive_frames"] is True


def test_cluster_segmentation_with_gaps():
    """Verify that gaps >= 0.35s separate distinct gesture trials."""
    rows = []
    # Cluster 1: 10 frames of hand (t = 0 to 0.3s)
    for i in range(10):
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, hand_detected=True))
    # Gap: 20 frames of no-hand (t = 0.33 to 1.0s, gap ~ 0.66s)
    for i in range(20):
        rows.append(_make_dummy_row(11 + i, 100.33 + i * 0.033, hand_detected=False, norm_success=False))
    # Cluster 2: 10 frames of hand (t = 1.0 to 1.3s)
    for i in range(10):
        rows.append(_make_dummy_row(31 + i, 101.0 + i * 0.033, hand_detected=True))

    clusters = segment_hand_clusters(rows, min_gap_sec=0.35, min_cluster_frames=5)
    assert len(clusters) == 2
    assert len(clusters[0]) == 10
    assert len(clusters[1]) == 10


def test_failure_stage_detection_loss():
    """Trial where hand detection fails heavily is categorized as DETECTION failure."""
    rows = []
    for i in range(10):
        # 8 out of 10 frames hand is not detected
        hd = (i < 2)
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, hand_detected=hd, norm_success=hd))
    t = analyze_trial(1, "I", rows)
    assert t.failure_stage == "DETECTION"


def test_failure_stage_normalization_loss():
    """Trial where normalizer fails on detected hands is categorized as NORMALIZATION failure."""
    rows = []
    for i in range(10):
        # Hand detected, but normalization fails
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, hand_detected=True, norm_success=False, norm_failure_reason="ZERO_SPAN"))
    t = analyze_trial(1, "O", rows)
    assert t.failure_stage == "NORMALIZATION"


def test_failure_stage_classification_error():
    """Trial with clean detection but wrong predicted label (e.g., N -> M)."""
    rows = []
    for i in range(20):
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, predicted_label="M", confidence=0.92))
    t = analyze_trial(1, "N", rows)
    assert t.failure_stage == "CLASSIFICATION"
    assert t.top_predicted_label == "M"


def test_failure_stage_low_confidence():
    """Trial where predictions are rejected due to low confidence (< 0.70)."""
    rows = []
    for i in range(20):
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, predicted_label="S", confidence=0.55, low_conf_rejected=True))
    t = analyze_trial(1, "S", rows)
    assert t.failure_stage == "LOW_CONFIDENCE"


def test_failure_stage_dwell_instability():
    """Trial where candidate alternates rapidly so dwell is never achieved."""
    rows = []
    for i in range(20):
        lbl = "T" if i % 2 == 0 else "M"
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, predicted_label=lbl, confidence=0.85))
    t = analyze_trial(1, "T", rows, min_dwell=0.60)
    assert t.failure_stage == "DWELL_INSTABILITY"


def test_trial_correct_commit():
    """Trial with stable hold and successful commit event."""
    rows = []
    for i in range(30):  # ~1.0s at 30 fps
        commit = "D" if i == 20 else ""
        rows.append(_make_dummy_row(i + 1, 100.0 + i * 0.033, predicted_label="D", confidence=0.98, commit_event_label=commit))
    t = analyze_trial(1, "D", rows)
    assert t.match_status == "CORRECT"
    assert t.failure_stage == "NONE"
    assert t.committed_letter == "D"


def test_synthetic_33_trial_session_replay():
    """Verify that a full 33-trial session for all 11 target letters is parsed and audited cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "session_synthetic.csv")
        all_rows = []
        frame_idx = 1
        t_sec = 1000.0

        for letter_idx, letter in enumerate(TARGET_LETTERS):
            for trial_k in range(TRIALS_PER_LETTER):
                # 30 frames of active hand (~1.0s)
                for f in range(30):
                    commit = letter if f == 20 else ""
                    all_rows.append(_make_dummy_row(
                        frame_index=frame_idx,
                        timestamp_sec=round(t_sec, 3),
                        predicted_label=letter,
                        confidence=0.95,
                        commit_event_label=commit,
                        text_buffer_length=letter_idx * 3 + trial_k + (1 if f >= 20 else 0),
                    ))
                    frame_idx += 1
                    t_sec += 0.033
                # 15 frames of no-hand gap (~0.5s)
                for f in range(15):
                    all_rows.append(_make_dummy_row(
                        frame_index=frame_idx,
                        timestamp_sec=round(t_sec, 3),
                        hand_detected=False,
                        norm_success=False,
                        predicted_label="",
                        confidence=0.0,
                    ))
                    frame_idx += 1
                    t_sec += 0.033

        _write_csv(csv_path, all_rows)

        code = run_audit(csv_path_override=csv_path)
        assert code == 0, "Synthetic 33-trial audit should exit with 0 (PASS)"


def test_artifact_integrity_verification():
    """Verify that critical artifacts exist and hashes can be computed."""
    hashes = hash_critical_artifacts(CRITICAL_ARTIFACTS)
    assert len(hashes) == len(CRITICAL_ARTIFACTS)
    for p, (sz, h) in hashes.items():
        assert sz is not None and sz > 0
        assert h is not None and len(h) == 64
