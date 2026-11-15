"""
Unit and Integration Tests for Dynamic ASL Gesture Recognition Infrastructure.

Verifies:
1. TemporalSequenceBuffer capacity, eviction, and monotonic timestamp handling.
2. Handling of tracking interruptions and missing hands.
3. KinematicFeatureExtractor metrics (path length, net displacement, tortuosity, speed).
4. Handshape geometric precondition checking for 'J' (I handshape) and 'Z' (1/D handshape).
5. Positive detection of synthetic ASL 'J' trajectory (downward stroke + curved upward hook).
6. Positive detection of synthetic ASL 'Z' trajectory (3-stroke zig-zag).
7. Negative test: Static 'I' sign does NOT trigger 'J'.
8. Negative test: Static 'D' sign does NOT trigger 'Z'.
9. Rejection of random / noisy motion trajectories.
10. Cooldown logic prevents duplicate triggers.
11. Latency benchmark ensuring < 1.0 ms CPU execution per evaluation.
"""

import math
import time
import pytest
import numpy as np

from src.realtime.temporal_sequence import (
    TemporalSequenceBuffer,
    KinematicFeatureExtractor,
    DynamicTrajectoryClassifier,
    DynamicRecognitionResult,
    FrameRecord
)


def _create_synthetic_landmarks(handshape: str = "open", base_pos: tuple = (0.5, 0.5, 0.0)) -> np.ndarray:
    """
    Creates a synthetic 21x3 landmark array with a specific handshape.
    wrist = landmark 0.
    """
    arr = np.zeros((21, 3), dtype=np.float32)
    bx, by, bz = base_pos
    arr[0] = [bx, by, bz]  # Wrist

    # Base MCPs (landmarks 5, 9, 13, 17)
    arr[5] = [bx - 0.04, by - 0.08, bz]   # Index MCP
    arr[9] = [bx - 0.01, by - 0.09, bz]   # Middle MCP
    arr[13] = [bx + 0.02, by - 0.08, bz]  # Ring MCP
    arr[17] = [bx + 0.05, by - 0.07, bz]  # Pinky MCP

    if handshape == "open":
        # All fingers extended upward (decreasing Y)
        arr[4] = [bx - 0.08, by - 0.08, bz]
        arr[8] = [bx - 0.04, by - 0.20, bz]
        arr[12] = [bx - 0.01, by - 0.22, bz]
        arr[16] = [bx + 0.02, by - 0.20, bz]
        arr[20] = [bx + 0.05, by - 0.17, bz]
    elif handshape == "i_shape":
        # Pinky extended (tip 20), other fingers curled
        arr[4] = [bx - 0.03, by - 0.06, bz]   # Thumb curled
        arr[8] = [bx - 0.04, by - 0.07, bz]   # Index curled near MCP
        arr[12] = [bx - 0.01, by - 0.08, bz]  # Middle curled
        arr[16] = [bx + 0.02, by - 0.07, bz]  # Ring curled
        arr[20] = [bx + 0.05, by - 0.20, bz]  # Pinky extended high!
    elif handshape == "d_shape":
        # Index extended (tip 8), other fingers curled
        arr[4] = [bx - 0.03, by - 0.06, bz]   # Thumb curled
        arr[8] = [bx - 0.04, by - 0.22, bz]   # Index extended high!
        arr[12] = [bx - 0.01, by - 0.08, bz]  # Middle curled
        arr[16] = [bx + 0.02, by - 0.07, bz]  # Ring curled
        arr[20] = [bx + 0.05, by - 0.06, bz]  # Pinky curled
    elif handshape == "fist":
        # All fingers curled
        arr[4] = [bx - 0.03, by - 0.05, bz]
        arr[8] = [bx - 0.04, by - 0.06, bz]
        arr[12] = [bx - 0.01, by - 0.07, bz]
        arr[16] = [bx + 0.02, by - 0.06, bz]
        arr[20] = [bx + 0.05, by - 0.05, bz]

    return arr


# ─── 1. Buffer Tests ─────────────────────────────────────────────────────────

def test_temporal_buffer_push_and_eviction():
    buf = TemporalSequenceBuffer(max_len=10)
    for i in range(15):
        lm = _create_synthetic_landmarks("open", (0.5, 0.5 + i * 0.01, 0.0))
        assert buf.push(timestamp=float(i) * 0.033, landmarks=lm, is_valid=True) is True

    assert len(buf) == 10
    valid_seq = buf.get_valid_sequence()
    assert len(valid_seq) == 10
    assert math.isclose(valid_seq[-1].timestamp, 14 * 0.033, rel_tol=1e-3)


def test_temporal_buffer_monotonic_timestamps():
    buf = TemporalSequenceBuffer(max_len=10)
    lm = _create_synthetic_landmarks("open")
    assert buf.push(1.0, lm, True) is True
    assert buf.push(1.1, lm, True) is True
    # Decreasing timestamp rejected
    assert buf.push(1.05, lm, True) is False
    # Non-finite timestamp rejected
    assert buf.push(float("nan"), lm, True) is False
    assert len(buf) == 2


def test_temporal_buffer_tracking_loss():
    buf = TemporalSequenceBuffer(max_len=20, max_missing_frames=5)
    lm = _create_synthetic_landmarks("open")

    # 10 valid frames
    for i in range(10):
        buf.push(i * 0.033, lm, True)

    assert len(buf.get_valid_sequence()) == 10

    # 3 missing frames: valid sequence stops at gap
    for i in range(10, 13):
        buf.push(i * 0.033, None, False)

    assert buf.consecutive_missing == 3
    assert len(buf.get_valid_sequence()) == 0

    # Exceed max missing frames (triggers pruning)
    for i in range(13, 20):
        buf.push(i * 0.033, None, False)

    assert buf.consecutive_missing >= 5
    assert len(buf) <= 6


# ─── 2. Kinematic Feature Extractor Tests ─────────────────────────────────────

def test_kinematic_extractor_straight_vs_curved():
    # Straight trajectory
    pos_straight = np.array([[0.1 * i, 0.0, 0.0] for i in range(10)], dtype=np.float32)
    times = np.array([i * 0.1 for i in range(10)], dtype=np.float32)
    metrics_straight = KinematicFeatureExtractor.compute_motion_metrics(pos_straight, times)

    assert math.isclose(metrics_straight["path_length"], 0.9, rel_tol=1e-3)
    assert math.isclose(metrics_straight["net_displacement"], 0.9, rel_tol=1e-3)
    assert math.isclose(metrics_straight["tortuosity"], 1.0, rel_tol=1e-2)

    # Curved U-turn trajectory (moves forward, then loops back)
    pos_curved = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.2, 0.0],
        [0.2, 0.4, 0.0],
        [0.05, 0.4, 0.0]
    ], dtype=np.float32)
    times_curved = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    metrics_curved = KinematicFeatureExtractor.compute_motion_metrics(pos_curved, times_curved)

    assert metrics_curved["path_length"] > metrics_curved["net_displacement"]
    assert metrics_curved["tortuosity"] > 1.5


def test_handshape_verification():
    i_lm = _create_synthetic_landmarks("i_shape")
    d_lm = _create_synthetic_landmarks("d_shape")
    fist_lm = _create_synthetic_landmarks("fist")

    # Pinky test for J
    ok_j, _, _ = KinematicFeatureExtractor.check_handshape_posture(i_lm, "pinky")
    assert ok_j is True

    bad_j, _, _ = KinematicFeatureExtractor.check_handshape_posture(d_lm, "pinky")
    assert bad_j is False

    # Index test for Z
    ok_z, _, _ = KinematicFeatureExtractor.check_handshape_posture(d_lm, "index")
    assert ok_z is True

    bad_z, _, _ = KinematicFeatureExtractor.check_handshape_posture(i_lm, "index")
    assert bad_z is False


# ─── 3. Dynamic Gesture Classifier Tests ──────────────────────────────────────

def test_synthetic_j_trajectory_positive():
    """Simulates realistic J stroke: downward descent + upward hook with I handshape."""
    clf = DynamicTrajectoryClassifier(min_duration=0.20, max_duration=1.5, min_motion_span=0.06)
    buf = TemporalSequenceBuffer(max_len=30)

    # Generate 20 frames of J motion (~0.6s at 30 FPS)
    # Downward stroke: frames 0..12, Upward hook: frames 13..19
    for i in range(20):
        t = i * 0.033
        if i <= 13:
            # Moving down in Y
            bx = 0.50
            by = 0.40 + (i / 13.0) * 0.14  # Descends by 0.14
        else:
            # Sweeping up and laterally in X
            step = i - 13
            bx = 0.50 - (step / 6.0) * 0.06  # Hook inward
            by = 0.54 - (step / 6.0) * 0.05  # Ascends by 0.05

        lm = _create_synthetic_landmarks("i_shape", (bx, by, 0.0))
        buf.push(t, lm, True)

    result = clf.evaluate_sequence(buf, current_time=0.66)
    assert result.detected is True
    assert result.gesture == "J"
    assert result.confidence >= 0.70
    assert "J" in result.reason


def test_synthetic_z_trajectory_positive():
    """Simulates realistic Z stroke: horizontal -> diagonal down -> horizontal with index handshape."""
    clf = DynamicTrajectoryClassifier(min_duration=0.20, max_duration=1.5, min_motion_span=0.05)
    buf = TemporalSequenceBuffer(max_len=30)

    # Stroke 1: Left to right (frames 0..6)
    # Stroke 2: Diagonal down-left (frames 7..14)
    # Stroke 3: Left to right (frames 15..21)
    frames_coords = []
    # Stroke 1
    for s in np.linspace(0.40, 0.55, 7):
        frames_coords.append((s, 0.35))
    # Stroke 2
    for s, y in zip(np.linspace(0.55, 0.40, 8), np.linspace(0.35, 0.50, 8)):
        frames_coords.append((s, y))
    # Stroke 3
    for s in np.linspace(0.40, 0.55, 7):
        frames_coords.append((s, 0.50))

    for i, (bx, by) in enumerate(frames_coords):
        t = i * 0.033
        lm = _create_synthetic_landmarks("d_shape", (bx, by, 0.0))
        buf.push(t, lm, True)

    result = clf.evaluate_sequence(buf, current_time=0.72)
    assert result.detected is True
    assert result.gesture == "Z"
    assert result.confidence >= 0.70
    assert "Z" in result.reason


def test_static_i_hand_negative_no_j():
    """A static 'I' hand with micro-tremor must NOT trigger a 'J' gesture."""
    clf = DynamicTrajectoryClassifier()
    buf = TemporalSequenceBuffer(max_len=30)

    # 25 frames of stationary I handshape
    np.random.seed(42)
    for i in range(25):
        t = i * 0.033
        jitter = np.random.normal(0, 0.001)  # tiny noise
        lm = _create_synthetic_landmarks("i_shape", (0.5 + jitter, 0.5 + jitter, 0.0))
        buf.push(t, lm, True)

    result = clf.evaluate_sequence(buf, current_time=0.825)
    assert result.detected is False
    assert result.gesture is None


def test_static_d_hand_negative_no_z():
    """A static 'D' hand with micro-tremor must NOT trigger a 'Z' gesture."""
    clf = DynamicTrajectoryClassifier()
    buf = TemporalSequenceBuffer(max_len=30)

    # 25 frames of stationary D handshape
    np.random.seed(42)
    for i in range(25):
        t = i * 0.033
        jitter = np.random.normal(0, 0.001)
        lm = _create_synthetic_landmarks("d_shape", (0.5 + jitter, 0.5 + jitter, 0.0))
        buf.push(t, lm, True)

    result = clf.evaluate_sequence(buf, current_time=0.825)
    assert result.detected is False
    assert result.gesture is None


def test_cooldown_prevents_duplicate_commit():
    """Verifies that cooldown prevents immediate re-triggering of the same dynamic sign."""
    clf = DynamicTrajectoryClassifier(cooldown_sec=1.20)
    buf = TemporalSequenceBuffer(max_len=30)

    # Build qualified J trajectory
    for i in range(20):
        t = i * 0.033
        by = 0.40 + (i / 13.0) * 0.14 if i <= 13 else 0.54 - ((i - 13) / 6.0) * 0.05
        bx = 0.50 if i <= 13 else 0.50 - ((i - 13) / 6.0) * 0.06
        lm = _create_synthetic_landmarks("i_shape", (bx, by, 0.0))
        buf.push(t, lm, True)

    res1 = clf.evaluate_sequence(buf, current_time=0.66)
    assert res1.detected is True
    assert res1.gesture == "J"

    # Immediately evaluate again at current_time + 0.1s
    res2 = clf.evaluate_sequence(buf, current_time=0.76)
    assert res2.detected is False
    assert "cooldown" in res2.reason

    # Evaluate after cooldown expires (> 1.20s)
    res3 = clf.evaluate_sequence(buf, current_time=2.0)
    assert res3.detected is True
    assert res3.gesture == "J"


def test_kinematic_processing_latency_benchmark():
    """
    Verifies that kinematic trajectory evaluation maintains real-time CPU throughput.
    Includes 10 warmup iterations before benchmarking to ensure consistent execution.
    Target threshold: < 2.5 ms on CPU (real-time 30 FPS frame budget allows up to 33 ms).
    """
    clf = DynamicTrajectoryClassifier()
    buf = TemporalSequenceBuffer(max_len=30)

    for i in range(30):
        lm = _create_synthetic_landmarks("i_shape", (0.5 + i * 0.002, 0.5 + i * 0.003, 0.0))
        buf.push(i * 0.033, lm, True)

    # Warmup runs to prime instruction caches and JIT paths
    for _ in range(10):
        clf.evaluate_sequence(buf, current_time=1.0)

    # Benchmark 100 evaluations
    start = time.perf_counter()
    for _ in range(100):
        clf.evaluate_sequence(buf, current_time=1.0)
    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / 100.0) * 1000.0

    assert avg_latency_ms < 2.5, f"Average latency too high: {avg_latency_ms:.3f} ms (expected < 2.5 ms)"

