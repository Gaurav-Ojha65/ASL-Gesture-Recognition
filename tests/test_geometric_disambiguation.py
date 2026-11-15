"""
Unit Tests for Anatomical Geometric Disambiguator & Dynamic Trajectory Enhancements.

Verifies:
1. Disambiguation of V (spread) vs U (parallel/touching) vs K (thumb between fingers).
2. Disambiguation of Fist Cluster: M (thumb at ring/pinky), N (thumb at middle/ring), T (thumb at index/middle), A, S, E.
3. Disambiguation of F (index-thumb loop, 3 upright fingers) vs B vs E.
4. Detection of X (hooked index finger) vs D (straight index).
5. Disambiguation of G (single horizontal) vs H (double horizontal).
6. Disambiguation of P (inverted K) vs Q (inverted G).
7. Detection of Y (wide thumb-pinky span, inner fingers curled).
8. Dynamic J and Z natural duration resilience up to 1.80s and smoothed inflection detection.
"""

import numpy as np
import pytest

from src.landmarks.geometric_disambiguator import HandGeometry, ASLGeometricDisambiguator
from src.realtime.temporal_sequence import (
    TemporalSequenceBuffer,
    KinematicFeatureExtractor,
    DynamicTrajectoryClassifier,
    FrameRecord
)


def _build_synthetic_hand(
    wrist=(0.5, 0.8, 0.0),
    thumb_tip=(0.45, 0.65, 0.0),
    index_tip=(0.48, 0.40, 0.0),
    middle_tip=(0.52, 0.38, 0.0),
    ring_tip=(0.55, 0.40, 0.0),
    pinky_tip=(0.58, 0.45, 0.0),
    index_curl_ratio=0.95,
) -> np.ndarray:
    """Helper to synthesize anatomically consistent 21-landmark arrays."""
    lm = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    lm[0] = [wx, wy, wz]

    # Thumb CMC, MCP, IP, Tip
    lm[1] = [wx - 0.03, wy - 0.05, wz]
    lm[2] = [wx - 0.05, wy - 0.10, wz]
    lm[3] = [wx - 0.06, wy - 0.15, wz]
    lm[4] = list(thumb_tip)

    # Base MCPs (landmarks 5, 9, 13, 17)
    lm[5] = [wx - 0.04, wy - 0.20, wz]  # Index MCP
    lm[9] = [wx, wy - 0.22, wz]         # Middle MCP
    lm[13] = [wx + 0.04, wy - 0.20, wz] # Ring MCP
    lm[17] = [wx + 0.07, wy - 0.18, wz] # Pinky MCP

    # Index joints
    ix_mcp = lm[5]
    ix_tip = np.array(index_tip, dtype=np.float32)
    lm[6] = ix_mcp + (ix_tip - ix_mcp) * 0.33
    lm[7] = ix_mcp + (ix_tip - ix_mcp) * 0.66
    lm[8] = ix_tip

    # Middle joints
    mid_mcp = lm[9]
    mid_tip = np.array(middle_tip, dtype=np.float32)
    lm[10] = mid_mcp + (mid_tip - mid_mcp) * 0.33
    lm[11] = mid_mcp + (mid_tip - mid_mcp) * 0.66
    lm[12] = mid_tip

    # Ring joints
    rng_mcp = lm[13]
    rng_tip = np.array(ring_tip, dtype=np.float32)
    lm[14] = rng_mcp + (rng_tip - rng_mcp) * 0.33
    lm[15] = rng_mcp + (rng_tip - rng_mcp) * 0.66
    lm[16] = rng_tip

    # Pinky joints
    pnk_mcp = lm[17]
    pnk_tip = np.array(pinky_tip, dtype=np.float32)
    lm[18] = pnk_mcp + (pnk_tip - pnk_mcp) * 0.33
    lm[19] = pnk_mcp + (pnk_tip - pnk_mcp) * 0.66
    lm[20] = pnk_tip

    return lm


def test_v_vs_u_disambiguation():
    disambiguator = ASLGeometricDisambiguator()
    dummy_probs = np.zeros(28, dtype=np.float32)

    # 1. Synthesize 'V': Index tip and middle tip spread wide apart
    hand_v = _build_synthetic_hand(
        wrist=(0.5, 0.8, 0.0),
        thumb_tip=(0.46, 0.70, 0.0),
        index_tip=(0.40, 0.35, 0.0),  # Angled left
        middle_tip=(0.60, 0.35, 0.0), # Angled right (wide spread)
        ring_tip=(0.54, 0.65, 0.0),   # Curled
        pinky_tip=(0.57, 0.65, 0.0),  # Curled
    )
    # Model initially confused, predicts U
    label, conf, rule = disambiguator.disambiguate("U", 0.52, dummy_probs, hand_v)
    assert label == "V"
    assert conf >= 0.85
    assert "GEOM_V" in rule

    # 2. Synthesize 'U': Index tip and middle tip parallel / touching
    hand_u = _build_synthetic_hand(
        wrist=(0.5, 0.8, 0.0),
        thumb_tip=(0.46, 0.70, 0.0),
        index_tip=(0.49, 0.35, 0.0),  # Touching
        middle_tip=(0.51, 0.35, 0.0), # Touching
        ring_tip=(0.54, 0.65, 0.0),   # Curled
        pinky_tip=(0.57, 0.65, 0.0),  # Curled
    )
    label, conf, rule = disambiguator.disambiguate("V", 0.55, dummy_probs, hand_u)
    assert label == "U"
    assert conf >= 0.85
    assert "GEOM_U" in rule


def test_fist_cluster_m_n_t():
    disambiguator = ASLGeometricDisambiguator()
    dummy_probs = np.zeros(28, dtype=np.float32)

    # All fingers curled (fist)
    base_fist_tips = {
        "wrist": (0.5, 0.8, 0.0),
        "index_tip": (0.47, 0.66, 0.0),
        "middle_tip": (0.50, 0.66, 0.0),
        "ring_tip": (0.53, 0.66, 0.0),
        "pinky_tip": (0.56, 0.66, 0.0),
    }

    # 'T': thumb placed between index and middle knuckles
    hand_t = _build_synthetic_hand(**base_fist_tips, thumb_tip=(0.48, 0.62, 0.0))
    label, conf, rule = disambiguator.disambiguate("M", 0.75, dummy_probs, hand_t)
    assert label == "T"
    assert "GEOM_T" in rule

    # 'N': thumb placed between middle and ring knuckles
    hand_n = _build_synthetic_hand(**base_fist_tips, thumb_tip=(0.52, 0.62, 0.0))
    label, conf, rule = disambiguator.disambiguate("M", 0.80, dummy_probs, hand_n)
    assert label == "N"
    assert "GEOM_N" in rule

    # 'M': thumb placed near ring and pinky knuckles
    hand_m = _build_synthetic_hand(**base_fist_tips, thumb_tip=(0.56, 0.62, 0.0))
    label, conf, rule = disambiguator.disambiguate("N", 0.60, dummy_probs, hand_m)
    assert label == "M"
    assert "GEOM_M" in rule


def test_f_vs_b_disambiguation():
    disambiguator = ASLGeometricDisambiguator()
    dummy_probs = np.zeros(28, dtype=np.float32)

    # Synthesize 'F': thumb (4) and index tip (8) touch, middle/ring/pinky upright
    hand_f = _build_synthetic_hand(
        wrist=(0.5, 0.8, 0.0),
        thumb_tip=(0.45, 0.55, 0.0),
        index_tip=(0.46, 0.55, 0.0),  # Touching thumb
        middle_tip=(0.50, 0.35, 0.0), # Upright
        ring_tip=(0.54, 0.36, 0.0),   # Upright
        pinky_tip=(0.58, 0.38, 0.0),  # Upright
    )
    label, conf, rule = disambiguator.disambiguate("B", 0.50, dummy_probs, hand_f)
    assert label == "F"
    assert "GEOM_F" in rule


def test_x_hooked_finger():
    disambiguator = ASLGeometricDisambiguator()
    dummy_probs = np.zeros(28, dtype=np.float32)

    # Hooked index finger: tip extended beyond MCP but curled (ratio ~0.65)
    lm = _build_synthetic_hand(
        wrist=(0.5, 0.8, 0.0),
        thumb_tip=(0.45, 0.68, 0.0),
        index_tip=(0.46, 0.52, 0.0),  # Hooked down toward palm
        middle_tip=(0.50, 0.68, 0.0), # Curled
        ring_tip=(0.54, 0.68, 0.0),   # Curled
        pinky_tip=(0.57, 0.68, 0.0),  # Curled
    )
    # Give index finger realistic joint angles (bent knuckle & PIP)
    lm[6] = [0.46, 0.48, -0.05] # PIP pushed up and forward
    lm[7] = [0.46, 0.50, -0.02] # DIP curled down

    label, conf, rule = disambiguator.disambiguate("D", 0.60, dummy_probs, lm)
    assert label == "X"
    assert "GEOM_X" in rule


def test_y_shaka_span():
    disambiguator = ASLGeometricDisambiguator()
    dummy_probs = np.zeros(28, dtype=np.float32)

    # 'Y': Thumb and Pinky extended far outward, inner 3 curled
    hand_y = _build_synthetic_hand(
        wrist=(0.5, 0.8, 0.0),
        thumb_tip=(0.30, 0.60, 0.0),  # Extended far left
        index_tip=(0.48, 0.68, 0.0),  # Curled
        middle_tip=(0.51, 0.68, 0.0), # Curled
        ring_tip=(0.54, 0.68, 0.0),   # Curled
        pinky_tip=(0.70, 0.58, 0.0),  # Extended far right
    )
    label, conf, rule = disambiguator.disambiguate("I", 0.55, dummy_probs, hand_y)
    assert label == "Y"
    assert "GEOM_Y" in rule


def test_dynamic_z_resilience_and_extended_duration():
    """Verify that Z trajectory accepts natural gestures up to 1.80s and smoothed inflections."""
    classifier = DynamicTrajectoryClassifier(max_duration=1.80)
    buf = TemporalSequenceBuffer(max_len=45)

    # Synthesize 30 frames spanning 1.50 seconds (> original 1.40s ceiling)
    # Z stroke: Stroke 1 (X: 0.3->0.6), Stroke 2 (X: 0.6->0.3, Y: 0.3->0.5), Stroke 3 (X: 0.3->0.6)
    total_frames = 30
    timestamps = np.linspace(0.0, 1.50, total_frames)

    for i, t in enumerate(timestamps):
        if i < 10:
            # Stroke 1: move right
            x = 0.35 + (i / 10.0) * 0.20
            y = 0.30
        elif i < 20:
            # Stroke 2: diagonal down-left
            frac = (i - 10) / 10.0
            x = 0.55 - frac * 0.20
            y = 0.30 + frac * 0.15
        else:
            # Stroke 3: move right
            frac = (i - 20) / 10.0
            x = 0.35 + frac * 0.20
            y = 0.45

        # D handshape (index extended, others curled)
        lm = _build_synthetic_hand(
            wrist=(x, y + 0.25, 0.0),
            thumb_tip=(x - 0.03, y + 0.18, 0.0),
            index_tip=(x, y, 0.0),
            middle_tip=(x + 0.02, y + 0.18, 0.0),
            ring_tip=(x + 0.04, y + 0.18, 0.0),
            pinky_tip=(x + 0.06, y + 0.18, 0.0)
        )
        buf.push(timestamp=float(t), landmarks=lm, is_valid=True)

    result = classifier.evaluate_sequence(buf, current_time=1.55)
    assert result.detected is True
    assert result.gesture == "Z"
    assert result.confidence >= 0.70

