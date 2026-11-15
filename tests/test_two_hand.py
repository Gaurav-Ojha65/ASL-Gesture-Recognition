"""
Tests for Two-Hand ASL Recognition Feasibility and Prototype (Phase 5).
"""

import pytest
import numpy as np

from src.landmarks.two_hand import (
    HandLandmarkObservation,
    TwoHandObservation,
    TwoHandNormalizer,
    sort_mediapipe_hands,
    TwoHandGesturePrototype
)


def _generate_synthetic_hand_landmarks(wrist_x: float = 0.5, wrist_y: float = 0.5, palm_size: float = 0.1):
    """Generates synthetic 21-landmark array centered at (wrist_x, wrist_y)."""
    landmarks = []
    # Index 0: wrist
    landmarks.append({'x': wrist_x, 'y': wrist_y, 'z': 0.0})
    # Thumb 1..4
    for i in range(1, 5):
        landmarks.append({'x': wrist_x - 0.02 * i, 'y': wrist_y - 0.02 * i, 'z': 0.0})
    # Index 5..8
    for i in range(1, 5):
        landmarks.append({'x': wrist_x - 0.01 * i, 'y': wrist_y - 0.03 * i, 'z': 0.0})
    # Middle 9..12 (Middle MCP is index 9)
    landmarks.append({'x': wrist_x, 'y': wrist_y - palm_size, 'z': 0.0})  # MCP
    for i in range(2, 5):
        landmarks.append({'x': wrist_x, 'y': wrist_y - palm_size - 0.02 * (i - 1), 'z': 0.0})
    # Ring 13..16
    for i in range(1, 5):
        landmarks.append({'x': wrist_x + 0.01 * i, 'y': wrist_y - 0.03 * i, 'z': 0.0})
    # Pinky 17..20
    for i in range(1, 5):
        landmarks.append({'x': wrist_x + 0.02 * i, 'y': wrist_y - 0.02 * i, 'z': 0.0})

    assert len(landmarks) == 21
    return landmarks


class TestTwoHandExtractionAndSorting:
    """Tests for sorting raw MediaPipe landmarks into Left and Right."""

    def test_sort_both_hands_detected(self):
        left_lms = _generate_synthetic_hand_landmarks(0.3, 0.5)
        right_lms = _generate_synthetic_hand_landmarks(0.7, 0.5)

        raw_landmarks = [right_lms, left_lms]
        raw_handedness = [{'category_name': 'Right', 'score': 0.98}, {'category_name': 'Left', 'score': 0.95}]

        obs = sort_mediapipe_hands(raw_landmarks, raw_handedness)

        assert obs.has_both_hands is True
        assert obs.left_hand is not None
        assert obs.left_hand.handedness == "Left"
        assert obs.right_hand is not None
        assert obs.right_hand.handedness == "Right"

    def test_sort_single_hand_fallback_left(self):
        left_lms = _generate_synthetic_hand_landmarks(0.3, 0.5)
        raw_landmarks = [left_lms]
        raw_handedness = [{'category_name': 'Left', 'score': 0.95}]

        obs = sort_mediapipe_hands(raw_landmarks, raw_handedness)

        assert obs.has_both_hands is False
        assert obs.is_left_only is True
        assert obs.right_hand is None

    def test_sort_single_hand_fallback_right(self):
        right_lms = _generate_synthetic_hand_landmarks(0.7, 0.5)
        raw_landmarks = [right_lms]
        raw_handedness = [{'category_name': 'Right', 'score': 0.98}]

        obs = sort_mediapipe_hands(raw_landmarks, raw_handedness)

        assert obs.has_both_hands is False
        assert obs.is_right_only is True
        assert obs.left_hand is None

    def test_sort_empty_observation(self):
        obs = sort_mediapipe_hands([], [])
        assert obs.has_any_hand is False
        assert obs.left_hand is None
        assert obs.right_hand is None

    def test_disambiguation_when_both_labeled_right(self):
        # Tracking edge case where MediaPipe mistakenly outputs two 'Right' labels
        h1_lms = _generate_synthetic_hand_landmarks(0.2, 0.5)
        h2_lms = _generate_synthetic_hand_landmarks(0.8, 0.5)
        raw_landmarks = [h1_lms, h2_lms]
        raw_handedness = [{'category_name': 'Right'}, {'category_name': 'Right'}]

        obs = sort_mediapipe_hands(raw_landmarks, raw_handedness)
        # Should disambiguate by X-coordinate: left x (0.2) becomes left_hand
        assert obs.left_hand is not None
        assert obs.right_hand is not None
        assert obs.left_hand.landmarks[0]['x'] == 0.2
        assert obs.right_hand.landmarks[0]['x'] == 0.8


class TestTwoHandFeatureRepresentation:
    """Tests for coordinate normalization, validity masks, and dimension consistency."""

    def test_feature_vector_dimension_133(self):
        normalizer = TwoHandNormalizer()
        left_lms = _generate_synthetic_hand_landmarks(0.3, 0.5, palm_size=0.1)
        right_lms = _generate_synthetic_hand_landmarks(0.7, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", left_lms),
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is True
        assert feat.shape == (133,)
        assert info["both_valid"] is True
        assert feat[131] == 1.0  # left valid
        assert feat[132] == 1.0  # right valid

    def test_inter_hand_distance_calculation(self):
        normalizer = TwoHandNormalizer()
        # Palm scale is 0.1 for both hands.
        # Left wrist at (0.3, 0.5, 0.0), Right wrist at (0.5, 0.5, 0.0) -> raw dx = 0.2.
        # Scale = 0.1 -> normalized dx = 0.2 / 0.1 = 2.0.
        left_lms = _generate_synthetic_hand_landmarks(0.3, 0.5, palm_size=0.1)
        right_lms = _generate_synthetic_hand_landmarks(0.5, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", left_lms),
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is True
        # Spatial features start at index 126
        # dx, dy, dz, dist, scale_ratio
        dx = feat[126]
        dy = feat[127]
        dist = feat[129]
        scale_ratio = feat[130]

        assert pytest.approx(dx, rel=1e-3) == 2.0
        assert pytest.approx(dy, abs=1e-3) == 0.0
        assert pytest.approx(dist, rel=1e-3) == 2.0
        assert pytest.approx(scale_ratio, rel=1e-3) == 1.0

    def test_single_hand_validity_mask_left_only(self):
        normalizer = TwoHandNormalizer()
        left_lms = _generate_synthetic_hand_landmarks(0.3, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", left_lms),
            right_hand=None
        )

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is True
        assert feat.shape == (133,)
        assert info["left_valid"] is True
        assert info["right_valid"] is False
        assert info["both_valid"] is False

        # Left features populated (first 63)
        assert not np.all(feat[:63] == 0)
        # Right features zeroed (indices 63:126)
        assert np.all(feat[63:126] == 0)
        # Spatial features zeroed (indices 126:131)
        assert np.all(feat[126:131] == 0)
        # Mask check
        assert feat[131] == 1.0  # left valid
        assert feat[132] == 0.0  # right invalid

    def test_single_hand_validity_mask_right_only(self):
        normalizer = TwoHandNormalizer()
        right_lms = _generate_synthetic_hand_landmarks(0.7, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=None,
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is True
        assert feat.shape == (133,)
        assert feat[131] == 0.0  # left invalid
        assert feat[132] == 1.0  # right valid
        assert np.all(feat[:63] == 0)
        assert not np.all(feat[63:126] == 0)

    def test_missing_both_hands(self):
        normalizer = TwoHandNormalizer()
        obs = TwoHandObservation(left_hand=None, right_hand=None)

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is False
        assert np.all(feat == 0)
        assert feat[131] == 0.0
        assert feat[132] == 0.0

    def test_non_finite_landmarks_handled_gracefully(self):
        normalizer = TwoHandNormalizer()
        corrupt_lms = _generate_synthetic_hand_landmarks(0.5, 0.5)
        corrupt_lms[3]['x'] = float('nan')

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", corrupt_lms),
            right_hand=None
        )

        success, feat, info = normalizer.extract_two_hand_features(obs)
        assert success is False
        assert info["left_valid"] is False
        assert np.all(feat == 0)


class TestTwoHandGesturePrototype:
    """Tests for the prototype evaluation interface."""

    def test_prototype_evaluates_separated_hands(self):
        prototype = TwoHandGesturePrototype()
        left_lms = _generate_synthetic_hand_landmarks(0.2, 0.5, palm_size=0.1)
        right_lms = _generate_synthetic_hand_landmarks(0.8, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", left_lms),
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        res = prototype.evaluate_two_hand_observation(obs)
        assert res["recognized"] is True
        assert res["is_two_hand"] is True
        assert res["label"] == "TWO_HAND_SEPARATED"

    def test_prototype_evaluates_interacting_hands(self):
        prototype = TwoHandGesturePrototype()
        # Wrists very close (distance = 0.1 / 0.1 = 1.0 < 2.5)
        left_lms = _generate_synthetic_hand_landmarks(0.45, 0.5, palm_size=0.1)
        right_lms = _generate_synthetic_hand_landmarks(0.55, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=HandLandmarkObservation("Left", left_lms),
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        res = prototype.evaluate_two_hand_observation(obs)
        assert res["recognized"] is True
        assert res["is_two_hand"] is True
        assert res["label"] == "TWO_HAND_PROXIMITY_INTERACTION"

    def test_prototype_single_hand_fallback(self):
        prototype = TwoHandGesturePrototype()
        right_lms = _generate_synthetic_hand_landmarks(0.5, 0.5, palm_size=0.1)

        obs = TwoHandObservation(
            left_hand=None,
            right_hand=HandLandmarkObservation("Right", right_lms)
        )

        res = prototype.evaluate_two_hand_observation(obs)
        assert res["recognized"] is False
        assert res["label"] == "SINGLE_HAND_ONLY"
        assert res["active_hand"] == "Right"

