"""
Two-Hand Landmark Extraction and Feature Representation Module (Phase 5).

Provides:
- MediaPipe two-hand landmark extraction with consistent handedness sorting (Left vs Right).
- Fallback handling for single-hand or missing-hand observations.
- Coordinate normalization preserving individual hand geometries and relative inter-hand spatial geometry.
- 133-dimensional feature representation:
    * Left Hand normalized 3D landmarks: 63 dims
    * Right Hand normalized 3D landmarks: 63 dims
    * Inter-hand spatial relationship features: 5 dims
        - Euclidean distance between wrists (normalized)
        - Vector dx, dy, dz from Left wrist to Right wrist (normalized)
        - Inter-hand orientation / palm normal angle (or pitch/yaw relation)
    * Hand validity mask: 2 dims (is_left_present, is_right_present)
    * Total vector dimension = 63 + 63 + 5 + 2 = 133 dims (or standard 126 dims for pure landmarks).
- TwoHandObservation container and modular prototype classification interface.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import math


@dataclass(frozen=True)
class HandLandmarkObservation:
    """Individual hand landmark data."""
    handedness: str  # 'Left' or 'Right'
    landmarks: List[Dict[str, float]]  # 21 MediaPipe dicts with x, y, z
    confidence: float = 1.0


@dataclass(frozen=True)
class TwoHandObservation:
    """Multi-hand observation container for a single video frame."""
    left_hand: Optional[HandLandmarkObservation] = None
    right_hand: Optional[HandLandmarkObservation] = None
    timestamp: float = 0.0

    @property
    def has_both_hands(self) -> bool:
        return self.left_hand is not None and self.right_hand is not None

    @property
    def has_any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None

    @property
    def is_left_only(self) -> bool:
        return self.left_hand is not None and self.right_hand is None

    @property
    def is_right_only(self) -> bool:
        return self.left_hand is None and self.right_hand is not None


class TwoHandNormalizer:
    """
    Normalizes multi-hand landmarks preserving:
    1. Independent intra-hand structure (wrist origin, middle-MCP scale).
    2. Global inter-hand spatial relationships (wrist-to-wrist vector, relative distance).
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _normalize_single_hand(
        self,
        landmarks: List[Dict[str, float]],
        canonicalize_to_right: bool = False,
        handedness: str = "Right"
    ) -> Tuple[bool, Optional[np.ndarray], float, np.ndarray, str]:
        """
        Normalizes a single hand's 21 landmarks.
        Returns:
            (is_valid, normalized_arr_21x3, scale, raw_wrist_coord, reason)
        """
        if not landmarks or len(landmarks) != 21:
            return False, None, 0.0, np.zeros(3), f"Expected 21 landmarks, got {len(landmarks) if landmarks else 0}"

        try:
            arr = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks], dtype=np.float32)
        except (KeyError, TypeError):
            return False, None, 0.0, np.zeros(3), "Invalid landmark format. Expected dicts with 'x', 'y', 'z'."

        if not np.isfinite(arr).all():
            return False, None, 0.0, np.zeros(3), "Landmarks contain non-finite values (NaN or Inf)."

        raw_wrist = arr[0].copy()
        arr -= raw_wrist

        middle_mcp = arr[9]
        scale = float(np.linalg.norm(middle_mcp))

        if scale < self.eps:
            return False, None, 0.0, raw_wrist, f"Near-zero scale: {scale:.8f}"

        arr /= scale

        if canonicalize_to_right:
            if handedness.strip().title() == "Left":
                arr[:, 0] = -arr[:, 0]

        return True, arr, scale, raw_wrist, "Success"

    def extract_two_hand_features(
        self,
        observation: TwoHandObservation,
        canonicalize_individual_hands: bool = False
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Extracts 133-dimensional feature vector.
        Components:
        - [0:63]   : Left hand landmarks (21x3 flattened), or zeros if absent.
        - [63:126] : Right hand landmarks (21x3 flattened), or zeros if absent.
        - [126:131]: Inter-hand spatial features:
                     [dx, dy, dz, distance, relative_scale_ratio]
        - [131:133]: Hand presence validity mask:
                     [is_left_present (0.0 or 1.0), is_right_present (0.0 or 1.0)]

        Returns:
            (success: bool, feature_vector: np.ndarray (shape (133,)), info: dict)
        """
        left_feat = np.zeros(63, dtype=np.float32)
        right_feat = np.zeros(63, dtype=np.float32)
        spatial_feat = np.zeros(5, dtype=np.float32)
        mask = np.zeros(2, dtype=np.float32)  # [left_valid, right_valid]

        left_valid = False
        right_valid = False
        left_wrist = np.zeros(3, dtype=np.float32)
        right_wrist = np.zeros(3, dtype=np.float32)
        left_scale = 1.0
        right_scale = 1.0

        # Process Left Hand
        if observation.left_hand is not None:
            ok_l, arr_l, s_l, w_l, _ = self._normalize_single_hand(
                observation.left_hand.landmarks,
                canonicalize_to_right=canonicalize_individual_hands,
                handedness="Left"
            )
            if ok_l:
                left_feat = arr_l.flatten()
                left_valid = True
                mask[0] = 1.0
                left_wrist = w_l
                left_scale = s_l

        # Process Right Hand
        if observation.right_hand is not None:
            ok_r, arr_r, s_r, w_r, _ = self._normalize_single_hand(
                observation.right_hand.landmarks,
                canonicalize_to_right=canonicalize_individual_hands,
                handedness="Right"
            )
            if ok_r:
                right_feat = arr_r.flatten()
                right_valid = True
                mask[1] = 1.0
                right_wrist = w_r
                right_scale = s_r

        # Process Inter-Hand Relationships if both present
        if left_valid and right_valid:
            # Reference scale = mean of both hand palm scales
            ref_scale = (left_scale + right_scale) / 2.0
            if ref_scale > self.eps:
                wrist_diff = (right_wrist - left_wrist) / ref_scale
                dist = float(np.linalg.norm(wrist_diff))
                scale_ratio = float(right_scale / max(left_scale, self.eps))
                spatial_feat = np.array([
                    wrist_diff[0],
                    wrist_diff[1],
                    wrist_diff[2],
                    dist,
                    scale_ratio
                ], dtype=np.float32)

        # Concatenate 63 + 63 + 5 + 2 = 133
        feature_vector = np.concatenate([left_feat, right_feat, spatial_feat, mask])

        overall_success = left_valid or right_valid

        info = {
            "left_valid": left_valid,
            "right_valid": right_valid,
            "both_valid": left_valid and right_valid,
            "left_scale": left_scale if left_valid else None,
            "right_scale": right_scale if right_valid else None,
            "wrist_distance": float(spatial_feat[3]) if (left_valid and right_valid) else None,
            "feature_dim": len(feature_vector)
        }

        return overall_success, feature_vector, info


def sort_mediapipe_hands(
    raw_landmarks_list: List[Any],
    raw_handedness_list: List[Any]
) -> TwoHandObservation:
    """
    Parses MediaPipe detection results (multi_hand_landmarks and handedness)
    and sorts them strictly into Left and Right slots.

    MediaPipe convention:
    - In mirror mode or standard camera mode, MediaPipe supplies category_name / label:
      'Left' or 'Right'.
    - If two hands are detected with identical handedness labels (e.g. edge tracking error),
      falls back to spatial X-coordinate sorting (Left hand has smaller or larger X depending on camera).
    """
    if not raw_landmarks_list:
        return TwoHandObservation()

    detected_hands: List[HandLandmarkObservation] = []

    for lms, handedness_meta in zip(raw_landmarks_list, raw_handedness_list):
        # Extract label
        label = "Right"
        conf = 1.0
        if hasattr(handedness_meta, 'category_name'):
            label = handedness_meta.category_name
            conf = getattr(handedness_meta, 'score', 1.0)
        elif isinstance(handedness_meta, list) and len(handedness_meta) > 0:
            first = handedness_meta[0]
            label = getattr(first, 'category_name', getattr(first, 'label', 'Right'))
            conf = getattr(first, 'score', 1.0)
        elif isinstance(handedness_meta, dict):
            label = handedness_meta.get("label", handedness_meta.get("category_name", "Right"))
            conf = handedness_meta.get("score", 1.0)

        # Parse landmark dicts
        parsed_lms: List[Dict[str, float]] = []
        for p in lms:
            if hasattr(p, 'x'):
                parsed_lms.append({'x': float(p.x), 'y': float(p.y), 'z': float(p.z)})
            elif isinstance(p, dict):
                parsed_lms.append({'x': float(p['x']), 'y': float(p['y']), 'z': float(p.get('z', 0.0))})

        detected_hands.append(HandLandmarkObservation(
            handedness=label.strip().title(),
            landmarks=parsed_lms,
            confidence=float(conf)
        ))

    left_hand: Optional[HandLandmarkObservation] = None
    right_hand: Optional[HandLandmarkObservation] = None

    if len(detected_hands) == 1:
        h = detected_hands[0]
        if h.handedness == "Left":
            left_hand = h
        else:
            right_hand = h
    elif len(detected_hands) >= 2:
        h1, h2 = detected_hands[0], detected_hands[1]
        if h1.handedness != h2.handedness:
            if h1.handedness == "Left":
                left_hand, right_hand = h1, h2
            else:
                left_hand, right_hand = h2, h1
        else:
            # Disambiguate by spatial X position of wrist
            w1_x = h1.landmarks[0]['x'] if h1.landmarks else 0.5
            w2_x = h2.landmarks[0]['x'] if h2.landmarks else 0.5
            if w1_x < w2_x:
                left_hand, right_hand = h1, h2
            else:
                left_hand, right_hand = h2, h1

    return TwoHandObservation(left_hand=left_hand, right_hand=right_hand)


class TwoHandGesturePrototype:
    """
    Experimental prototype interface for two-handed ASL recognition.
    Does NOT replace or alter ASLLandmarkMLP.
    Serves as an architectural foundation for future multi-hand trained models.
    """

    SUPPORTED_TWO_HAND_PROTOTYPE_GESTURES = [
        "BOOK", "HELP", "MORE", "PLAY", "THANK_YOU", "TWO_HAND_IDLE"
    ]

    def __init__(self, normalizer: Optional[TwoHandNormalizer] = None):
        self.normalizer = normalizer or TwoHandNormalizer()

    def evaluate_two_hand_observation(
        self,
        observation: TwoHandObservation
    ) -> Dict[str, Any]:
        """
        Evaluates a two-hand observation.
        Returns diagnostic features, validity masks, and explainable prototype heuristics.
        """
        success, features, info = self.normalizer.extract_two_hand_features(observation)

        if not success:
            return {
                "recognized": False,
                "label": "NO_HANDS",
                "confidence": 0.0,
                "is_two_hand": False,
                "info": info
            }

        # Check if single-hand only
        if not info["both_valid"]:
            return {
                "recognized": False,
                "label": "SINGLE_HAND_ONLY",
                "confidence": 0.0,
                "is_two_hand": False,
                "active_hand": "Left" if info["left_valid"] else "Right",
                "info": info
            }

        # Rule-based architectural prototype for two hands in close interaction:
        # e.g., BOOK (palms open side-by-side) or HELP (fist resting on open palm)
        dist = info.get("wrist_distance")
        if dist is not None and dist < 2.5:
            # Hands are in interacting proximity
            candidate_label = "TWO_HAND_PROXIMITY_INTERACTION"
            confidence = 0.85
        else:
            candidate_label = "TWO_HAND_SEPARATED"
            confidence = 0.60

        return {
            "recognized": True,
            "label": candidate_label,
            "confidence": confidence,
            "is_two_hand": True,
            "feature_dim": len(features),
            "info": info
        }

