"""
Geometric Disambiguation Module for Fine-Grained ASL Gesture Recognition.

Provides mathematically rigorous, scale- and orientation-invariant anatomical feature
extractors on 21 MediaPipe hand landmarks to resolve classical ambiguity clusters:
1. Fist Cluster: M, N, T, S, A, E (knuckle projection ratio, thumb placement, curl metrics)
2. Open Two-Finger Cluster: V vs. U vs. K (inter-fingertip spread ratio, knuckle angle)
3. Circle / Open Fingers Cluster: F vs. B vs. E (index-thumb contact distance, upright finger count)
4. Hooked Finger Cluster: X vs. D vs. fist (index PIP curl ratio, tip extension ratio)
5. Horizontal / Inverted Cluster: G vs. H, P vs. Q vs. K (relative palm axis, finger extension)
6. Shaka Span Cluster: Y vs. I vs. J (thumb-pinky span vs. middle-ring-index curl)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class HandGeometry:
    """
    Computes invariant anatomical measurements from 21 MediaPipe hand landmarks.
    Landmarks can be in raw image coordinates or normalized coordinates.
    """

    def __init__(self, landmarks: np.ndarray):
        """
        Args:
            landmarks: Array of shape (21, 3) with (x, y, z) coordinates.
        """
        assert landmarks.shape == (21, 3), f"Expected (21, 3), got {landmarks.shape}"
        self.lm = landmarks.astype(np.float32)

        # Key Landmarks
        self.wrist = self.lm[0]
        self.thumb_cmc = self.lm[1]
        self.thumb_mcp = self.lm[2]
        self.thumb_ip = self.lm[3]
        self.thumb_tip = self.lm[4]

        self.index_mcp = self.lm[5]
        self.index_pip = self.lm[6]
        self.index_dip = self.lm[7]
        self.index_tip = self.lm[8]

        self.middle_mcp = self.lm[9]
        self.middle_pip = self.lm[10]
        self.middle_dip = self.lm[11]
        self.middle_tip = self.lm[12]

        self.ring_mcp = self.lm[13]
        self.ring_pip = self.lm[14]
        self.ring_dip = self.lm[15]
        self.ring_tip = self.lm[16]

        self.pinky_mcp = self.lm[17]
        self.pinky_pip = self.lm[18]
        self.pinky_dip = self.lm[19]
        self.pinky_tip = self.lm[20]

        # Characteristic Hand Scale (wrist to middle MCP)
        self.hand_scale = max(float(np.linalg.norm(self.middle_mcp - self.wrist)), 1e-4)

        # Palm Knuckle Vector: Vector from Index MCP to Pinky MCP across the palm
        self.knuckle_vector = self.pinky_mcp - self.index_mcp
        self.knuckle_span = max(float(np.linalg.norm(self.knuckle_vector)), 1e-4)
        self.knuckle_unit = self.knuckle_vector / self.knuckle_span

        # Palm Long Axis Vector: Vector from Wrist to Middle MCP
        self.palm_long_axis = (self.middle_mcp - self.wrist) / self.hand_scale

    def dist(self, idx1: int, idx2: int) -> float:
        """Normalized Euclidean distance between two landmarks."""
        return float(np.linalg.norm(self.lm[idx1] - self.lm[idx2])) / self.hand_scale

    def finger_extension_ratio(self, tip_idx: int, mcp_idx: int) -> float:
        """Ratio of tip distance from wrist to MCP distance from wrist."""
        dist_tip = np.linalg.norm(self.lm[tip_idx] - self.wrist)
        dist_mcp = np.linalg.norm(self.lm[mcp_idx] - self.wrist)
        return float(dist_tip / max(dist_mcp, 1e-4))

    def is_finger_extended(self, tip_idx: int, mcp_idx: int, thresh: float = 1.25) -> bool:
        """Returns True if fingertip is significantly further from wrist than MCP."""
        return self.finger_extension_ratio(tip_idx, mcp_idx) >= thresh

    def finger_curl_ratio(self, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> float:
        """
        Straightness ratio: straight distance (tip to MCP) divided by
        sum of segment lengths (tip-dip + dip-pip + pip-mcp).
        1.0 = fully straight; ~0.4 = fully curled.
        """
        direct_dist = np.linalg.norm(self.lm[tip_idx] - self.lm[mcp_idx])
        bone_sum = (
            np.linalg.norm(self.lm[tip_idx] - self.lm[dip_idx]) +
            np.linalg.norm(self.lm[dip_idx] - self.lm[pip_idx]) +
            np.linalg.norm(self.lm[pip_idx] - self.lm[mcp_idx])
        )
        return float(direct_dist / max(bone_sum, 1e-4))


class ASLGeometricDisambiguator:
    """
    Applies invariant geometric rules to arbitrate and refine MLP predictions
    for known confusion clusters without altering model weights.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def disambiguate(
        self,
        predicted_label: str,
        confidence: float,
        probabilities: np.ndarray,
        landmarks: np.ndarray,
        handedness: str = "Right"
    ) -> Tuple[str, float, str]:
        """
        Refines predictions for problematic letters.

        Args:
            predicted_label: Model's raw or smoothed top-1 class.
            confidence: Current top-1 confidence score.
            probabilities: Full 28-class probability distribution.
            landmarks: Shape (21, 3) raw or normalized landmark coordinates.
            handedness: 'Right' or 'Left'.

        Returns:
            Tuple of (refined_label: str, refined_confidence: float, rule_applied: str)
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return predicted_label, confidence, "NONE (invalid landmarks)"

        geo = HandGeometry(landmarks)

        # 1. Cluster: V vs. U vs. K
        # If model predicted U, V, or K, or if both U and V have competitive probabilities
        if predicted_label in ("U", "V", "K") or (
            probabilities is not None and len(probabilities) == 28 and (
                probabilities[20] > 0.20 or probabilities[21] > 0.20 or probabilities[10] > 0.20
            )
        ):
            refined = self._disambiguate_u_v_k(geo, predicted_label, confidence)
            if refined:
                return refined

        # 2. Cluster: Fist Cluster (M, N, T, S, A, E)
        if predicted_label in ("M", "N", "T", "S", "A", "E"):
            refined = self._disambiguate_fist_cluster(geo, predicted_label, confidence)
            if refined:
                return refined

        # 3. Cluster: F vs. B vs. E
        if predicted_label in ("F", "B", "E"):
            refined = self._disambiguate_f_b_e(geo, predicted_label, confidence)
            if refined:
                return refined

        # 4. Cluster: X vs. D vs. Fist
        if predicted_label in ("X", "D", "A", "S", "M"):
            refined = self._disambiguate_x(geo, predicted_label, confidence)
            if refined:
                return refined

        # 5. Cluster: G vs. H
        if predicted_label in ("G", "H"):
            refined = self._disambiguate_g_h(geo, predicted_label, confidence)
            if refined:
                return refined

        # 6. Cluster: P vs. Q vs. K
        if predicted_label in ("P", "Q", "K", "G"):
            refined = self._disambiguate_p_q(geo, predicted_label, confidence)
            if refined:
                return refined

        # 7. Cluster: Y vs. I
        if predicted_label in ("Y", "I", "J"):
            refined = self._disambiguate_y(geo, predicted_label, confidence)
            if refined:
                return refined

        return predicted_label, confidence, "UNCHANGED"

    # ── Private Cluster Handlers ──────────────────────────────────────────────

    def _disambiguate_u_v_k(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Differentiates U and V using the normalized spread between index tip and middle tip.
        """
        index_ext = geo.is_finger_extended(8, 5, thresh=1.15)
        middle_ext = geo.is_finger_extended(12, 9, thresh=1.15)
        ring_ext = geo.is_finger_extended(16, 13, thresh=1.15)
        pinky_ext = geo.is_finger_extended(20, 17, thresh=1.15)

        # Both index and middle must be extended, while ring and pinky are curled
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            # Distance between index tip (8) and middle tip (12)
            tip_spread = geo.dist(8, 12)
            # Distance between index MCP (5) and middle MCP (9)
            base_spread = geo.dist(5, 9)

            # Ratio of fingertip spread to knuckle spread
            spread_ratio = tip_spread / max(base_spread, 1e-4)

            # Check K: in K, thumb is upright between index and middle, and middle points slightly forward
            thumb_in_middle = geo.dist(4, 9) < 0.45 or geo.dist(4, 10) < 0.45
            if current_label == "K" and thumb_in_middle:
                return "K", max(confidence, 0.85), "GEOM_K (Thumb between index and middle)"

            # Spread ratio >= 1.6 indicates clear V shape; < 1.4 indicates U touching/parallel
            if spread_ratio >= 1.55 or tip_spread >= 0.36:
                return "V", max(confidence, 0.88), f"GEOM_V (Spread ratio: {spread_ratio:.2f})"
            elif spread_ratio < 1.35 or tip_spread < 0.28:
                return "U", max(confidence, 0.88), f"GEOM_U (Parallel ratio: {spread_ratio:.2f})"

        return None

    def _disambiguate_fist_cluster(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Differentiates M, N, T, S, A, E based on thumb position relative to the knuckle line.
        """
        # Verify fist: all 4 fingers curled
        index_curled = not geo.is_finger_extended(8, 5, thresh=1.20)
        middle_curled = not geo.is_finger_extended(12, 9, thresh=1.20)
        ring_curled = not geo.is_finger_extended(16, 13, thresh=1.20)
        pinky_curled = not geo.is_finger_extended(20, 17, thresh=1.20)

        if not (index_curled and middle_curled and ring_curled and pinky_curled):
            return None

        # Project thumb tip onto knuckle vector (from index MCP 5 to pinky MCP 17)
        thumb_vec = geo.thumb_tip - geo.index_mcp
        proj_scalar = float(np.dot(thumb_vec, geo.knuckle_unit) / geo.knuckle_span)

        # Distance from thumb tip to index MCP, middle MCP, ring MCP
        d_index = geo.dist(4, 5)
        d_middle = geo.dist(4, 9)
        d_ring = geo.dist(4, 13)
        d_pinky = geo.dist(4, 17)

        # Check E: Fingertips tightly curled, thumb tucked below fingertips (close to palm base)
        # Thumb tip is close to middle MCP or ring MCP in height, fingers curled down on top of it
        finger_tips_mcp_dist = (geo.dist(8, 5) + geo.dist(12, 9) + geo.dist(16, 13) + geo.dist(20, 17)) / 4.0
        if current_label == "E" and finger_tips_mcp_dist < 0.40:
            return "E", max(confidence, 0.85), "GEOM_E (All tips curled on thumb)"

        # Check A: Thumb tip is extended vertically along the side of index MCP/PIP
        thumb_tip_upright = geo.thumb_tip[1] < geo.index_mcp[1] or d_index < 0.35
        if current_label == "A" and proj_scalar < 0.15 and thumb_tip_upright:
            return "A", max(confidence, 0.88), f"GEOM_A (Thumb lateral to index: {proj_scalar:.2f})"

        # Check S: Thumb crossed horizontally over the front of the fingers
        # Proj scalar is typically between 0.30 and 0.70 (over middle and ring)
        if current_label == "S":
            if 0.25 <= proj_scalar <= 0.75:
                return "S", max(confidence, 0.85), f"GEOM_S (Thumb crossing knuckles: {proj_scalar:.2f})"

        # Disambiguate M vs N vs T
        # T: Thumb between index and middle (proj_scalar < 0.35)
        # N: Thumb between middle and ring (0.35 <= proj_scalar <= 0.65)
        # M: Thumb between ring and pinky (proj_scalar > 0.65)
        if current_label in ("M", "N", "T"):
            if proj_scalar < 0.35:
                return "T", max(confidence, 0.85), f"GEOM_T (Thumb at index-middle: {proj_scalar:.2f})"
            elif proj_scalar > 0.65:
                return "M", max(confidence, 0.85), f"GEOM_M (Thumb at ring-pinky: {proj_scalar:.2f})"
            else:
                return "N", max(confidence, 0.85), f"GEOM_N (Thumb at middle-ring: {proj_scalar:.2f})"

        return None

    def _disambiguate_f_b_e(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Differentiates F from B and E.
        In F: index and thumb touch (distance < 0.30 scale), other 3 fingers extended upright.
        In B: all 4 fingers extended upright, thumb across palm.
        In E: all fingers curled down.
        """
        index_thumb_dist = geo.dist(8, 4)
        middle_ext = geo.is_finger_extended(12, 9, thresh=1.20)
        ring_ext = geo.is_finger_extended(16, 13, thresh=1.20)
        pinky_ext = geo.is_finger_extended(20, 17, thresh=1.20)
        index_ext = geo.is_finger_extended(8, 5, thresh=1.20)

        # F: 3 fingers upright, index and thumb touching
        if middle_ext and ring_ext and pinky_ext:
            if index_thumb_dist < 0.35 or not index_ext:
                return "F", max(confidence, 0.90), f"GEOM_F (Index-thumb loop dist: {index_thumb_dist:.2f})"
            elif index_ext and index_thumb_dist >= 0.40:
                return "B", max(confidence, 0.88), f"GEOM_B (Index upright, dist: {index_thumb_dist:.2f})"

        return None

    def _disambiguate_x(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Detects X (hooked index finger).
        Index straightness ratio: straight distance (tip to MCP) / sum of bones.
        In straight finger (D): ratio > 0.85.
        In hooked finger (X): ratio between 0.50 and 0.78.
        In fist (A/S/M): ratio < 0.45.
        """
        middle_curled = not geo.is_finger_extended(12, 9, thresh=1.15)
        ring_curled = not geo.is_finger_extended(16, 13, thresh=1.15)
        pinky_curled = not geo.is_finger_extended(20, 17, thresh=1.15)

        if middle_curled and ring_curled and pinky_curled:
            index_curl = geo.finger_curl_ratio(8, 7, 6, 5)
            # Distance from index tip to wrist
            index_tip_dist = geo.dist(8, 0)
            index_mcp_dist = geo.dist(5, 0)

            # X has hooked index: extended past MCP or forward, but curled significantly
            if (0.35 <= index_curl <= 0.80) and (index_tip_dist > index_mcp_dist * 0.85):
                return "X", max(confidence, 0.88), f"GEOM_X (Index hooked curl ratio: {index_curl:.2f})"
            elif index_curl > 0.84 and index_tip_dist > index_mcp_dist * 1.3:
                return "D", max(confidence, 0.90), f"GEOM_D (Index straight: {index_curl:.2f})"

        return None

    def _disambiguate_g_h(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Differentiates G (one finger horizontal) from H (two fingers horizontal).
        """
        index_ext = geo.is_finger_extended(8, 5, thresh=1.20)
        middle_ext = geo.is_finger_extended(12, 9, thresh=1.20)
        ring_curled = not geo.is_finger_extended(16, 13, thresh=1.15)
        pinky_curled = not geo.is_finger_extended(20, 17, thresh=1.15)

        if index_ext and ring_curled and pinky_curled:
            if middle_ext:
                return "H", max(confidence, 0.88), "GEOM_H (Index and Middle extended)"
            else:
                return "G", max(confidence, 0.88), "GEOM_G (Index extended, Middle curled)"

        return None

    def _disambiguate_p_q(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Differentiates P (downward K) and Q (downward G).
        """
        # Inverted hand orientation check: middle MCP Y is greater than wrist Y (pointing downward in camera space)
        is_pointing_down = geo.middle_mcp[1] > geo.wrist[1]

        if is_pointing_down:
            index_ext = geo.is_finger_extended(8, 5, thresh=1.15)
            middle_ext = geo.is_finger_extended(12, 9, thresh=1.15)
            ring_curled = not geo.is_finger_extended(16, 13, thresh=1.15)
            pinky_curled = not geo.is_finger_extended(20, 17, thresh=1.15)

            if ring_curled and pinky_curled and index_ext:
                if middle_ext or current_label in ("P", "K"):
                    return "P", max(confidence, 0.88), "GEOM_P (Downward K handshape)"
                elif not middle_ext or current_label in ("Q", "G"):
                    return "Q", max(confidence, 0.88), "GEOM_Q (Downward G handshape)"

        return None

    def _disambiguate_y(
        self, geo: HandGeometry, current_label: str, confidence: float
    ) -> Optional[Tuple[str, float, str]]:
        """
        Validates Y: Thumb and Pinky extended, Index, Middle, Ring curled.
        """
        thumb_ext = geo.is_finger_extended(4, 2, thresh=1.15)
        pinky_ext = geo.is_finger_extended(20, 17, thresh=1.15)
        index_curled = not geo.is_finger_extended(8, 5, thresh=1.15)
        middle_curled = not geo.is_finger_extended(12, 9, thresh=1.15)
        ring_curled = not geo.is_finger_extended(16, 13, thresh=1.15)

        if thumb_ext and pinky_ext and index_curled and middle_curled and ring_curled:
            span = geo.dist(4, 20)
            if span >= 0.85:
                return "Y", max(confidence, 0.90), f"GEOM_Y (Shaka thumb-pinky span: {span:.2f})"

        return None
