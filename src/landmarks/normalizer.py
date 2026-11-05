"""
Landmark Normalization Module for ASL Recognition.

Performs translation, scale, and handedness normalization on 21 MediaPipe hand landmarks.
Canonicalizes all hands to a right-hand representation.
Outputs a flattened 63-dimensional feature vector.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

class LandmarkNormalizer:
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Small value to prevent division by zero during scale normalization.
        """
        self.eps = eps

    def normalize(self, landmarks: List[Dict[str, float]], handedness: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Normalizes a list of 21 MediaPipe landmarks.

        Args:
            landmarks: List of 21 dictionaries with 'x', 'y', 'z' keys.
            handedness: String, 'Right' or 'Left'.

        Returns:
            Tuple of (is_valid: bool, features: np.ndarray or None, reason: str)
            If valid, features is a (63,) float32 numpy array.
        """
        if not landmarks or len(landmarks) != 21:
            return False, None, f"Expected 21 landmarks, got {len(landmarks) if landmarks else 0}"

        # Convert to numpy array of shape (21, 3)
        try:
            arr = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks], dtype=np.float32)
        except (KeyError, TypeError):
            return False, None, "Invalid landmark format. Expected dicts with 'x', 'y', 'z'."

        # 1. Check for non-finite values
        if not np.isfinite(arr).all():
            return False, None, "Landmarks contain non-finite values (NaN or Inf)."

        # 2. Translation normalization: Make wrist (landmark 0) the origin
        wrist = arr[0].copy()
        arr -= wrist

        # 3. Scale normalization: Scale relative to wrist-to-middle-MCP distance
        # wrist is index 0, middle MCP is index 9
        middle_mcp = arr[9]
        # Distance computed in 3D space
        scale = np.linalg.norm(middle_mcp)

        if scale < self.eps:
            return False, None, f"Zero or near-zero scale (wrist-to-middle-MCP distance: {scale:.8f})"

        arr /= scale

        # 4. Handedness canonicalization:
        # MediaPipe 'Left' / 'Right' refers to the physical hand.
        # We canonicalize everything to 'Right' hand space.
        # Mirroring along the X axis for Left hands.
        handedness = handedness.strip().title()
        if handedness == 'Left':
            arr[:, 0] = -arr[:, 0]
        elif handedness != 'Right':
            return False, None, f"Unknown handedness: {handedness}"

        # 5. Flatten to 63 features
        features = arr.flatten()

        return True, features, "Success"
