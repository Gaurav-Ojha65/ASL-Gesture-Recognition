"""
Shared Model Service for FastAPI Backend.
Maintains shared, thread-safe, stateless recognition resources:
1. Frozen PyTorch 28-Class ASLLandmarkMLP (stateless forward pass).
2. MediaPipe HandLandmarker in IMAGE running mode with serialized thread lock.
"""

import os
import threading
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.realtime.inference import ASLInferenceEngine
from src.landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX


class SharedModelService:
    """
    Singleton service managing shared immutable perception and inference assets.
    Does NOT contain mutable per-session state or temporal smoothing histories.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/asl_landmark_mlp_best.pt",
        task_model_path: str = "hand_landmarker.task",
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.task_model_path = task_model_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.engine: Optional[ASLInferenceEngine] = None
        self.detector: Optional[vision.HandLandmarker] = None
        self._detector_lock = threading.Lock()
        self._is_loaded: bool = False

    def load(self) -> None:
        """Loads and initializes model checkpoint and MediaPipe detector."""
        if self._is_loaded:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"MLP checkpoint not found at: {self.checkpoint_path}")
        if not os.path.exists(self.task_model_path):
            raise FileNotFoundError(f"MediaPipe task model not found at: {self.task_model_path}")

        # 1. Load PyTorch inference engine in stateless mode (smoothing disabled at model level)
        self.engine = ASLInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            enable_smoothing=False,
            smoothing_window=1,
        )

        # 2. Load MediaPipe HandLandmarker in stateless RunningMode.IMAGE
        base_options = python.BaseOptions(model_asset_path=self.task_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._is_loaded = True

    def is_ready(self) -> bool:
        """Returns True if both model and detector are initialized."""
        return self._is_loaded and self.engine is not None and self.detector is not None

    def detect_hand(self, rgb_image: np.ndarray) -> Any:
        """
        Executes MediaPipe hand landmark detection on an RGB image array (H, W, 3).
        Serialized with a thread lock to prevent concurrent TFLite delegate race conditions.

        Args:
            rgb_image: uint8 numpy array in RGB format.

        Returns:
            MediaPipe HandLandmarkerResult.
        """
        if not self.is_ready():
            raise RuntimeError("SharedModelService is not loaded. Call load() first.")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        with self._detector_lock:
            return self.detector.detect(mp_image)

    def predict_stateless(
        self,
        features_63: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, str, float]:
        """
        Executes pure stateless inference on a 63-D normalized landmark vector.
        Thread-safe, applies no mutation to model or probability histories.

        Args:
            features_63: 63-element normalized float32 array.

        Returns:
            Tuple of:
            - raw_probabilities: (28,) float32 array
            - logits: (28,) float32 array
            - top1_index: int
            - top1_label: str
            - top1_confidence: float
        """
        if not self.is_ready():
            raise RuntimeError("SharedModelService is not loaded. Call load() first.")

        return self.engine.predict_stateless(features_63)

    def close(self) -> None:
        """Releases underlying MediaPipe detector resources."""
        with self._detector_lock:
            if self.detector is not None:
                try:
                    self.detector.close()
                except Exception:
                    pass
                self.detector = None
        self.engine = None
        self._is_loaded = False

