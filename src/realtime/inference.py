"""
Inference Module for Real-Time ASL Gesture Recognition Interface.

Wraps the trained ASLLandmarkMLP model with:
- Strict architecture and parameter validation (18,524 parameters, 63->28)
- Checkpoint metadata validation against src/landmarks/labels.py
- Automatic device selection (CUDA when available, CPU fallback)
- Safe eval-mode inference with torch.no_grad()
- Raw logit preservation and softmax probability computation
- Input shape and finite-value verification
"""

import os
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import torch

from src.model.mlp import ASLLandmarkMLP
from src.landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, get_semantic_label

@dataclass(frozen=True)
class InferenceResult:
    """
    Structured prediction output from the ASL inference engine.
    """
    label: str
    class_index: int
    confidence: float
    probabilities: np.ndarray
    logits: np.ndarray


class ASLInferenceEngine:
    """
    Inference wrapper for the 28-class ASL Landmark MLP.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/asl_landmark_mlp_best.pt",
        device: Optional[torch.device] = None,
        smoothing_window: int = 3,
        enable_smoothing: bool = True
    ):
        """
        Args:
            checkpoint_path: Path to the trained PyTorch checkpoint.
            device: Explicit torch device (if None, selects CUDA if available, else CPU).
            smoothing_window: Temporal moving average window size (default: 3 frames).
            enable_smoothing: Whether to apply temporal probability smoothing (default: True).

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            ValueError: If checkpoint metadata, parameter count, or architecture mismatches.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smoothing_window = max(1, int(smoothing_window))
        self.enable_smoothing = enable_smoothing
        self._prob_history: deque = deque(maxlen=self.smoothing_window)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 1. Validate checkpoint metadata
        input_dim = checkpoint.get("input_dim", 63)
        if input_dim != 63:
            raise ValueError(f"Expected input_dim 63, got {input_dim}")

        num_classes = checkpoint.get("num_classes", 28)
        if num_classes != 28:
            raise ValueError(f"Expected num_classes 28, got {num_classes}")

        ckpt_class_to_idx = checkpoint.get("class_to_index")
        if ckpt_class_to_idx is not None and ckpt_class_to_idx != CLASS_TO_IDX:
            raise ValueError("Checkpoint class_to_index mapping conflicts with src/landmarks/labels.py")

        # 2. Instantiate and validate architecture
        self.model = ASLLandmarkMLP()
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.param_count != 18524:
            raise ValueError(f"Expected 18,524 trainable parameters, got {self.param_count}")

        self.model.to(self.device)
        self.model.eval()

    def reset_smoothing(self) -> None:
        """
        Clears the temporal probability history.
        Must be called upon hand tracking loss or session reset to prevent
        blending predictions across different hand instances.
        """
        self._prob_history.clear()

    def predict_logits(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs the underlying model to return raw, unnormalized logits.

        Args:
            features_tensor: Tensor of shape (batch_size, 63).

        Returns:
            torch.Tensor of shape (batch_size, 28) containing raw logits.
        """
        with torch.inference_mode():
            return self.model(features_tensor)

    def predict_stateless(
        self,
        features: Union[np.ndarray, list, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, int, str, float]:
        """
        Executes pure stateless forward inference on a single 63-D feature vector.
        Thread-safe, applies no internal mutation or probability history.
        Uses torch.inference_mode() for optimal speed and memory efficiency.

        Args:
            features: 63-element feature vector (array, list, or tensor).

        Returns:
            Tuple of:
            - raw_probabilities: np.ndarray shape (28,)
            - logits: np.ndarray shape (28,)
            - top1_idx: int
            - top1_label: str
            - top1_confidence: float

        Raises:
            ValueError: If input shape is not 63 elements, or contains NaN/Inf values.
        """
        if isinstance(features, torch.Tensor):
            arr = features.detach().cpu().numpy()
        else:
            arr = np.asarray(features, dtype=np.float32)

        if arr.size != 63 or arr.ndim > 2 or (arr.ndim == 2 and arr.shape != (1, 63)):
            raise ValueError(f"Expected 63-dimensional feature vector, got shape: {arr.shape}")

        arr = arr.reshape(1, 63).astype(np.float32)

        if not np.isfinite(arr).all():
            raise ValueError("Input features contain non-finite values (NaN or Inf).")

        feat_tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=1)

        raw_probs_np = probs[0].cpu().numpy()
        logits_np = logits[0].cpu().numpy()
        top1_idx = int(np.argmax(raw_probs_np))
        top1_conf = float(raw_probs_np[top1_idx])
        top1_label = get_semantic_label(top1_idx)

        return raw_probs_np, logits_np, top1_idx, top1_label, top1_conf

    def predict(
        self,
        features: Union[np.ndarray, list, torch.Tensor],
        raw_only: bool = False,
        history_deque: Optional[deque] = None
    ) -> InferenceResult:
        """
        Executes inference on a single 63-dimensional normalized landmark vector.
        Applies temporal probability smoothing across a rolling 3-frame window.

        Args:
            features: 63-element feature vector (array, list, or tensor).
            raw_only: If True, bypasses temporal probability smoothing (returns raw single frame).
            history_deque: Optional external deque for per-session smoothing isolation.
                           If None, uses the engine's internal self._prob_history.

        Returns:
            InferenceResult with predicted label, index, confidence, probabilities, and logits.

        Raises:
            ValueError: If input shape is not 63 elements, or contains NaN/Inf values.
        """
        raw_probs_np, logits_np, raw_idx, raw_label, raw_conf = self.predict_stateless(features)

        # Target history queue: external session queue if provided, else internal
        target_history = history_deque if history_deque is not None else self._prob_history

        # Apply 3-frame moving-average probability smoothing
        if self.enable_smoothing and not raw_only and self.smoothing_window > 1:
            target_history.append(raw_probs_np)
            smoothed_probs = np.mean(target_history, axis=0)
            # Ensure strict sum-to-1 normalization against float roundoff
            prob_sum = smoothed_probs.sum()
            if prob_sum > 0:
                smoothed_probs = smoothed_probs / prob_sum
            pred_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[pred_idx])
            final_probs = smoothed_probs
        else:
            pred_idx = raw_idx
            confidence = raw_conf
            final_probs = raw_probs_np

        pred_label = get_semantic_label(pred_idx)

        return InferenceResult(
            label=pred_label,
            class_index=pred_idx,
            confidence=confidence,
            probabilities=final_probs,
            logits=logits_np
        )
