"""
Unit tests for the frozen MLP inference wrapper (ASLInferenceEngine).

Verifies:
1. Loads production checkpoint correctly.
2. Architecture is 63 inputs, 28 outputs, and 18,524 parameters.
3. Returns valid class labels and probabilities.
4. Probabilities are non-negative, finite, and sum to ~1.0.
5. Returns raw, unnormalized logits when requested.
6. Rejects invalid input shapes/dimensions.
7. Rejects non-finite values (NaN, Inf).
8. Agrees with authoritative labels in src/landmarks/labels.py.
9. Handles non-existent checkpoint gracefully.
"""

import os
import math
import numpy as np
import pytest
import torch

from src.realtime.inference import ASLInferenceEngine, InferenceResult
from src.landmarks.labels import SEMANTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.model.mlp import ASLLandmarkMLP


CHECKPOINT_PATH = "models/asl_landmark_mlp_best.pt"


@pytest.fixture(scope="module")
def engine():
    """Provides an initialized ASLInferenceEngine on CPU for deterministic tests."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    return ASLInferenceEngine(checkpoint_path=CHECKPOINT_PATH, device=torch.device("cpu"))


def test_checkpoint_exists():
    assert os.path.exists(CHECKPOINT_PATH)


def test_inference_engine_architecture(engine):
    """Verifies input/output dims and exact parameter count."""
    assert isinstance(engine.model, ASLLandmarkMLP)
    assert engine.param_count == 18524

    # Check layer dimensions
    assert engine.model.net[0].in_features == 63
    assert engine.model.net[0].out_features == 128
    assert engine.model.net[4].in_features == 128
    assert engine.model.net[4].out_features == 64
    assert engine.model.net[7].in_features == 64
    assert engine.model.net[7].out_features == 28


def test_inference_returns_valid_result(engine):
    """Verifies that predict returns valid labels, confidence, and shapes."""
    # Synthetic valid right-hand feature vector (63 floats)
    dummy_feat = np.random.randn(63).astype(np.float32)
    result = engine.predict(dummy_feat)

    assert isinstance(result, InferenceResult)
    assert result.label in SEMANTIC_CLASSES
    assert 0 <= result.class_index < 28
    assert result.label == IDX_TO_CLASS[result.class_index]
    assert 0.0 <= result.confidence <= 1.0
    assert result.probabilities.shape == (28,)
    assert result.logits.shape == (28,)


def test_probabilities_sum_to_one(engine):
    """Verifies probabilities are finite, non-negative, and sum to 1.0."""
    dummy_feat = np.ones(63, dtype=np.float32) * 0.1
    result = engine.predict(dummy_feat)

    assert np.all(np.isfinite(result.probabilities))
    assert np.all(result.probabilities >= 0.0)
    assert math.isclose(float(np.sum(result.probabilities)), 1.0, rel_tol=1e-5)
    assert math.isclose(result.confidence, float(result.probabilities[result.class_index]), rel_tol=1e-5)


def test_predict_logits_raw(engine):
    """Verifies model produces raw, unnormalized logits without softmax."""
    dummy_feat = torch.zeros((1, 63), dtype=torch.float32)
    logits = engine.predict_logits(dummy_feat)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 28)
    # Raw logits can be negative and typically do not sum to 1
    logits_np = logits.detach().cpu().numpy()[0]
    assert not math.isclose(float(np.sum(logits_np)), 1.0, abs_tol=0.1)


def test_rejects_invalid_feature_dimensions(engine):
    """Verifies shape mismatches are rejected with ValueError."""
    # Too short
    with pytest.raises(ValueError, match="63-dimensional"):
        engine.predict(np.zeros(62, dtype=np.float32))

    # Too long
    with pytest.raises(ValueError, match="63-dimensional"):
        engine.predict(np.zeros(64, dtype=np.float32))

    # Batch size > 1
    with pytest.raises(ValueError, match="63-dimensional"):
        engine.predict(np.zeros((2, 63), dtype=np.float32))

    # Empty
    with pytest.raises(ValueError, match="63-dimensional"):
        engine.predict(np.array([], dtype=np.float32))


def test_rejects_non_finite_features(engine):
    """Verifies NaN and Inf features are safely rejected."""
    # NaN
    nan_feat = np.zeros(63, dtype=np.float32)
    nan_feat[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        engine.predict(nan_feat)

    # +Inf
    inf_feat = np.zeros(63, dtype=np.float32)
    inf_feat[5] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        engine.predict(inf_feat)

    # -Inf
    ninf_feat = np.zeros(63, dtype=np.float32)
    ninf_feat[20] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        engine.predict(ninf_feat)


def test_class_mapping_agrees_with_labels(engine):
    """Verifies checkpoint mapping perfectly matches src/landmarks/labels.py."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "class_to_index" in ckpt:
        assert ckpt["class_to_index"] == CLASS_TO_IDX
    if "num_classes" in ckpt:
        assert ckpt["num_classes"] == 28
    if "classes" in ckpt:
        assert ckpt["classes"] == SEMANTIC_CLASSES


def test_missing_checkpoint_raises_error():
    """Verifies FileNotFoundError is raised if checkpoint does not exist."""
    with pytest.raises(FileNotFoundError):
        ASLInferenceEngine(checkpoint_path="non_existent_path.pt")
