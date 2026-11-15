"""
Tests for Robustness and Performance Evaluation Framework (Phase 4).
"""

import os
import pytest
import numpy as np

from scripts.evaluate_robustness import (
    DegradationPipeline,
    compute_classification_metrics,
    PipelineProfiler
)
from src.landmarks.normalizer import LandmarkNormalizer
from src.realtime.temporal_sequence import DynamicTrajectoryClassifier
from src.realtime.gesture_state_machine import GestureStateMachine, StateMachineConfig
from src.realtime.text_buffer import TextBuffer


class DummyInferenceEngine:
    """Mock inference engine for fast unit tests without loading weights."""
    def predict(self, feature_vector):
        return "A", 0.95


class TestDegradationPipeline:
    """Verify synthetic image degradation behaviors."""

    def test_brightness_scaling(self):
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        dark = DegradationPipeline.adjust_brightness(img, 0.5)
        bright = DegradationPipeline.adjust_brightness(img, 1.5)

        assert dark.shape == img.shape
        assert dark.dtype == np.uint8
        assert np.all(dark <= 55)

        assert bright.shape == img.shape
        assert bright.dtype == np.uint8
        assert np.all(bright >= 145)

    def test_gaussian_noise_reproducibility(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy1 = DegradationPipeline.add_gaussian_noise(img, sigma=20.0, seed=123)
        noisy2 = DegradationPipeline.add_gaussian_noise(img, sigma=20.0, seed=123)
        noisy3 = DegradationPipeline.add_gaussian_noise(img, sigma=20.0, seed=999)

        assert np.array_equal(noisy1, noisy2)
        assert not np.array_equal(noisy1, noisy3)

    def test_motion_blur_kernel(self):
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        blurred = DegradationPipeline.apply_motion_blur(img, kernel_size=9)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_occlusion_patch(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        occluded_center = DegradationPipeline.apply_occlusion(img, box_fraction=0.2, position="center")
        assert occluded_center.shape == img.shape
        # Center pixel should be black (0)
        center_val = occluded_center[50, 50, 0]
        assert center_val == 0

        occluded_corner = DegradationPipeline.apply_occlusion(img, box_fraction=0.2, position="top_left")
        assert occluded_corner[20, 20, 0] == 0

    def test_scale_and_rotation(self):
        img = np.full((80, 80, 3), 150, dtype=np.uint8)
        scaled = DegradationPipeline.apply_scale(img, scale_factor=0.8)
        rotated = DegradationPipeline.apply_rotation(img, angle_deg=15.0)

        assert scaled.shape == img.shape
        assert rotated.shape == img.shape


class TestClassificationMetrics:
    """Verify statistical and classification metrics computation."""

    def test_perfect_accuracy(self):
        y_true = ["A", "B", "C", "D"]
        y_pred = ["A", "B", "C", "D"]
        classes = ["A", "B", "C", "D"]

        metrics = compute_classification_metrics(y_true, y_pred, classes)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["sample_count"] == 4
        assert metrics["per_class"]["A"]["precision"] == 1.0
        assert metrics["per_class"]["A"]["recall"] == 1.0

    def test_imperfect_predictions(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "B", "B", "B"]
        classes = ["A", "B"]

        metrics = compute_classification_metrics(y_true, y_pred, classes)
        assert metrics["accuracy"] == 0.75
        assert metrics["per_class"]["A"]["precision"] == 1.0
        assert metrics["per_class"]["A"]["recall"] == 0.5
        assert metrics["per_class"]["B"]["recall"] == 1.0

    def test_empty_or_zero_division_guard(self):
        y_true = ["A"]
        y_pred = ["B"]
        classes = ["A", "B", "C"]

        metrics = compute_classification_metrics(y_true, y_pred, classes)
        assert metrics["accuracy"] == 0.0
        assert metrics["per_class"]["C"]["precision"] == 0.0
        assert metrics["per_class"]["C"]["recall"] == 0.0


class TestPipelineProfiler:
    """Verify latency profiling measurements."""

    def test_profiler_output_structure(self):
        profiler = PipelineProfiler(warmup_runs=2, benchmark_runs=5)
        normalizer = LandmarkNormalizer()
        dummy_engine = DummyInferenceEngine()
        dynamic_classifier = DynamicTrajectoryClassifier()
        config = StateMachineConfig(dwell_duration=0.5)
        state_machine = GestureStateMachine(config=config)
        text_buffer = TextBuffer()

        perf = profiler.profile_components(
            normalizer=normalizer,
            inference_engine=dummy_engine,
            dynamic_classifier=dynamic_classifier,
            state_machine=state_machine,
            text_buffer=text_buffer,
            speech_engine=None
        )

        assert "landmark_normalization" in perf
        assert "pytorch_mlp_inference" in perf
        assert "dynamic_trajectory_evaluation" in perf
        assert "state_machine_and_buffer" in perf

        norm_stats = perf["landmark_normalization"]
        assert "mean_ms" in norm_stats
        assert "median_ms" in norm_stats
        assert "p95_ms" in norm_stats
        assert norm_stats["mean_ms"] >= 0.0
        assert perf["speech_queue_enqueue"] is None
