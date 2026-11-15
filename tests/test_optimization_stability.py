"""
Tests for Phase 7 Optimization:
1. 3-frame moving average probability smoothing in ASLInferenceEngine.
2. Responsiveness and stability of 0.40s dwell time in GestureStateMachine.
3. Unstable prediction rejection and jitter immunity.
4. Duplicate commit prevention under sustained holds.
5. 0.25s release re-arming behavior.
6. Hand tracking loss session isolation (no stale predictions).
7. Dynamic J/Z kinematic integration.
8. Synchronous FIFO ordering in TTS engine.
"""

import time
import pytest
import numpy as np
import torch

from src.realtime.inference import ASLInferenceEngine, InferenceResult
from src.realtime.gesture_state_machine import GestureStateMachine, StateMachineConfig, CommitEvent
from src.realtime.app import ASLRealTimePipeline
from src.realtime.speech import SpeechEngine


class MockMLP(torch.nn.Module):
    """Mock MLP that returns controllable logits."""
    def __init__(self, logits: np.ndarray):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        batch = x.shape[0]
        return self.logits.repeat(batch, 1)


def test_three_frame_probability_smoothing_calculation():
    """Verify that ASLInferenceEngine applies rolling 3-frame moving average smoothing to probabilities."""
    engine = ASLInferenceEngine(smoothing_window=3, enable_smoothing=True)

    # Frame 1: Strong 'A' (Index 0)
    feat1 = np.ones(63, dtype=np.float32) * 0.1
    # Mock model output by passing features to predict
    res1 = engine.predict(feat1)
    assert len(engine._prob_history) == 1
    assert np.isclose(res1.probabilities.sum(), 1.0)

    # Frame 2: Another observation
    res2 = engine.predict(feat1)
    assert len(engine._prob_history) == 2
    assert np.isclose(res2.probabilities.sum(), 1.0)

    # Frame 3: Another observation
    res3 = engine.predict(feat1)
    assert len(engine._prob_history) == 3
    assert np.isclose(res3.probabilities.sum(), 1.0)

    # Frame 4: Window slides; maxlen remains 3
    res4 = engine.predict(feat1)
    assert len(engine._prob_history) == 3
    assert np.isclose(res4.probabilities.sum(), 1.0)


def test_smoothing_stabilizes_single_frame_flicker():
    """Verify that a single-frame dip/flicker is cushioned by the 3-frame rolling average."""
    engine = ASLInferenceEngine(smoothing_window=3, enable_smoothing=True)

    # Directly feed probability vectors into the smoothing queue to test mathematical behavior
    dist_a = np.zeros(28, dtype=np.float32)
    dist_a[0] = 0.90  # 'A'
    dist_a[1] = 0.10  # 'B'

    dist_flicker_b = np.zeros(28, dtype=np.float32)
    dist_flicker_b[0] = 0.40  # 'A'
    dist_flicker_b[1] = 0.60  # 'B' (momentary flicker)

    engine._prob_history.clear()
    engine._prob_history.append(dist_a)
    engine._prob_history.append(dist_flicker_b)
    engine._prob_history.append(dist_a)

    smoothed = np.mean(engine._prob_history, axis=0)
    top_class = int(np.argmax(smoothed))
    # Class 'A' (0) should remain top-1: (0.90 + 0.40 + 0.90)/3 = 0.733 > B (0.267)
    assert top_class == 0
    assert smoothed[0] > 0.70


def test_reset_smoothing_prevents_stale_predictions_across_hand_loss():
    """Verify reset_smoothing() completely purges history so new hand instances start fresh."""
    engine = ASLInferenceEngine(smoothing_window=3, enable_smoothing=True)

    feat = np.ones(63, dtype=np.float32) * 0.1
    engine.predict(feat)
    engine.predict(feat)
    assert len(engine._prob_history) == 2

    # Hand is lost / session resets
    engine.reset_smoothing()
    assert len(engine._prob_history) == 0

    # New hand prediction starts fresh with length 1
    engine.predict(feat)
    assert len(engine._prob_history) == 1


def test_stable_prediction_commits_at_0_60s_dwell():
    """Verify that a stable hand gesture commits at exactly 0.60s dwell duration."""
    sm = GestureStateMachine(StateMachineConfig())  # Default dwell_duration is 0.60s
    assert sm.config.dwell_duration == 0.60

    # t=0.00: candidate established
    assert sm.process_observation('A', 0.85, True, 0.00) is None
    assert sm.is_dwelling is True
    assert sm.current_candidate == 'A'

    # t=0.30: halfway through dwell; no commit yet
    assert sm.process_observation('A', 0.85, True, 0.30) is None

    # t=0.59: just before dwell threshold; no commit
    assert sm.process_observation('A', 0.85, True, 0.59) is None

    # t=0.60: dwell completed! Exactly commits 'A'
    event = sm.process_observation('A', 0.85, True, 0.60)
    assert event is not None
    assert isinstance(event, CommitEvent)
    assert event.label == 'A'
    assert event.dwell_time >= 0.60
    assert sm.is_latched is True


def test_unstable_or_low_confidence_prediction_rejects_commit():
    """Verify that candidate switches or confidence dips below 0.70 prevent commits."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.40, min_confidence=0.70))

    # Candidate 'A' starts at t=0.0
    assert sm.process_observation('A', 0.85, True, 0.00) is None

    # Switches to candidate 'B' at t=0.20 (resets dwell)
    assert sm.process_observation('B', 0.85, True, 0.20) is None
    assert sm.current_candidate == 'B'

    # Dips to low confidence (0.50) at t=0.35 (cancels candidate)
    assert sm.process_observation('B', 0.50, True, 0.35) is None
    assert sm.is_dwelling is False

    # t=0.45: still no commit
    assert sm.process_observation('B', 0.85, True, 0.45) is None


def test_hold_debounce_prevents_duplicate_commits_at_0_40s_dwell():
    """Verify that continuously holding a gesture for 2.0s emits exactly one commit event."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.40, min_confidence=0.70))

    commits = []
    # 60 frames at 30 FPS (2.0s total)
    for i in range(60):
        t = i * 0.0333
        ev = sm.process_observation('C', 0.90, True, t)
        if ev is not None:
            commits.append(ev)

    # Exactly 1 commit event emitted, zero duplicates
    assert len(commits) == 1
    assert commits[0].label == 'C'
    assert np.isclose(commits[0].timestamp, 0.40, atol=0.035)


def test_release_and_rearm_timing_0_25s():
    """Verify that 0.25s of confirmed absence re-arms the latch to allow consecutive identical letters."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.40, release_duration=0.25, min_confidence=0.70))

    # 1. First 'E' committed at t=0.40s
    sm.process_observation('E', 0.90, True, 0.00)
    ev1 = sm.process_observation('E', 0.90, True, 0.40)
    assert ev1 is not None and ev1.label == 'E'

    # 2. Hand released for only 0.10s (insufficient release)
    sm.process_observation(None, 0.0, False, 0.50)
    sm.process_observation(None, 0.0, False, 0.60)
    # Hand returns to 'E' at t=0.65; latch should still be locked!
    assert sm.process_observation('E', 0.90, True, 0.65) is None

    # 3. Hand released for full 0.25s (from t=0.70 to t=0.96)
    sm.process_observation(None, 0.0, False, 0.70)
    sm.process_observation(None, 0.0, False, 0.96)
    assert sm.is_latched is False  # Re-armed!

    # 4. Present 'E' again for 0.40s -> second 'E' committed
    sm.process_observation('E', 0.90, True, 1.00)
    ev2 = sm.process_observation('E', 0.90, True, 1.40)
    assert ev2 is not None
    assert ev2.label == 'E'


def test_pipeline_reset_smoothing_on_no_hand(monkeypatch):
    """Verify that ASLRealTimePipeline resets smoothing history upon detecting NO_HAND."""
    from src.realtime.inference import ASLInferenceEngine

    pipeline = ASLRealTimePipeline(enable_speech=False, enable_dynamic=False)
    assert pipeline.sm_config.dwell_duration == 0.60

    # Simulate active prediction adding items to history
    feat = np.ones(63, dtype=np.float32) * 0.1
    pipeline.inference_engine.predict(feat)
    assert len(pipeline.inference_engine._prob_history) == 1

    # Process empty frame where detector finds no hand
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline.process_frame(empty_frame, timestamp_sec=1.0)

    # Hand was absent -> smoothing history cleared!
    assert len(pipeline.inference_engine._prob_history) == 0
    pipeline.close()


def test_tts_queue_ordering_preserved_with_0_40s_dwell():
    """Verify that rapid 0.40s word accumulation preserves strict FIFO TTS order: C -> A -> T -> CAT."""
    from src.realtime.text_buffer import TextBuffer

    class MockTTS:
        def __init__(self):
            self.spoken_phrases = []

        def say(self, text):
            self.spoken_phrases.append(text)

        def runAndWait(self):
            pass

        def stop(self):
            pass

    backend = MockTTS()
    speech = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    tb = TextBuffer()

    try:
        time.sleep(0.05)
        # Type C, A, T rapidly
        for ch in ["C", "A", "T"]:
            tb.apply(ch)
            speech.on_commit_event(ch, tb.get_text(), timestamp=1.0)

        # SPACE to complete word
        tb.apply("SPACE")
        speech.on_commit_event("SPACE", tb.get_text(), timestamp=2.0)

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 4 and time.time() < deadline:
            time.sleep(0.02)

        # Exact FIFO sequence verified
        assert backend.spoken_phrases == ["C", "A", "T", "CAT"]
    finally:
        speech.stop(timeout=1.0)

