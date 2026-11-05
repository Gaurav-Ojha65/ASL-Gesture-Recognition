import pytest
import math
from src.realtime.gesture_state_machine import GestureStateMachine, StateMachineConfig, CommitEvent

def test_1_single_observation_does_not_commit():
    """1. A single observation does not immediately commit a letter."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    event = sm.process_observation(label='A', confidence=0.9, hand_detected=True, timestamp=0.0)
    assert event is None
    assert sm.is_dwelling is True
    assert sm.current_candidate == 'A'

def test_2_stable_observation_commits_after_dwell():
    """2. A stable observation commits after the dwell duration."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    # Frame 1: t=0.0s
    assert sm.process_observation('A', 0.95, True, 0.0) is None
    # Frame 2: t=0.3s (not yet)
    assert sm.process_observation('A', 0.95, True, 0.3) is None
    # Frame 3: t=0.59s (not yet)
    assert sm.process_observation('A', 0.95, True, 0.59) is None
    # Frame 4: t=0.60s (dwell completed!)
    event = sm.process_observation('A', 0.95, True, 0.60)
    assert event is not None
    assert isinstance(event, CommitEvent)
    assert event.label == 'A'
    assert event.timestamp == 0.60
    assert event.confidence == 0.95
    assert event.dwell_time >= 0.60
    assert sm.is_latched is True
    assert sm.latched_label == 'A'

def test_3_holding_emits_only_one_commit():
    """3. Holding a committed gesture for several seconds emits only one commit."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    sm.process_observation('A', 0.95, True, 0.0)
    event = sm.process_observation('A', 0.95, True, 0.6)
    assert event is not None
    assert event.label == 'A'

    # Hold for 5 seconds (many frames)
    for t in [0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        evt = sm.process_observation('A', 0.95, True, t)
        assert evt is None, f"Duplicate commit emitted at t={t}"

def test_4_short_candidate_change_resets_dwell():
    """4. A short candidate change resets the dwell timer."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    sm.process_observation('A', 0.95, True, 0.0)
    sm.process_observation('A', 0.95, True, 0.4) # dwelt 0.4s

    # Flicker to B at t=0.5s
    assert sm.process_observation('B', 0.95, True, 0.5) is None
    assert sm.current_candidate == 'B'

    # Switch back to A at t=0.6s (A must restart dwell from 0.6s)
    assert sm.process_observation('A', 0.95, True, 0.6) is None
    assert sm.current_candidate == 'A'
    # At t=1.1s, A has dwelt 0.5s (0.6 -> 1.1), not yet 0.6s
    assert sm.process_observation('A', 0.95, True, 1.1) is None
    # At t=1.2s, A has dwelt 0.6s (0.6 -> 1.2), commits!
    event = sm.process_observation('A', 0.95, True, 1.2)
    assert event is not None
    assert event.label == 'A'

def test_5_stable_new_gesture_can_commit():
    """5. A stable new gesture can commit without hand disappearing."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    # Commit A
    sm.process_observation('A', 0.9, True, 0.0)
    assert sm.process_observation('A', 0.9, True, 0.6) is not None

    # Switch directly to B
    sm.process_observation('B', 0.9, True, 1.0)
    assert sm.process_observation('B', 0.9, True, 1.3) is None
    # Commit B at 1.0 + 0.6 = 1.6s
    event_b = sm.process_observation('B', 0.9, True, 1.6)
    assert event_b is not None
    assert event_b.label == 'B'
    assert sm.latched_label == 'B'

def test_6_same_letter_commits_again_after_release():
    """6. The same letter can be committed again after a sufficiently long NO_HAND release."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, release_duration=0.25, min_confidence=0.7))
    # Commit first A
    sm.process_observation('A', 0.9, True, 0.0)
    assert sm.process_observation('A', 0.9, True, 0.6) is not None

    # NO_HAND starts at t=1.0s, lasts until t=1.3s (0.3s >= 0.25s)
    sm.process_observation(None, 0.0, False, 1.0)
    sm.process_observation(None, 0.0, False, 1.3)
    assert sm.is_latched is False # re-armed!

    # Second A appears at t=1.5s
    sm.process_observation('A', 0.9, True, 1.5)
    event_second_a = sm.process_observation('A', 0.9, True, 2.1)
    assert event_second_a is not None
    assert event_second_a.label == 'A'

def test_7_brief_hand_disappearance_does_not_rearm():
    """7. A brief hand disappearance (< release_duration) does not re-arm the latch."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, release_duration=0.25, min_confidence=0.7))
    # Commit A
    sm.process_observation('A', 0.9, True, 0.0)
    assert sm.process_observation('A', 0.9, True, 0.6) is not None

    # Brief hand disappearance for 0.10s (t=1.0 to t=1.1)
    sm.process_observation(None, 0.0, False, 1.0)
    sm.process_observation(None, 0.0, False, 1.1) # 0.1s < 0.25s

    # Hand returns with A at t=1.15s
    sm.process_observation('A', 0.9, True, 1.15)
    # Even if held until 2.0s, it must NOT commit because latch was never released
    for t in [1.5, 1.8, 2.0]:
        assert sm.process_observation('A', 0.9, True, t) is None

def test_8_low_confidence_does_not_commit():
    """8. Low-confidence predictions do not commit."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    for t in [0.0, 0.3, 0.6, 1.0, 2.0]:
        evt = sm.process_observation('A', 0.65, True, t) # 0.65 < 0.70
        assert evt is None

def test_9_low_confidence_does_not_falsely_rearm():
    """9. Low-confidence predictions do not falsely count as a confirmed NO_HAND release."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, release_duration=0.25, min_confidence=0.7))
    # Commit A
    sm.process_observation('A', 0.9, True, 0.0)
    assert sm.process_observation('A', 0.9, True, 0.6) is not None
    assert sm.latched_label == 'A'

    # Several low-confidence frames occur
    for t in [1.0, 1.1, 1.2, 1.3, 1.4]:
        sm.process_observation('A', 0.50, True, t)
        assert sm.is_latched is True # Latch must remain active!

    # High-confidence A returns: latch prevents re-commit
    sm.process_observation('A', 0.9, True, 1.5)
    assert sm.process_observation('A', 0.9, True, 2.1) is None

def test_10_space_and_delete_can_be_committed():
    """10. SPACE and DELETE can be emitted as stable semantic events."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    # Commit SPACE
    sm.process_observation('SPACE', 0.85, True, 0.0)
    event_space = sm.process_observation('SPACE', 0.85, True, 0.6)
    assert event_space is not None
    assert event_space.label == 'SPACE'

    # Switch to DELETE
    sm.process_observation('DELETE', 0.90, True, 1.0)
    event_del = sm.process_observation('DEL', 0.90, True, 1.6)
    assert event_del is not None
    assert event_del.label == 'DELETE'

def test_11_nothing_never_emits_commit():
    """11. NOTHING never emits a literal NOTHING character commit."""
    sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, min_confidence=0.7))
    for t in [0.0, 0.3, 0.6, 1.0, 2.0]:
        evt = sm.process_observation('NOTHING', 0.99, False, t)
        assert evt is None
        evt2 = sm.process_observation('NOTHING', 0.99, True, t + 0.05)
        assert evt2 is None

def test_12_invalid_labels_handled():
    """12. Invalid labels are handled with ValueError when hand is detected."""
    sm = GestureStateMachine()
    with pytest.raises(ValueError):
        sm.process_observation('INVALID_CLASS', 0.9, True, 1.0)

def test_13_invalid_confidence_handled():
    """13. Invalid confidence values are rejected with ValueError."""
    sm = GestureStateMachine()
    with pytest.raises(ValueError):
        sm.process_observation('A', -0.1, True, 1.0)
    with pytest.raises(ValueError):
        sm.process_observation('A', 1.05, True, 1.0)
    with pytest.raises(ValueError):
        sm.process_observation('A', float('nan'), True, 1.0)
    with pytest.raises(ValueError):
        sm.process_observation('A', float('inf'), True, 1.0)

def test_14_non_finite_timestamps_handled():
    """14. Non-finite timestamps are rejected with ValueError."""
    sm = GestureStateMachine()
    with pytest.raises(ValueError):
        sm.process_observation('A', 0.9, True, float('nan'))
    with pytest.raises(ValueError):
        sm.process_observation('A', 0.9, True, float('inf'))

def test_15_decreasing_timestamps_handled():
    """15. Decreasing timestamps are rejected with ValueError."""
    sm = GestureStateMachine()
    sm.process_observation('A', 0.9, True, 10.0)
    with pytest.raises(ValueError):
        sm.process_observation('A', 0.9, True, 9.9)

def test_16_deterministic_event_replay():
    """16. Replaying the same sequence and timestamps yields identical events."""
    sequence = [
        ('A', 0.9, True, 0.0),
        ('A', 0.9, True, 0.3),
        ('A', 0.9, True, 0.6), # Commit A
        ('A', 0.9, True, 0.8),
        (None, 0.0, False, 1.0),
        (None, 0.0, False, 1.3), # Released
        ('A', 0.95, True, 1.4),
        ('A', 0.95, True, 2.0), # Commit A
        ('SPACE', 0.9, True, 2.5),
        ('SPACE', 0.9, True, 3.1), # Commit SPACE
        ('B', 0.9, True, 3.5),
        ('B', 0.9, True, 4.1), # Commit B
    ]

    def run_seq():
        sm = GestureStateMachine(StateMachineConfig(dwell_duration=0.6, release_duration=0.25, min_confidence=0.7))
        events = []
        for obs in sequence:
            evt = sm.process_observation(*obs)
            if evt is not None:
                events.append((evt.label, evt.timestamp, evt.confidence))
        return events

    events1 = run_seq()
    events2 = run_seq()
    assert events1 == events2
    assert len(events1) == 4
    labels = [e[0] for e in events1]
    assert labels == ['A', 'A', 'SPACE', 'B']
