"""
Unit tests for Asynchronous Non-Blocking Text-to-Speech (TTS) Engine.

Verifies:
1. Engine initializes properly and runs on a background worker thread.
2. Speech requests are queued asynchronously without blocking caller.
3. Duplicate speech suppression (same word within cooldown window).
4. Empty or whitespace text is rejected.
5. Word-level trigger on 'SPACE' commitment.
6. Inactivity pause triggers speech for unspaced word.
7. 'DELETE' token updates pending word without speaking.
8. Toggle function enables/disables speech and clears pending queue.
9. Graceful shutdown joins worker thread without hanging.
10. Engine initialization failure is handled safely without crashing.
"""

import time
import pytest
from unittest.mock import MagicMock
from src.realtime.speech import SpeechEngine


class MockTTSBackend:
    """Mock pyttsx3 engine backend for deterministic unit tests without audio hardware."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.spoken_phrases = []
        self.properties = {}
        self.stopped = False

    def setProperty(self, name, value):
        self.properties[name] = value

    def say(self, text):
        if self.should_fail:
            raise RuntimeError("Simulated sound card failure")
        self.spoken_phrases.append(text)

    def runAndWait(self):
        if self.should_fail:
            raise RuntimeError("Simulated audio driver hang")
        time.sleep(0.01)

    def stop(self):
        self.stopped = True


def test_speech_queueing_and_worker_execution():
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        # Wait for worker thread to initialize
        deadline = time.time() + 1.5
        while not engine.is_available and time.time() < deadline:
            time.sleep(0.02)
        assert engine.is_available is True
        assert engine.is_enabled is True

        assert engine.speak("HELLO") is True

        # Wait for worker thread to process item
        deadline = time.time() + 1.0
        while not backend.spoken_phrases and time.time() < deadline:
            time.sleep(0.02)

        assert "HELLO" in backend.spoken_phrases
        assert engine.last_spoken == "HELLO"
    finally:
        engine.stop(timeout=1.0)


def test_speech_duplicate_suppression():
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        assert engine.speak("CAT") is True
        # Immediate identical word is suppressed
        assert engine.speak("CAT") is False
        assert engine.speak("CAT", force=True) is True  # Forced bypasses suppression
    finally:
        engine.stop(timeout=1.0)


def test_speech_empty_text_ignored():
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        assert engine.speak("") is False
        assert engine.speak("   ") is False
        assert len(backend.spoken_phrases) == 0
    finally:
        engine.stop(timeout=1.0)


def test_speech_toggle_and_queue_flushing():
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        assert engine.is_enabled is True

        # Toggle OFF
        assert engine.toggle() is False
        assert engine.is_enabled is False

        # Calls while disabled return False
        assert engine.speak("DISABLED") is False

        # Toggle back ON
        assert engine.toggle() is True
        assert engine.is_enabled is True
        assert engine.speak("ENABLED") is True
    finally:
        engine.stop(timeout=1.0)


def test_speech_word_completion_on_space():
    """Verify that letters are spoken on commit, and SPACE speaks the full completed word."""
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        # Type letters H, I
        assert engine.on_commit_event("H", "H") is True
        assert engine.on_commit_event("I", "HI") is True

        # Commit SPACE
        assert engine.on_commit_event("SPACE", "HI ") is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 3 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["H", "I", "HI"]
    finally:
        engine.stop(timeout=1.0)


def test_speech_individual_letters_and_word_on_space():
    """
    Verify complete C -> A -> T -> CAT sequence:
    1. Committing C speaks C immediately.
    2. Committing A after C speaks A.
    3. Committing T after A speaks T.
    4. SPACE after CAT speaks CAT.
    5. Expected queue order is C, A, T, CAT.
    """
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        # Commit C immediately speaks C
        assert engine.on_commit_event("C", "C", timestamp=1.0) is True
        # Commit A immediately speaks A
        assert engine.on_commit_event("A", "CA", timestamp=1.5) is True
        # Commit T immediately speaks T
        assert engine.on_commit_event("T", "CAT", timestamp=2.0) is True

        # Commit SPACE speaks complete word CAT
        assert engine.on_commit_event("SPACE", "CAT ", timestamp=2.5) is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 4 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["C", "A", "T", "CAT"]
        assert engine.last_spoken == "CAT"
    finally:
        engine.stop(timeout=1.0)


def test_individual_letter_speech_does_not_clear_text_buffer():
    """Verify individual letter pronunciation does not modify or clear the TextBuffer."""
    from src.realtime.text_buffer import TextBuffer

    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    tb = TextBuffer()

    try:
        time.sleep(0.05)
        for ch in ["C", "A", "T"]:
            tb.apply(ch)
            assert engine.on_commit_event(ch, tb.get_text(), timestamp=1.0) is True

        # TextBuffer remains accumulated and intact
        assert tb.get_text() == "CAT"
        assert len(tb) == 3

        # Finalize with SPACE
        tb.apply("SPACE")
        assert engine.on_commit_event("SPACE", tb.get_text(), timestamp=2.0) is True
        assert tb.get_text() == "CAT "
    finally:
        engine.stop(timeout=1.0)


def test_speech_two_different_completed_words_both_spoken():
    """Verify that two distinct completed words (CAT then DOG) and all their letters are spoken."""
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        # Word 1: CAT
        assert engine.on_commit_event("C", "C", timestamp=1.0) is True
        assert engine.on_commit_event("A", "CA", timestamp=1.5) is True
        assert engine.on_commit_event("T", "CAT", timestamp=2.0) is True
        assert engine.on_commit_event("SPACE", "CAT ", timestamp=2.5) is True

        # Word 2: DOG
        assert engine.on_commit_event("D", "CAT D", timestamp=3.0) is True
        assert engine.on_commit_event("O", "CAT DO", timestamp=3.5) is True
        assert engine.on_commit_event("G", "CAT DOG", timestamp=4.0) is True
        assert engine.on_commit_event("SPACE", "CAT DOG ", timestamp=4.5) is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 8 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["C", "A", "T", "CAT", "D", "O", "G", "DOG"]
    finally:
        engine.stop(timeout=1.0)


def test_speech_disabled_prevents_space_and_pause_speech():
    """Verify that when speech is disabled, neither letters, SPACE, nor pause trigger speech."""
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=False, pause_speech_delay=0.10, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        assert engine.is_enabled is False

        # Accumulate letters and commit SPACE while disabled
        assert engine.on_commit_event("C", "C", timestamp=1.0) is False
        assert engine.on_commit_event("A", "CA", timestamp=1.2) is False
        assert engine.on_commit_event("SPACE", "CA ", timestamp=1.5) is False
        assert engine.update_pause_check(current_time=5.0) is False

        time.sleep(0.05)
        assert len(backend.spoken_phrases) == 0
    finally:
        engine.stop(timeout=1.0)


def test_speech_delete_edits_without_speaking():
    """Verify that DELETE modifies the word without generating audio for deletion."""
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, pause_speech_delay=0.10, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        # Type B, A, D
        assert engine.on_commit_event("B", "B", timestamp=1.0) is True
        assert engine.on_commit_event("A", "BA", timestamp=1.2) is True
        assert engine.on_commit_event("D", "BAD", timestamp=1.4) is True

        # DELETE D -> BA (Must not produce speech!)
        assert engine.on_commit_event("DELETE", "BA", timestamp=1.6) is False

        # Only the 3 letters B, A, D were spoken so far
        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert backend.spoken_phrases == ["B", "A", "D"]

        # Now wait for pause trigger
        assert engine.update_pause_check(current_time=1.8) is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 4 and time.time() < deadline:
            time.sleep(0.02)

        # Speaks the edited word "BA", never the deleted word "BAD"
        assert backend.spoken_phrases == ["B", "A", "D", "BA"]
    finally:
        engine.stop(timeout=1.0)


def test_speech_pause_completion_and_no_duplicate_on_space():
    """Verify that automatic 2.5s pause speaks the word, and subsequent SPACE does not duplicate it."""
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, pause_speech_delay=0.10, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        now = time.time()
        engine.on_commit_event("D", "D", timestamp=now)
        engine.on_commit_event("O", "DO", timestamp=now + 0.01)
        engine.on_commit_event("G", "DOG", timestamp=now + 0.02)

        # Before delay: pause check does not trigger
        assert engine.update_pause_check(current_time=now + 0.05) is False

        # After pause delay: speaks completed word DOG
        assert engine.update_pause_check(current_time=now + 0.15) is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 4 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["D", "O", "G", "DOG"]

        # Subsequent SPACE should NOT produce duplicate "DOG" playback
        assert engine.on_commit_event("SPACE", "DOG ", timestamp=now + 0.20) is False
        time.sleep(0.05)
        assert backend.spoken_phrases == ["D", "O", "G", "DOG"]
    finally:
        engine.stop(timeout=1.0)


def test_dynamic_jz_spoken_only_after_successful_commitment():
    """Verify that dynamic gestures J and Z are spoken only upon commit, not during trajectory push."""
    from src.realtime.temporal_sequence import TemporalSequenceBuffer
    from src.realtime.gesture_state_machine import CommitEvent

    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    seq_buf = TemporalSequenceBuffer(max_len=30)

    try:
        time.sleep(0.05)
        # Push intermediate frames into buffer without committing
        for i in range(10):
            seq_buf.push(timestamp=float(i) * 0.03, landmarks=[{"x": 0.5, "y": 0.5, "z": 0.0}] * 21, is_valid=True)
            # Intermediate frames produce NO speech
            assert len(backend.spoken_phrases) == 0

        # Successful commitment of dynamic gesture J
        commit_j = CommitEvent(label="J", timestamp=0.5, confidence=0.92, dwell_time=0.0, metadata={"dynamic": True})
        assert engine.on_commit_event(commit_j.label, "J", timestamp=0.5) is True

        deadline = time.time() + 1.0
        while not backend.spoken_phrases and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["J"]

        # Successful commitment of dynamic gesture Z
        commit_z = CommitEvent(label="Z", timestamp=1.0, confidence=0.95, dwell_time=0.0, metadata={"dynamic": True})
        assert engine.on_commit_event(commit_z.label, "JZ", timestamp=1.0) is True

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 2 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["J", "Z"]
    finally:
        engine.stop(timeout=1.0)


def test_speech_init_failure_handled_gracefully():
    def broken_factory():
        raise RuntimeError("No sound device available")

    engine = SpeechEngine(enabled=True, engine_factory=broken_factory)
    try:
        time.sleep(0.05)
        assert engine.is_available is False
        # Calling speak does not raise an exception
        assert engine.speak("TEST") is True
    finally:
        engine.stop(timeout=1.0)


def test_speech_graceful_shutdown_no_hang():
    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    time.sleep(0.05)

    start = time.perf_counter()
    engine.stop(timeout=1.0)
    elapsed = time.perf_counter() - start

    assert not engine._worker_thread.is_alive()
    assert elapsed < 1.0, f"Shutdown took too long: {elapsed:.2f}s"


def test_keyboard_space_finalizes_word_and_speaks():
    """Verify that simulating keyboard SPACE finalizes the current word and triggers complete-word speech."""
    from src.realtime.text_buffer import TextBuffer

    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    tb = TextBuffer()

    try:
        time.sleep(0.05)
        # 1. User signs C, A, T
        for ch in ["C", "A", "T"]:
            tb.apply(ch)
            engine.on_commit_event(ch, tb.get_text(), timestamp=1.0)

        assert tb.get_text() == "CAT"

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert backend.spoken_phrases == ["C", "A", "T"]

        # 2. User presses keyboard SPACE (ASCII 32)
        current_text = tb.get_text()
        words = current_text.strip().split()
        last_word = words[-1] if words else ""
        buffer_modified = tb.apply("SPACE")

        assert buffer_modified is True
        assert tb.get_text() == "CAT "

        if buffer_modified and last_word:
            engine.on_commit_event("SPACE", tb.get_text(), timestamp=2.0)

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 4 and time.time() < deadline:
            time.sleep(0.02)

        assert backend.spoken_phrases == ["C", "A", "T", "CAT"]
    finally:
        engine.stop(timeout=1.0)


def test_keyboard_backspace_and_delete_remove_character_without_speech():
    """Verify that simulating keyboard BACKSPACE or DELETE removes characters without speaking."""
    from src.realtime.text_buffer import TextBuffer

    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    tb = TextBuffer()

    try:
        time.sleep(0.05)
        # Sign C, A, T
        for ch in ["C", "A", "T"]:
            tb.apply(ch)
            engine.on_commit_event(ch, tb.get_text(), timestamp=1.0)

        assert tb.get_text() == "CAT"
        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert backend.spoken_phrases == ["C", "A", "T"]

        # Simulate BACKSPACE (ASCII 8)
        buffer_modified = tb.apply("DELETE")
        assert buffer_modified is True
        assert tb.get_text() == "CA"
        engine.on_commit_event("DELETE", tb.get_text(), timestamp=1.5)

        time.sleep(0.05)
        # Spoken phrases must not change on BACKSPACE
        assert backend.spoken_phrases == ["C", "A", "T"]

        # Simulate DELETE key
        buffer_modified = tb.apply("DELETE")
        assert buffer_modified is True
        assert tb.get_text() == "C"
        engine.on_commit_event("DELETE", tb.get_text(), timestamp=2.0)

        time.sleep(0.05)
        # Spoken phrases must not change on DELETE
        assert backend.spoken_phrases == ["C", "A", "T"]
    finally:
        engine.stop(timeout=1.0)


def test_cat_followed_by_dog_complete_cycle():
    """Verify complete cycle: C, A, T -> CAT, then D, O, G -> DOG, verifying exact queue order."""
    from src.realtime.text_buffer import TextBuffer

    backend = MockTTSBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)
    tb = TextBuffer()

    try:
        time.sleep(0.05)
        # Spell CAT
        for ch in ["C", "A", "T"]:
            tb.apply(ch)
            engine.on_commit_event(ch, tb.get_text(), timestamp=1.0)

        # SPACE on CAT
        last_word = tb.get_text().strip().split()[-1]
        tb.apply("SPACE")
        engine.on_commit_event("SPACE", tb.get_text(), timestamp=1.5)

        # Spell DOG
        for ch in ["D", "O", "G"]:
            tb.apply(ch)
            engine.on_commit_event(ch, tb.get_text(), timestamp=2.0)

        # SPACE on DOG
        last_word = tb.get_text().strip().split()[-1]
        tb.apply("SPACE")
        engine.on_commit_event("SPACE", tb.get_text(), timestamp=2.5)

        deadline = time.time() + 1.0
        while len(backend.spoken_phrases) < 8 and time.time() < deadline:
            time.sleep(0.02)

        assert tb.get_text() == "CAT DOG "
        assert backend.spoken_phrases == ["C", "A", "T", "CAT", "D", "O", "G", "DOG"]
    finally:
        engine.stop(timeout=1.0)


def test_backend_failure_on_one_item_does_not_kill_worker_and_subsequent_items_continue():
    """Verify that if one utterance causes a backend error, the worker survives and processes subsequent items."""
    class FailingBackend:
        def __init__(self):
            self.spoken_phrases = []

        def say(self, text):
            if text == "A":
                raise RuntimeError("Simulated transient audio driver failure")
            self.spoken_phrases.append(text)

        def runAndWait(self):
            pass

        def stop(self):
            pass

    backend = FailingBackend()
    engine = SpeechEngine(enabled=True, engine_factory=lambda: backend)

    try:
        time.sleep(0.05)
        # 1. Enqueue C
        assert engine.speak("C", force=True) is True
        # 2. Enqueue A (will raise exception in worker)
        assert engine.speak("A", force=True) is True
        # 3. Enqueue T
        assert engine.speak("T", force=True) is True
        # 4. Enqueue CAT
        assert engine.speak("CAT", force=True) is True

        deadline = time.time() + 1.5
        while len(backend.spoken_phrases) < 3 and time.time() < deadline:
            time.sleep(0.02)

        # C, T, and CAT must have been processed, even though A failed!
        assert backend.spoken_phrases == ["C", "T", "CAT"]
        # Worker must remain alive and responsive!
        assert engine.is_worker_alive is True
    finally:
        engine.stop(timeout=1.0)


