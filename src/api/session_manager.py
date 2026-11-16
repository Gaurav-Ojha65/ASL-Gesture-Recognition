"""
Session Manager for ASL Gesture Recognition FastAPI Backend.
Maintains strictly isolated per-session state for WebSocket connections:
- 3-frame moving-average probability smoothing deque.
- Temporal dynamic trajectory sequence buffer.
- GestureStateMachine (dwell: 0.60s, release: 0.25s, min_conf: 0.70).
- TextBuffer.
- DynamicTrajectoryClassifier.
- LandmarkedNormalizer.
- Speech metadata and sequence tracking.
"""

import time
import uuid
import threading
from collections import deque
from typing import Optional, Dict, Any, List
import numpy as np

from src.landmarks.normalizer import LandmarkNormalizer
from src.realtime.gesture_state_machine import (
    GestureStateMachine,
    StateMachineConfig,
    CommitEvent
)
from src.realtime.text_buffer import TextBuffer
from src.realtime.temporal_sequence import (
    TemporalSequenceBuffer,
    DynamicTrajectoryClassifier,
    DynamicRecognitionResult
)


class UserSessionContext:
    """
    Encapsulates all mutable temporal and interaction state for one active client.
    Guarantees 100% isolation from other concurrent users.
    """

    def __init__(
        self,
        session_id: str,
        dwell_duration: float = 0.60,
        release_duration: float = 0.25,
        min_confidence: float = 0.70,
        smoothing_window: int = 3,
        seq_buffer_len: int = 35,
    ):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active_at = self.created_at

        # Session-isolated temporal probability smoothing
        self.smoothing_window = smoothing_window
        self.prob_history: deque = deque(maxlen=self.smoothing_window)

        # Normalization & trajectory buffers
        self.normalizer = LandmarkNormalizer()
        self.seq_buffer = TemporalSequenceBuffer(max_len=seq_buffer_len)
        self.dynamic_classifier = DynamicTrajectoryClassifier()

        # Session-isolated gesture state machine
        self.sm_config = StateMachineConfig(
            dwell_duration=dwell_duration,
            release_duration=release_duration,
            min_confidence=min_confidence,
        )
        self.state_machine = GestureStateMachine(self.sm_config)

        # Session-isolated text buffer & speech state
        self.text_buffer = TextBuffer()
        self.speech_enabled: bool = True
        self.last_spoken_word: str = ""
        self.last_speech_timestamp: float = 0.0

        # Frame sequence & backpressure tracking
        self.last_sequence_id: int = -1
        self.is_processing: bool = False
        self._lock = threading.Lock()

    def touch(self) -> None:
        """Updates last active timestamp."""
        self.last_active_at = time.time()

    def reset_smoothing(self) -> None:
        """Wipes the session's private smoothing deque (e.g. on hand tracking loss)."""
        self.prob_history.clear()

    def reset_state(self) -> None:
        """Resets text buffer, state machine, and trajectory buffer."""
        self.text_buffer.clear()
        self.state_machine.reset()
        self.prob_history.clear()
        self.seq_buffer.clear()
        self.last_spoken_word = ""

    def cleanup(self) -> None:
        """Cleans up session resources on connection teardown."""
        self.reset_state()


class SessionManager:
    """
    Thread-safe registry managing active UserSessionContext instances.
    Provides bounded storage, TTL eviction, and deterministic cleanup.
    """

    MAX_SESSIONS: int = 100
    SESSION_TTL_SECONDS: float = 300.0  # 5 minutes idle TTL

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_seconds: float = SESSION_TTL_SECONDS):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, UserSessionContext] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: Optional[str] = None) -> UserSessionContext:
        """
        Creates a new isolated session context.
        If max_sessions is reached, evicts oldest stale session.
        """
        with self._lock:
            self._evict_stale_locked()

            if len(self._sessions) >= self.max_sessions:
                # Evict oldest session by last_active_at
                oldest_id = min(self._sessions.keys(), key=lambda k: self._sessions[k].last_active_at)
                old_sess = self._sessions.pop(oldest_id)
                old_sess.cleanup()

            sid = session_id.strip() if session_id and session_id.strip() else f"sess_{uuid.uuid4().hex[:12]}"
            ctx = UserSessionContext(session_id=sid)
            self._sessions[sid] = ctx
            return ctx

    def get_session(self, session_id: str) -> Optional[UserSessionContext]:
        """Retrieves an existing session if present, updating its activity timestamp."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is not None:
                ctx.touch()
            return ctx

    def remove_session(self, session_id: str) -> None:
        """Removes and cleans up a session context."""
        with self._lock:
            ctx = self._sessions.pop(session_id, None)
            if ctx is not None:
                ctx.cleanup()

    def active_count(self) -> int:
        """Returns the number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def _evict_stale_locked(self) -> int:
        """Internal helper to evict sessions exceeding TTL (must be called with _lock held)."""
        now = time.time()
        stale_ids = [
            sid for sid, ctx in self._sessions.items()
            if (now - ctx.last_active_at) > self.ttl_seconds
        ]
        for sid in stale_ids:
            ctx = self._sessions.pop(sid)
            ctx.cleanup()
        return len(stale_ids)

    def evict_stale_sessions(self) -> int:
        """Public method to evict stale sessions."""
        with self._lock:
            return self._evict_stale_locked()

    def close_all(self) -> None:
        """Closes and cleans up all active sessions."""
        with self._lock:
            for ctx in self._sessions.values():
                ctx.cleanup()
            self._sessions.clear()

