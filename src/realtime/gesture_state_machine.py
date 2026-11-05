"""
Temporal Gesture State Machine Module for Real-Time ASL Gesture Recognition Interface.

Converts sequential noisy model observations into stable, deterministic character commit events.
Implements:
- Configurable dwell confirmation duration
- Hold debounce (prevents repeated commit on sustained sign)
- Re-arm release logic on confirmed NO_HAND
- Seamless distinct gesture transitions
- Confidence filtering (low confidence does not commit and does not re-arm)
- Robust input validation (finite, non-decreasing timestamps; probability ranges)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.landmarks.labels import SEMANTIC_CLASSES

@dataclass(frozen=True)
class StateMachineConfig:
    """
    Configuration parameters for the gesture temporal state machine.

    Provisional engineering defaults:
    - dwell_duration: 0.60s (sustained gesture duration before committing)
    - release_duration: 0.25s (sustained absence of hand before re-arming the same sign)
    - min_confidence: 0.70 (minimum prediction probability to be considered valid)
    """
    dwell_duration: float = 0.60
    release_duration: float = 0.25
    min_confidence: float = 0.70

    def __post_init__(self):
        if not math.isfinite(self.dwell_duration) or self.dwell_duration <= 0.0:
            raise ValueError(f"dwell_duration must be a finite positive number, got: {self.dwell_duration}")
        if not math.isfinite(self.release_duration) or self.release_duration <= 0.0:
            raise ValueError(f"release_duration must be a finite positive number, got: {self.release_duration}")
        if not math.isfinite(self.min_confidence) or not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in range [0.0, 1.0], got: {self.min_confidence}")


@dataclass(frozen=True)
class CommitEvent:
    """
    Represents a qualified character commit emitted by the state machine.
    """
    label: str
    timestamp: float
    confidence: float
    dwell_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GestureStateMachine:
    """
    Finite state machine for temporal gesture stabilization.
    """

    def __init__(self, config: Optional[StateMachineConfig] = None):
        self.config = config if config is not None else StateMachineConfig()

        # Candidate gesture tracking
        self.candidate_label: Optional[str] = None
        self.candidate_first_seen: Optional[float] = None
        self.candidate_last_seen: Optional[float] = None
        self.candidate_confidence: float = 0.0
        self.candidate_frame_count: int = 0

        # Debounce hold latch (label that was committed and is still held)
        self.committed_latch: Optional[str] = None

        # Release timer (records when NO_HAND was first observed after a commit)
        self.release_start_time: Optional[float] = None

        # Monotonic timestamp validation
        self.last_timestamp: Optional[float] = None

    def _reset_candidate(self) -> None:
        """Resets pending candidate state."""
        self.candidate_label = None
        self.candidate_first_seen = None
        self.candidate_last_seen = None
        self.candidate_confidence = 0.0
        self.candidate_frame_count = 0

    def reset(self) -> None:
        """Fully resets the state machine to its initial state."""
        self._reset_candidate()
        self.committed_latch = None
        self.release_start_time = None
        self.last_timestamp = None

    def process_observation(
        self,
        label: Optional[str],
        confidence: float,
        hand_detected: bool,
        timestamp: float
    ) -> Optional[CommitEvent]:
        """
        Processes a single prediction observation and returns a CommitEvent if a gesture qualifies.

        Args:
            label: Predicted semantic label (e.g., 'A'-'Z', 'SPACE', 'DELETE', 'NOTHING', 'NO_HAND')
            confidence: Prediction confidence / probability in [0.0, 1.0]
            hand_detected: Boolean indicating whether a hand was localized
            timestamp: Monotonic time in seconds

        Returns:
            Optional[CommitEvent]: A commit event if a gesture was confirmed, otherwise None.

        Raises:
            ValueError: On invalid inputs (non-finite values, out-of-range confidence, decreasing timestamps, unknown labels).
        """
        # 1. Validate Timestamp
        if not math.isfinite(timestamp):
            raise ValueError(f"Invalid non-finite timestamp: {timestamp}")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError(f"Decreasing timestamp: {timestamp} < {self.last_timestamp}")

        # 2. Validate Confidence
        if not math.isfinite(confidence):
            raise ValueError(f"Invalid non-finite confidence: {confidence}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence {confidence} is out of range [0.0, 1.0]")

        # 3. Validate & Normalize Label
        norm_label: Optional[str] = None
        if label is not None:
            raw_upper = label.strip().upper()
            if raw_upper in ('DEL', 'DELETE'):
                norm_label = 'DELETE'
            elif raw_upper in ('SPACE',):
                norm_label = 'SPACE'
            elif raw_upper in ('NOTHING', 'NO_HAND'):
                norm_label = 'NOTHING'
            elif len(raw_upper) == 1 and 'A' <= raw_upper <= 'Z':
                norm_label = raw_upper
            else:
                if hand_detected:
                    raise ValueError(f"Invalid semantic label: '{label}'")

        # If label is explicitly NOTHING or NO_HAND, treat as no hand
        is_active_hand = hand_detected and (norm_label not in ('NOTHING', None))

        # 4. Handle NO_HAND / NOTHING Observation
        if not is_active_hand:
            # Cancel any unfinished candidate
            self._reset_candidate()

            # Update release timer if a gesture is currently latched
            if self.committed_latch is not None:
                if self.release_start_time is None:
                    self.release_start_time = timestamp
                elapsed_release = timestamp - self.release_start_time
                if elapsed_release >= (self.config.release_duration - 1e-5):
                    # Sustained absence re-arms the latch!
                    self.committed_latch = None
                    self.release_start_time = None

            self.last_timestamp = timestamp
            return None

        # 5. Active Hand Detected: Hand is present, so cancel release timer
        self.release_start_time = None

        # Check Confidence Threshold
        if confidence < self.config.min_confidence:
            # Low-confidence prediction:
            # - Does NOT count as a confirmed NO_HAND release (hold latch remains locked)
            # - Does NOT advance or maintain candidate dwell
            self._reset_candidate()
            self.last_timestamp = timestamp
            return None

        # 6. Active High-Confidence Gesture
        # Check hold debounce: if this gesture is already latched, holding it emits no commit
        if norm_label == self.committed_latch:
            self._reset_candidate()
            self.last_timestamp = timestamp
            return None

        # Check candidate transition: if candidate changed, restart dwell timer
        if self.candidate_label != norm_label:
            self.candidate_label = norm_label
            self.candidate_first_seen = timestamp
            self.candidate_last_seen = timestamp
            self.candidate_confidence = confidence
            self.candidate_frame_count = 1
            self.last_timestamp = timestamp
            # A single frame never commits immediately
            return None

        # Candidate continues
        self.candidate_last_seen = timestamp
        self.candidate_confidence = confidence
        self.candidate_frame_count += 1
        self.last_timestamp = timestamp

        dwell_time = timestamp - self.candidate_first_seen
        if dwell_time >= (self.config.dwell_duration - 1e-5):
            # Dwell duration satisfied! Emit CommitEvent
            event = CommitEvent(
                label=norm_label,
                timestamp=timestamp,
                confidence=confidence,
                dwell_time=dwell_time,
                metadata={'frame_count': self.candidate_frame_count}
            )
            # Latch this gesture so holding it does not re-commit
            self.committed_latch = norm_label
            self._reset_candidate()
            return event

        return None

    @property
    def is_dwelling(self) -> bool:
        """Returns True if a candidate gesture is currently accumulating dwell time."""
        return self.candidate_label is not None

    @property
    def current_candidate(self) -> Optional[str]:
        """Returns the current candidate gesture being evaluated."""
        return self.candidate_label

    @property
    def is_latched(self) -> bool:
        """Returns True if a previously committed gesture is currently latched."""
        return self.committed_latch is not None

    @property
    def latched_label(self) -> Optional[str]:
        """Returns the label currently latched by hold debounce."""
        return self.committed_latch
