"""
Temporal Sequence Processing and Kinematic Trajectory Recognition Module.

Provides modular, explainable dynamic ASL gesture recognition for movement-based signs
(specifically ASL letters 'J' and 'Z') without altering or interfering with the
static 28-class landmark MLP classifier.

Components:
1. TemporalSequenceBuffer: Rolling sliding-window ring buffer for multi-frame hand landmarks.
2. KinematicFeatureExtractor: Computes velocities, accelerations, path curvature, and turning points.
3. DynamicTrajectoryClassifier: Rule-assisted, explainable geometric trajectory classifier
   for 'J' (curved pinky stroke) and 'Z' (zig-zag index stroke) with posture gating and cooldown debounce.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    """Represents a single frame's geometric landmark observation."""
    timestamp: float
    landmarks: Optional[np.ndarray]  # Shape (21, 3) in image coordinates (or normalized)
    is_valid: bool
    handedness: str = "Right"
    features_63: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DynamicRecognitionResult:
    """Result emitted by the dynamic trajectory classifier."""
    detected: bool
    gesture: Optional[str]  # 'J', 'Z', or None
    confidence: float
    start_time: float
    end_time: float
    stroke_count: int
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalSequenceBuffer:
    """
    Rolling ring buffer for tracking consecutive hand landmark observations over time.
    Maintains a temporal history of up to max_len frames, handles tracking loss,
    and validates monotonic timestamps.
    """

    def __init__(self, max_len: int = 45, max_missing_frames: int = 6):
        """
        Args:
            max_len: Maximum number of frames to store (e.g., 45 frames ~1.5s at 30 FPS).
            max_missing_frames: Number of consecutive missing hand detections before
                                resetting motion trajectory.
        """
        if max_len < 5:
            raise ValueError(f"max_len must be at least 5 frames, got {max_len}")
        self.max_len = max_len
        self.max_missing_frames = max_missing_frames
        self._buffer: deque[FrameRecord] = deque(maxlen=max_len)
        self._consecutive_missing: int = 0
        self._last_timestamp: Optional[float] = None

    def push(
        self,
        timestamp: float,
        landmarks: Optional[Any],
        is_valid: bool,
        handedness: str = "Right",
        features_63: Optional[np.ndarray] = None
    ) -> bool:
        """
        Appends a frame observation to the rolling buffer.

        Args:
            timestamp: Monotonic timestamp in seconds.
            landmarks: Raw landmarks (list of 21 dicts or numpy array of shape (21, 3)).
            is_valid: Whether landmark detection and normalization succeeded.
            handedness: 'Right' or 'Left'.
            features_63: Optional 63-D normalized feature vector.

        Returns:
            bool: True if accepted, False if rejected (e.g., decreasing timestamp).
        """
        if not math.isfinite(timestamp):
            return False

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            # Monotonic violation; ignore backwards timestamp
            return False

        # Parse landmarks to (21, 3) float32 array
        lm_arr: Optional[np.ndarray] = None
        if is_valid and landmarks is not None:
            try:
                if isinstance(landmarks, np.ndarray):
                    if landmarks.shape == (21, 3) and np.isfinite(landmarks).all():
                        lm_arr = landmarks.astype(np.float32)
                elif isinstance(landmarks, list) and len(landmarks) == 21:
                    lm_arr = np.array(
                        [[lm['x'], lm['y'], lm.get('z', 0.0)] for lm in landmarks],
                        dtype=np.float32
                    )
                    if not np.isfinite(lm_arr).all():
                        lm_arr = None
            except (KeyError, TypeError, ValueError):
                lm_arr = None

        if lm_arr is None or not is_valid:
            self._consecutive_missing += 1
            record = FrameRecord(
                timestamp=timestamp,
                landmarks=None,
                is_valid=False,
                handedness=handedness,
                features_63=None
            )
        else:
            self._consecutive_missing = 0
            record = FrameRecord(
                timestamp=timestamp,
                landmarks=lm_arr,
                is_valid=True,
                handedness=handedness.strip().title(),
                features_63=features_63
            )

        self._buffer.append(record)
        self._last_timestamp = timestamp

        # Reset buffer if hand has been missing for too long
        if self._consecutive_missing >= self.max_missing_frames:
            # Keep buffer bounded to avoid stale trajectory stitching across long gaps
            self._prune_stale_history()

        return True

    def _prune_stale_history(self) -> None:
        """Clears old frames if hand was lost for extended duration."""
        # Only keep the last few non-valid frames to preserve missing count
        if len(self._buffer) > self.max_missing_frames:
            while len(self._buffer) > self.max_missing_frames:
                self._buffer.popleft()

    def clear(self) -> None:
        """Completely flushes the sequence buffer."""
        self._buffer.clear()
        self._consecutive_missing = 0
        self._last_timestamp = None

    def __len__(self) -> int:
        return len(self._buffer)

    def get_valid_sequence(self) -> List[FrameRecord]:
        """Returns contiguous recent sequence of valid frames from the buffer."""
        out = []
        for rec in reversed(self._buffer):
            if not rec.is_valid or rec.landmarks is None:
                break
            out.append(rec)
        out.reverse()
        return out

    def get_all_records(self) -> List[FrameRecord]:
        """Returns all records currently in the buffer."""
        return list(self._buffer)

    @property
    def consecutive_missing(self) -> int:
        return self._consecutive_missing


class KinematicFeatureExtractor:
    """
    Extracts geometric kinematics and motion trajectory descriptors from a sequence of frames.
    """

    @staticmethod
    def extract_landmark_trajectory(
        records: List[FrameRecord],
        landmark_idx: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts positions and timestamps for a specific landmark index.

        Args:
            records: List of FrameRecord objects.
            landmark_idx: 0 to 20 (e.g., 8 for index finger tip, 20 for pinky tip).

        Returns:
            Tuple of (positions: np.ndarray of shape (K, 3), timestamps: np.ndarray of shape (K,))
        """
        valid_pos = []
        valid_times = []
        for r in records:
            if r.is_valid and r.landmarks is not None and r.landmarks.shape == (21, 3):
                valid_pos.append(r.landmarks[landmark_idx])
                valid_times.append(r.timestamp)

        if not valid_pos:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return np.array(valid_pos, dtype=np.float32), np.array(valid_times, dtype=np.float32)

    @staticmethod
    def compute_motion_metrics(
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes path length, net displacement, tortuosity, bounding spans, and mean speed.

        Args:
            positions: Shape (K, 3).
            timestamps: Shape (K,).

        Returns:
            Dictionary of kinematic metrics.
        """
        K = len(positions)
        if K < 2:
            return {
                "count": float(K),
                "duration": 0.0,
                "path_length": 0.0,
                "net_displacement": 0.0,
                "tortuosity": 1.0,
                "span_x": 0.0,
                "span_y": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0
            }

        duration = max(1e-6, float(timestamps[-1] - timestamps[0]))
        deltas = np.diff(positions, axis=0)  # Shape (K-1, 3)
        dt = np.diff(timestamps)
        dt[dt <= 0] = 1e-4

        segment_dists = np.linalg.norm(deltas[:, :2], axis=1)  # 2D projection on image plane
        path_length = float(np.sum(segment_dists))
        net_disp = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))

        speeds = segment_dists / dt
        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))

        span_x = float(np.ptp(positions[:, 0]))
        span_y = float(np.ptp(positions[:, 1]))

        # Tortuosity = path_length / (net_displacement + eps)
        # Straight line: tortuosity ~ 1.0. Complex curved or zig-zag: tortuosity >> 1.0
        tortuosity = float(path_length / max(net_disp, 1e-5))

        return {
            "count": float(K),
            "duration": duration,
            "path_length": path_length,
            "net_displacement": net_disp,
            "tortuosity": tortuosity,
            "span_x": span_x,
            "span_y": span_y,
            "mean_speed": mean_speed,
            "max_speed": max_speed
        }

    @staticmethod
    def check_handshape_posture(
        landmarks: np.ndarray,
        target_finger: str = "pinky"
    ) -> Tuple[bool, float, str]:
        """
        Geometric posture verification relative to the palm.
        Ensures the hand is in the correct prerequisite handshape for the dynamic sign:
        - For 'J': Hand starts/remains in 'I' shape (pinky extended, index/middle/ring curled).
        - For 'Z': Hand starts/remains in 'D'/'1' shape (index extended, middle/ring/pinky curled).

        Args:
            landmarks: Shape (21, 3) coordinates.
            target_finger: "pinky" (for J) or "index" (for Z).

        Returns:
            Tuple of (is_handshape_match: bool, score: float, explanation: str).
        """
        if landmarks.shape != (21, 3):
            return False, 0.0, "Invalid landmark dimensions"

        wrist = landmarks[0]

        # Finger tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        # Finger MCPs: Index=5, Middle=9, Ring=13, Pinky=17
        dist_index = np.linalg.norm(landmarks[8] - wrist)
        dist_middle = np.linalg.norm(landmarks[12] - wrist)
        dist_ring = np.linalg.norm(landmarks[16] - wrist)
        dist_pinky = np.linalg.norm(landmarks[20] - wrist)

        mcp_index = np.linalg.norm(landmarks[5] - wrist)
        mcp_middle = np.linalg.norm(landmarks[9] - wrist)
        mcp_ring = np.linalg.norm(landmarks[13] - wrist)
        mcp_pinky = np.linalg.norm(landmarks[17] - wrist)

        # A finger is considered extended if its tip is significantly farther from wrist than its MCP
        index_extended = dist_index > (mcp_index * 1.15)
        middle_extended = dist_middle > (mcp_middle * 1.15)
        ring_extended = dist_ring > (mcp_ring * 1.15)
        pinky_extended = dist_pinky > (mcp_pinky * 1.15)

        if target_finger == "pinky":
            # For J: pinky must be extended, middle and ring should be curled
            # Index can be partially curled or resting
            if pinky_extended and not middle_extended and not ring_extended:
                score = (dist_pinky / max(dist_middle, 1e-4))
                return True, min(1.0, float(score * 0.5)), "Pinky extended, others curled (I handshape)"
            return False, 0.2, f"Expected pinky extended: pinky={pinky_extended}, middle={middle_extended}"

        elif target_finger == "index":
            # For Z: index must be extended, middle, ring, and pinky should be curled
            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                score = (dist_index / max(dist_middle, 1e-4))
                return True, min(1.0, float(score * 0.5)), "Index extended, others curled (1/D handshape)"
            return False, 0.2, f"Expected index extended: index={index_extended}, others curled"

        return False, 0.0, f"Unknown target finger: {target_finger}"


class DynamicTrajectoryClassifier:
    """
    Explainable, rule-assisted dynamic gesture classifier for ASL letters 'J' and 'Z'.
    Operates on rolling temporal windows, enforces stroke kinematics, handshape preconditions,
    and a temporal cooldown latch to eliminate duplicate triggers.
    """

    def __init__(
        self,
        min_motion_span: float = 0.07,    # Minimum coordinate travel span (screen ratio)
        min_path_length: float = 0.13,    # Minimum cumulative path length
        min_duration: float = 0.20,       # Minimum gesture duration (seconds)
        max_duration: float = 1.80,       # Maximum gesture duration (seconds)
        cooldown_sec: float = 1.20        # Debounce cooldown period after detection
    ):
        self.min_motion_span = min_motion_span
        self.min_path_length = min_path_length
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.cooldown_sec = cooldown_sec

        self._last_trigger_time: Dict[str, float] = {
            "J": -999.0,
            "Z": -999.0
        }

    def reset(self) -> None:
        """Resets the classifier's internal cooldown timers."""
        self._last_trigger_time["J"] = -999.0
        self._last_trigger_time["Z"] = -999.0

    def evaluate_sequence(
        self,
        buffer: TemporalSequenceBuffer,
        current_time: float
    ) -> DynamicRecognitionResult:
        """
        Evaluates the current temporal window for ASL 'J' or 'Z' dynamic trajectories.

        Args:
            buffer: TemporalSequenceBuffer instance.
            current_time: Current monotonic time in seconds.

        Returns:
            DynamicRecognitionResult with classification outcome and explainability details.
        """
        records = buffer.get_valid_sequence()
        if len(records) < 8:
            return DynamicRecognitionResult(
                detected=False,
                gesture=None,
                confidence=0.0,
                start_time=current_time,
                end_time=current_time,
                stroke_count=0,
                reason="Insufficient valid frames in window (< 8)"
            )

        # 1. Test for 'J' Trajectory (Pinky Tip: Index 20)
        j_result = self._evaluate_j_trajectory(records, current_time)
        if j_result.detected:
            return j_result

        # 2. Test for 'Z' Trajectory (Index Tip: Index 8)
        z_result = self._evaluate_z_trajectory(records, current_time)
        if z_result.detected:
            return z_result

        # If a candidate gesture was rejected specifically because of cooldown, surface it
        if "cooldown" in j_result.reason:
            return j_result
        if "cooldown" in z_result.reason:
            return z_result

        return DynamicRecognitionResult(
            detected=False,
            gesture=None,
            confidence=0.0,
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            stroke_count=0,
            reason="No dynamic gesture pattern matched"
        )

    def _evaluate_j_trajectory(
        self,
        records: List[FrameRecord],
        current_time: float
    ) -> DynamicRecognitionResult:
        """
        Recognizes 'J' gesture:
        - Handshape: Pinky extended, other fingers curled.
        - Motion: Downward vertical stroke followed by a curved upward/inward scoop (J-hook).
        """
        # Cooldown check
        if (current_time - self._last_trigger_time["J"]) < self.cooldown_sec:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=current_time, end_time=current_time, stroke_count=0,
                reason="J in cooldown"
            )

        # Handshape check on recent frames
        handshape_ok, hs_score, hs_reason = KinematicFeatureExtractor.check_handshape_posture(
            records[-1].landmarks, target_finger="pinky"
        )
        if not handshape_ok:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=records[0].timestamp, end_time=records[-1].timestamp, stroke_count=0,
                reason=f"J handshape mismatch: {hs_reason}"
            )

        pos, times = KinematicFeatureExtractor.extract_landmark_trajectory(records, landmark_idx=20)
        metrics = KinematicFeatureExtractor.compute_motion_metrics(pos, times)

        duration = metrics["duration"]
        if not (self.min_duration <= duration <= self.max_duration):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Duration {duration:.2f}s outside bounds [{self.min_duration}, {self.max_duration}]"
            )

        if metrics["path_length"] < self.min_path_length or metrics["span_y"] < self.min_motion_span:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Motion magnitude too small (path={metrics['path_length']:.3f}, span_y={metrics['span_y']:.3f})"
            )

        # Trajectory geometric analysis for J:
        # A 'J' moves downward in Y (Y increases in image coords), reaches a minimum elevation (max Y),
        # then curls back up (Y decreases) with a lateral hook in X.
        y_coords = pos[:, 1]
        x_coords = pos[:, 0]

        # Find the turning point in Y (the bottom of the hook = max Y coordinate in camera frame)
        bottom_idx = int(np.argmax(y_coords))

        # Bottom must occur in the middle-to-late section of the trajectory (between 25% and 88%)
        ratio_bottom = bottom_idx / len(y_coords)
        if not (0.25 <= ratio_bottom <= 0.88):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"J hook turning point index ratio {ratio_bottom:.2f} outside [0.25, 0.88]"
            )

        # Measure downward descent before turn, and upward ascent after turn
        downward_stroke = y_coords[bottom_idx] - y_coords[0]
        upward_stroke = y_coords[bottom_idx] - y_coords[-1]

        # Lateral hook span in X during or near the turn
        hook_x_span = np.ptp(x_coords[bottom_idx:])

        if downward_stroke < (self.min_motion_span * 0.65):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Downward stroke too shallow ({downward_stroke:.3f})"
            )

        if upward_stroke < (self.min_motion_span * 0.20):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Upward hook stroke too shallow ({upward_stroke:.3f})"
            )

        # Qualified J gesture
        confidence = min(0.98, max(0.70, float(0.65 + hs_score * 0.2 + (upward_stroke / max(downward_stroke, 1e-4)) * 0.15)))
        self._last_trigger_time["J"] = current_time

        return DynamicRecognitionResult(
            detected=True,
            gesture="J",
            confidence=confidence,
            start_time=float(times[0]),
            end_time=float(times[-1]),
            stroke_count=2,
            reason=f"Qualified J trajectory (down={downward_stroke:.2f}, up={upward_stroke:.2f}, hook_x={hook_x_span:.2f})",
            metadata={"metrics": metrics, "bottom_ratio": ratio_bottom}
        )

    def _evaluate_z_trajectory(
        self,
        records: List[FrameRecord],
        current_time: float
    ) -> DynamicRecognitionResult:
        """
        Recognizes 'Z' gesture:
        - Handshape: Index extended, middle/ring/pinky curled (1/D shape).
        - Motion: 3-stroke zig-zag:
          Stroke 1: Horizontal line left-to-right (or right-to-left depending on perspective).
          Stroke 2: Diagonal line downward and backward.
          Stroke 3: Horizontal line in the initial direction.
        """
        # Cooldown check
        if (current_time - self._last_trigger_time["Z"]) < self.cooldown_sec:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=current_time, end_time=current_time, stroke_count=0,
                reason="Z in cooldown"
            )

        # Handshape check on recent frames
        handshape_ok, hs_score, hs_reason = KinematicFeatureExtractor.check_handshape_posture(
            records[-1].landmarks, target_finger="index"
        )
        if not handshape_ok:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=records[0].timestamp, end_time=records[-1].timestamp, stroke_count=0,
                reason=f"Z handshape mismatch: {hs_reason}"
            )

        pos, times = KinematicFeatureExtractor.extract_landmark_trajectory(records, landmark_idx=8)
        metrics = KinematicFeatureExtractor.compute_motion_metrics(pos, times)

        duration = metrics["duration"]
        if not (self.min_duration <= duration <= self.max_duration):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Duration {duration:.2f}s outside bounds [{self.min_duration}, {self.max_duration}]"
            )

        if metrics["path_length"] < (self.min_path_length * 1.2) or metrics["span_x"] < self.min_motion_span:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Z motion too small (path={metrics['path_length']:.3f}, span_x={metrics['span_x']:.3f})"
            )

        # Tortuosity check: a zig-zag must have higher tortuosity than a straight line (path length > net displacement)
        if metrics["tortuosity"] < 1.35:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=0,
                reason=f"Z tortuosity too low ({metrics['tortuosity']:.2f} < 1.35, motion is too straight)"
            )

        # Count inflection points in X velocity (directional reversals)
        x_coords = pos[:, 0]
        # Smooth coordinates with 3-point moving average to suppress tracking jitter
        if len(x_coords) >= 5:
            x_smooth = np.convolve(x_coords, np.ones(3)/3.0, mode='valid')
        else:
            x_smooth = x_coords

        dx = np.diff(x_smooth)
        # Filter micro-noise
        significant_thresh = max(1e-4, float(np.std(dx) * 0.25))
        dx_filtered = np.where(np.abs(dx) > significant_thresh, dx, 0.0)

        # Detect sign changes in dx
        signs = np.sign(dx_filtered)
        # Remove consecutive duplicate signs and zeros
        condensed_signs = []
        for s in signs:
            if s != 0 and (not condensed_signs or s != condensed_signs[-1]):
                condensed_signs.append(s)

        inflection_count = max(0, len(condensed_signs) - 1)

        # A classic 'Z' stroke has 2 turning points in X (Stroke 1 -> Stroke 2 reversal -> Stroke 3 reversal)
        # We allow 2 to 5 inflections to tolerate minor tremor
        if inflection_count < 2 or inflection_count > 6:
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=inflection_count,
                reason=f"Z inflection count {inflection_count} outside expected range [2, 6]"
            )

        # Net vertical progression: overall Y should trend downwards during the zig-zag
        y_progression = pos[-1, 1] - pos[0, 1]  # positive Y = downward in image space
        if y_progression < (self.min_motion_span * 0.25):
            return DynamicRecognitionResult(
                detected=False, gesture=None, confidence=0.0,
                start_time=times[0], end_time=times[-1], stroke_count=inflection_count,
                reason=f"Insufficient net downward progression for Z ({y_progression:.3f})"
            )

        # Qualified Z gesture
        confidence = min(0.98, max(0.72, float(0.70 + hs_score * 0.15 + min(0.15, (metrics['tortuosity'] - 1.35) * 0.1))))
        self._last_trigger_time["Z"] = current_time

        return DynamicRecognitionResult(
            detected=True,
            gesture="Z",
            confidence=confidence,
            start_time=float(times[0]),
            end_time=float(times[-1]),
            stroke_count=inflection_count + 1,
            reason=f"Qualified Z zig-zag trajectory (inflections={inflection_count}, tortuosity={metrics['tortuosity']:.2f})",
            metadata={"metrics": metrics, "inflections": inflection_count}
        )
