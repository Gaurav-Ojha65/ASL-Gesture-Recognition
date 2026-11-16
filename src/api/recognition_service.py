"""
Recognition Service for FastAPI Backend.
Executes perception and ASL recognition on binary JPEG frames.
Decoupled from OpenCV GUI display and camera loops; fully thread-safe.
"""

import time
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

from src.api.schemas import RecognitionResponse, PredictionItem
from src.api.model_service import SharedModelService
from src.api.session_manager import UserSessionContext
from src.landmarks.labels import get_semantic_label
from src.realtime.gesture_state_machine import CommitEvent


MAX_JPEG_BYTES: int = 2 * 1024 * 1024       # 2 MB maximum payload limit
MAX_FRAME_WIDTH: int = 1280                 # Maximum decoded frame width
MAX_FRAME_HEIGHT: int = 720                 # Maximum decoded frame height


class RecognitionService:
    """
    Stateless processing service executing the 10-stage ASL recognition pipeline
    for a given binary frame payload and session context.
    """

    @staticmethod
    def process_frame(
        raw_bytes: bytes,
        session: UserSessionContext,
        model_service: SharedModelService,
        sequence_id: int,
        server_receive_time_ms: int,
        mirror_feed: bool = True,
    ) -> RecognitionResponse:
        """
        Processes a single binary JPEG frame in a worker thread.

        Args:
            raw_bytes: Encoded JPEG bytes from the client.
            session: Active UserSessionContext for the connection.
            model_service: SharedModelService providing model and detector.
            sequence_id: Client sequence identifier.
            server_receive_time_ms: Timestamp when frame arrived.
            mirror_feed: Whether to horizontally mirror frame for natural user interaction.

        Returns:
            RecognitionResponse populated with perception, inference, and FSM telemetry.

        Raises:
            ValueError: If payload exceeds size limits, cannot be decoded, or exceeds dimension caps.
        """
        t0 = time.time()
        now_sec = t0

        # 1. Payload validation
        if not raw_bytes or len(raw_bytes) == 0:
            raise ValueError("Empty frame payload received.")
        if len(raw_bytes) > MAX_JPEG_BYTES:
            raise ValueError(f"Payload size ({len(raw_bytes)} bytes) exceeds maximum limit ({MAX_JPEG_BYTES} bytes).")

        # 2. JPEG Decoding
        arr = np.frombuffer(raw_bytes, np.uint8)
        bgr_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Corrupt or invalid JPEG image data.")

        # 3. Dimension caps validation
        h, w = bgr_frame.shape[:2]
        if w > MAX_FRAME_WIDTH or h > MAX_FRAME_HEIGHT:
            raise ValueError(f"Frame dimensions ({w}x{h}) exceed maximum allowed cap ({MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}).")

        # 4. Mirror feed if enabled
        if mirror_feed:
            bgr_frame = cv2.flip(bgr_frame, 1)

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # 5. MediaPipe Hand Landmark Detection (serialized through SharedModelService lock)
        detection_result = model_service.detect_hand(rgb_frame)
        has_hand = bool(detection_result and detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0)

        # Telemetry variables
        landmarks_dict: Optional[List[Dict[str, float]]] = None
        handedness_cat: Optional[str] = None
        landmark_valid: bool = False
        raw_pred_label: Optional[str] = None
        raw_conf: float = 0.0
        smoothed_label: Optional[str] = None
        smoothed_conf: float = 0.0
        conf_gate_passed: bool = False
        static_fallback_lbl: Optional[str] = None
        dyn_candidate: Optional[str] = None
        confirmed_dynamic: Optional[str] = None
        commit_event: Optional[CommitEvent] = None
        speech_token: Optional[str] = None
        error_msg: Optional[str] = None
        top_preds: List[PredictionItem] = []

        if not has_hand:
            # Case A: No hand detected (Genuine absence)
            session.reset_smoothing()
            session.seq_buffer.push(timestamp=now_sec, landmarks=None, is_valid=False)

            safe_label = session.state_machine.latched_label or session.state_machine.current_candidate or 'A'
            commit_event = session.state_machine.process_observation(
                label=safe_label,
                confidence=0.0,
                hand_detected=False,
                timestamp=now_sec
            )
        else:
            # Case B: Hand detected
            raw_landmarks = detection_result.hand_landmarks[0]
            landmarks_dict = [{'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)} for lm in raw_landmarks]
            if detection_result.handedness and len(detection_result.handedness) > 0:
                handedness_cat = detection_result.handedness[0][0].category_name

            # 6. Normalize landmarks into 63-D feature representation
            is_valid, features, norm_reason = session.normalizer.normalize(landmarks_dict, handedness_cat)
            landmark_valid = is_valid

            if not is_valid:
                session.reset_smoothing()
                session.seq_buffer.push(timestamp=now_sec, landmarks=None, is_valid=False)
                safe_label = session.state_machine.latched_label or session.state_machine.current_candidate or 'A'
                commit_event = session.state_machine.process_observation(
                    label=safe_label,
                    confidence=0.0,
                    hand_detected=True,
                    timestamp=now_sec
                )
                error_msg = f"INVALID_FEATURES: {norm_reason}"
            else:
                # 7. Stateless PyTorch Model Inference
                raw_probs, logits, top1_idx, top1_label, top1_conf = model_service.predict_stateless(features)
                raw_pred_label = top1_label
                raw_conf = top1_conf

                # 8. Session-isolated 3-frame moving-average probability smoothing
                session.prob_history.append(raw_probs)
                smoothed_probs = np.mean(session.prob_history, axis=0)
                prob_sum = smoothed_probs.sum()
                if prob_sum > 0:
                    smoothed_probs = smoothed_probs / prob_sum

                sm_idx = int(np.argmax(smoothed_probs))
                smoothed_conf = float(smoothed_probs[sm_idx])
                smoothed_label = get_semantic_label(sm_idx)
                conf_gate_passed = bool(smoothed_conf >= session.sm_config.min_confidence)

                # Extract Top-3 predicted classes with confidence
                top3_indices = np.argsort(smoothed_probs)[::-1][:3]
                top_preds = [
                    PredictionItem(
                        label=get_semantic_label(int(idx)),
                        confidence=round(float(smoothed_probs[idx]), 4)
                    )
                    for idx in top3_indices
                ]

                # 9. Static Handshape Fallback Arbitration for dynamic J and Z
                static_fallback_lbl = smoothed_label
                fsm_label = smoothed_label
                if smoothed_label in ('J', 'Z'):
                    fsm_label = 'I' if smoothed_label == 'J' else 'D'

                # 10. Temporal Gesture State Machine (Hysteresis Dwell Confirmation)
                commit_event = session.state_machine.process_observation(
                    label=fsm_label,
                    confidence=smoothed_conf,
                    hand_detected=True,
                    timestamp=now_sec
                )

                # 11. Dynamic Trajectory Evaluation (Kinematic tracking for J and Z)
                session.seq_buffer.push(
                    timestamp=now_sec,
                    landmarks=landmarks_dict,
                    is_valid=True,
                    handedness=handedness_cat or "Right",
                    features_63=features
                )
                dyn_res = session.dynamic_classifier.evaluate_sequence(
                    session.seq_buffer,
                    current_time=now_sec
                )
                if dyn_res.detected and dyn_res.gesture is not None:
                    confirmed_dynamic = dyn_res.gesture
                    commit_event = CommitEvent(
                        label=dyn_res.gesture,
                        timestamp=now_sec,
                        confidence=dyn_res.confidence,
                        dwell_time=0.0,
                        metadata={"dynamic": True, "reason": dyn_res.reason}
                    )
                    session.state_machine.reset()
                elif dyn_res.gesture is not None:
                    dyn_candidate = dyn_res.gesture

        # 12. TextBuffer Commitment
        if commit_event is not None and commit_event.label:
            session.text_buffer.apply(commit_event)

            # Speech token arbitration
            if session.speech_enabled:
                if commit_event.label == "SPACE":
                    words = session.text_buffer.get_text().strip().split()
                    if words:
                        speech_token = words[-1]
                        session.last_spoken_word = speech_token
                elif commit_event.label != "DELETE":
                    speech_token = commit_event.label
                    session.last_speech_timestamp = now_sec

        # Conversational Pause Completion Speech (2.5s idle on completed word)
        elif session.speech_enabled and session.text_buffer.get_text().strip():
            words = session.text_buffer.get_text().strip().split()
            if words:
                candidate_word = words[-1]
                if candidate_word != session.last_spoken_word:
                    if session.last_speech_timestamp > 0 and (now_sec - session.last_speech_timestamp) >= 2.5:
                        speech_token = candidate_word
                        session.last_spoken_word = candidate_word

        t_end = time.time()
        completion_ms = int(t_end * 1000)
        proc_duration_ms = (t_end - t0) * 1000

        fsm_candidate = session.state_machine.current_candidate if has_hand else None
        fsm_dwelling = session.state_machine.is_dwelling if has_hand else False
        fsm_dwell_time = (now_sec - session.state_machine.candidate_first_seen) if (session.state_machine.candidate_first_seen and has_hand) else 0.0
        dwell_progress = min(1.0, fsm_dwell_time / session.sm_config.dwell_duration) if (fsm_dwelling and has_hand) else 0.0

        return RecognitionResponse(
            type="recognition_result",
            sequence_id=sequence_id,
            server_receive_timestamp_ms=server_receive_time_ms,
            processing_completion_timestamp_ms=completion_ms,
            processing_duration_ms=round(proc_duration_ms, 2),
            hand_detected=has_hand,
            handedness=handedness_cat,
            landmark_valid=landmark_valid,
            landmarks=landmarks_dict,
            raw_prediction=raw_pred_label,
            raw_confidence=round(raw_conf, 4),
            smoothed_prediction=smoothed_label,
            smoothed_confidence=round(smoothed_conf, 4),
            confidence_gate_passed=conf_gate_passed,
            top_predictions=top_preds,
            static_fallback_label=static_fallback_lbl,
            dynamic_candidate=dyn_candidate,
            confirmed_dynamic_result=confirmed_dynamic,
            fsm_candidate=fsm_candidate,
            fsm_is_dwelling=fsm_dwelling,
            fsm_dwell_time=round(fsm_dwell_time, 2),
            dwell_progress=round(dwell_progress, 4),
            committed_label=commit_event.label if commit_event else None,
            text_buffer=session.text_buffer.get_text(),
            speech_token=speech_token,
            error_status=error_msg,
        )
