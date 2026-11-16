"""
Pydantic Schemas for FastAPI ASL Gesture Studio.
Defines contracts for HTTP endpoints and WebSocket messaging protocols.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ─── HTTP Health & System Telemetry Schemas ───────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall backend health status")
    timestamp_utc: float = Field(..., description="Server UTC timestamp in seconds")
    model_loaded: bool = Field(..., description="Whether the frozen MLP is loaded")
    detector_loaded: bool = Field(..., description="Whether the MediaPipe detector is loaded")
    active_sessions: int = Field(..., description="Number of currently active WebSocket sessions")


class SystemInfoResponse(BaseModel):
    app_name: str = Field("ASL Gesture Studio API", description="Application name")
    version: str = Field("1.0.0", description="Backend version")
    python_version: str = Field(..., description="Python runtime version")
    torch_version: str = Field(..., description="PyTorch version")
    cuda_available: bool = Field(..., description="Whether CUDA acceleration is enabled")
    device: str = Field(..., description="Active compute device (cuda / cpu)")
    model_checkpoint: str = Field(..., description="Relative path to model checkpoint")
    model_param_count: int = Field(18524, description="Trainable model parameter count")
    mediapipe_version: str = Field(..., description="MediaPipe Tasks API version")
    opencv_version: str = Field(..., description="OpenCV runtime version")
    input_dim: int = Field(63, description="Input landmark feature dimension")
    num_classes: int = Field(28, description="Number of gesture classes")
    classes: List[str] = Field(..., description="List of 28 semantic output classes")
    min_confidence: float = Field(0.70, description="Hysteresis acceptance threshold")
    dwell_duration: float = Field(0.60, description="Dwell confirmation duration in seconds")
    release_duration: float = Field(0.25, description="Release debounce duration in seconds")


# ─── WebSocket Client Messages ────────────────────────────────────────────────

class ClientFrameMetadata(BaseModel):
    """Metadata message sent immediately preceding a binary JPEG frame payload."""
    type: Literal["frame"] = "frame"
    sequence_id: int = Field(..., description="Monotonically increasing client frame sequence ID")
    capture_timestamp_ms: Optional[int] = Field(None, description="Client capture wall-clock timestamp in ms")


class ClientControlMessage(BaseModel):
    """Control commands sent over the WebSocket channel."""
    type: Literal["init", "space", "delete", "clear", "speech_pref", "shutdown"]
    session_id: Optional[str] = Field(None, description="Optional client session identifier")
    speech_enabled: Optional[bool] = Field(None, description="Speech audio synthesis toggle preference")


# ─── WebSocket Server Responses ───────────────────────────────────────────────

class PredictionItem(BaseModel):
    """Candidate prediction with class label and softmax confidence."""
    label: str = Field(..., description="Gesture class label")
    confidence: float = Field(..., description="Softmax confidence probability")


class RecognitionResponse(BaseModel):
    """Structured telemetry returned for every processed webcam frame."""
    type: Literal["recognition_result"] = "recognition_result"
    sequence_id: int = Field(..., description="Echoed client frame sequence ID")
    server_receive_timestamp_ms: int = Field(..., description="Timestamp when server received frame")
    processing_completion_timestamp_ms: int = Field(..., description="Timestamp when processing finished")
    processing_duration_ms: float = Field(..., description="Total frame processing latency in ms")

    # Perception status
    hand_detected: bool = Field(..., description="Whether MediaPipe detected a hand")
    handedness: Optional[str] = Field(None, description="Handedness category ('Right' or 'Left')")
    landmark_valid: bool = Field(..., description="Whether landmarks were normalized successfully")
    landmarks: Optional[List[Dict[str, float]]] = Field(None, description="Normalized 21 landmarks (x, y, z)")

    # Model inference
    raw_prediction: Optional[str] = Field(None, description="Raw Top-1 predicted label before smoothing")
    raw_confidence: float = Field(0.0, description="Raw Top-1 softmax probability")
    smoothed_prediction: Optional[str] = Field(None, description="Smoothed predicted label")
    smoothed_confidence: float = Field(0.0, description="Smoothed softmax probability")
    confidence_gate_passed: bool = Field(False, description="Whether confidence >= 0.70 threshold")
    top_predictions: List[PredictionItem] = Field(default_factory=list, description="Top-3 predictions with label and confidence")

    # Dynamic & fallback recognition
    static_fallback_label: Optional[str] = Field(None, description="Static remapped label (J->I, Z->D)")
    dynamic_candidate: Optional[str] = Field(None, description="Dynamic gesture trajectory candidate")
    confirmed_dynamic_result: Optional[str] = Field(None, description="Confirmed dynamic gesture commit")

    # Temporal state machine (FSM)
    fsm_candidate: Optional[str] = Field(None, description="Current FSM dwell candidate")
    fsm_is_dwelling: bool = Field(False, description="Whether candidate is currently dwelling")
    fsm_dwell_time: float = Field(0.0, description="Elapsed dwell time in seconds")
    dwell_progress: float = Field(0.0, description="Normalized dwell progress from 0.0 to 1.0")
    committed_label: Optional[str] = Field(None, description="Label committed on this exact frame")

    # Text & speech
    text_buffer: str = Field("", description="Current accumulated text buffer")
    speech_token: Optional[str] = Field(None, description="Speech token emitted on this frame")
    error_status: Optional[str] = Field(None, description="Error or diagnostic status message")


class ControlAckResponse(BaseModel):
    """Acknowledgment returned for control messages."""
    type: Literal["control_ack"] = "control_ack"
    action: str = Field(..., description="Action that was executed")
    status: str = Field("ok", description="Execution status ('ok' or 'error')")
    text_buffer: str = Field("", description="Current text buffer after action")
    speech_token: Optional[str] = Field(None, description="Spoken token if generated by action")
    speech_enabled: Optional[bool] = Field(None, description="Current speech synthesis setting")


class ErrorResponse(BaseModel):
    """Error message returned when a request fails or violates constraints."""
    type: Literal["error"] = "error"
    sequence_id: Optional[int] = Field(None, description="Sequence ID that triggered error")
    message: str = Field(..., description="Human-readable error description")
    code: str = Field(..., description="Machine-readable error code")

