// API and WebSocket type definitions — matches the actual FastAPI backend schemas exactly.

// ─── HTTP Response Types ───────────────────────────────────────────────────────

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  timestamp_utc: number;
  model_loaded: boolean;
  detector_loaded: boolean;
  active_sessions: number;
}

export interface SystemInfoResponse {
  app_name: string;
  version: string;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  device: string;
  model_checkpoint: string;
  model_param_count: number;
  mediapipe_version: string;
  opencv_version: string;
  input_dim: number;
  num_classes: number;
  classes: string[];
  min_confidence: number;
  dwell_duration: number;
  release_duration: number;
}

// ─── WebSocket Client Messages ─────────────────────────────────────────────────

export interface ClientFrameMetadata {
  type: 'frame';
  sequence_id: number;
  capture_timestamp_ms: number;
}

export interface ClientControlMessage {
  type: 'init' | 'space' | 'delete' | 'clear' | 'speech_pref' | 'shutdown';
  speech_enabled?: boolean;
}

// ─── WebSocket Server Responses ────────────────────────────────────────────────

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface PredictionItem {
  label: string;
  confidence: number;
}

export interface RecognitionResponse {
  type: 'recognition_result';
  sequence_id: number;
  server_receive_timestamp_ms: number;
  processing_completion_timestamp_ms: number;
  processing_duration_ms: number;

  // Perception
  hand_detected: boolean;
  handedness: string | null;
  landmark_valid: boolean;
  landmarks: LandmarkPoint[] | null;

  // Inference
  raw_prediction: string | null;
  raw_confidence: number;
  smoothed_prediction: string | null;
  smoothed_confidence: number;
  confidence_gate_passed: boolean;
  top_predictions?: PredictionItem[];

  // Dynamic & fallback
  static_fallback_label: string | null;
  dynamic_candidate: string | null;
  confirmed_dynamic_result: string | null;

  // FSM state
  fsm_candidate: string | null;
  fsm_is_dwelling: boolean;
  fsm_dwell_time: number;
  dwell_progress: number;
  committed_label: string | null;

  // Text & speech
  text_buffer: string;
  speech_token: string | null;
  error_status: string | null;
}

export interface ControlAckResponse {
  type: 'control_ack';
  action: string;
  status: 'ok' | 'error';
  text_buffer: string;
  speech_token: string | null;
  speech_enabled: boolean | null;
}

export interface ErrorResponse {
  type: 'error';
  sequence_id: number | null;
  message: string;
  code: string;
}

export type WebSocketServerMessage = RecognitionResponse | ControlAckResponse | ErrorResponse;

// ─── UI State Types ────────────────────────────────────────────────────────────

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
export type CameraStatus = 'idle' | 'requesting' | 'active' | 'error' | 'stopped';

export interface RecognitionState {
  lastResponse: RecognitionResponse | null;
  textBuffer: string;
  connectionStatus: ConnectionStatus;
  speechEnabled: boolean;
}
