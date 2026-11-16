import { useSearchParams } from 'react-router-dom';
import { RecognitionPanel } from '@/components/recognition/RecognitionPanel';
import type { RecognitionResponse, SystemInfoResponse } from '@/types/api';

const SYSTEM_INFO: SystemInfoResponse = {
  app_name: 'ASL Gesture Studio API',
  version: '1.0.0',
  python_version: '3.12.10',
  torch_version: '2.14.0',
  cuda_available: true,
  device: 'cuda',
  model_checkpoint: 'models/asl_landmark_mlp_best.pt',
  model_param_count: 18524,
  mediapipe_version: '1.0.1',
  opencv_version: '5.0.0',
  input_dim: 63,
  num_classes: 28,
  classes: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE', 'DELETE'],
  min_confidence: 0.70,
  dwell_duration: 0.60,
  release_duration: 0.25,
};

const BASE_RECOGNITION: RecognitionResponse = {
  type: 'recognition_result',
  sequence_id: 100,
  server_receive_timestamp_ms: 1000,
  processing_completion_timestamp_ms: 1012,
  processing_duration_ms: 12.0,
  hand_detected: false,
  handedness: null,
  landmark_valid: false,
  landmarks: null,
  raw_prediction: null,
  raw_confidence: 0,
  smoothed_prediction: null,
  smoothed_confidence: 0,
  confidence_gate_passed: false,
  static_fallback_label: null,
  dynamic_candidate: null,
  confirmed_dynamic_result: null,
  fsm_candidate: null,
  fsm_is_dwelling: false,
  fsm_dwell_time: 0,
  dwell_progress: 0,
  committed_label: null,
  text_buffer: '',
  speech_token: null,
  error_status: null,
  top_predictions: [],
};

const STATES: Record<string, { recognition: RecognitionResponse | null; title: string }> = {
  idle: {
    recognition: null,
    title: 'State: Idle',
  },
  no_hand: {
    recognition: {
      ...BASE_RECOGNITION,
      hand_detected: false,
      fsm_candidate: null,
      dwell_progress: 0,
    },
    title: 'State: No Hand',
  },
  dwelling: {
    recognition: {
      ...BASE_RECOGNITION,
      hand_detected: true,
      handedness: 'Right',
      fsm_candidate: 'B',
      fsm_is_dwelling: true,
      fsm_dwell_time: 0.42,
      dwell_progress: 0.70,
      smoothed_confidence: 0.92,
      confidence_gate_passed: true,
      top_predictions: [
        { label: 'B', confidence: 0.92 },
        { label: 'W', confidence: 0.05 },
        { label: 'F', confidence: 0.02 },
      ],
    },
    title: 'State: Dwelling (70% progress, Top-3)',
  },
  committed: {
    recognition: {
      ...BASE_RECOGNITION,
      hand_detected: true,
      handedness: 'Right',
      fsm_candidate: 'B',
      committed_label: 'B',
      fsm_is_dwelling: false,
      dwell_progress: 1.0,
      smoothed_confidence: 0.95,
      confidence_gate_passed: true,
      top_predictions: [
        { label: 'B', confidence: 0.95 },
        { label: 'W', confidence: 0.03 },
        { label: 'F', confidence: 0.01 },
      ],
    },
    title: 'State: Committed',
  },
};

export function DevStates() {
  const [params] = useSearchParams();
  const stateKey = params.get('state');

  if (stateKey && STATES[stateKey]) {
    const s = STATES[stateKey];
    return (
      <div className="p-8 max-w-[400px] mx-auto bg-bg min-h-screen">
        <h2 className="text-sm font-bold mb-4 text-muted uppercase tracking-wider">{s.title}</h2>
        <RecognitionPanel
          recognition={s.recognition}
          systemInfo={SYSTEM_INFO}
          connectionStatus="connected"
        />
      </div>
    );
  }

  return (
    <div className="p-8 max-w-[1400px] mx-auto bg-bg min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-ink">Recognition Panel States Gallery</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 items-start">
        {Object.entries(STATES).map(([key, s]) => (
          <div key={key} className="flex flex-col gap-2">
            <h2 className="text-xs font-bold text-muted uppercase tracking-wider">{s.title}</h2>
            <RecognitionPanel
              recognition={s.recognition}
              systemInfo={SYSTEM_INFO}
              connectionStatus="connected"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
