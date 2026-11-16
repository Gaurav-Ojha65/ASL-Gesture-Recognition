import { useEffect, useState, useRef } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { fetchHealth, fetchSystemInfo } from '@/services/api';
import type { HealthResponse, SystemInfoResponse } from '@/types/api';
import {
  Activity,
  Cpu,
  Wifi,
  ChevronDown,
  ChevronUp,
  Clock,
  Layers,
  Sliders,
} from 'lucide-react';

export function Diagnostics() {
  const ws = useWebSocket();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);

  // FPS counter
  const [observedFps, setObservedFps] = useState(0);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Connect WebSocket on mount for live telemetry inspection
  useEffect(() => {
    ws.connect();
    return () => {
      ws.disconnect();
    };
  }, []);

  useEffect(() => {
    if (ws.connectionStatus === 'connected') {
      ws.sendControl({ type: 'init', speech_enabled: false });
    }
  }, [ws.connectionStatus]);

  // Track FPS based on incoming recognition results
  useEffect(() => {
    if (ws.lastRecognition) {
      frameCountRef.current++;
    }
  }, [ws.lastRecognition]);

  useEffect(() => {
    fpsTimerRef.current = setInterval(() => {
      setObservedFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);
    return () => {
      if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
    };
  }, []);

  // Poll health and fetch system info
  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));

    fetchSystemInfo()
      .then((s) => setSystemInfo(s))
      .catch(() => {});

    const interval = setInterval(() => {
      fetchHealth()
        .then((h) => {
          setHealth(h);
          setHealthError(false);
        })
        .catch(() => setHealthError(true));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const rec = ws.lastRecognition;
  const dwellDuration = systemInfo?.dwell_duration ?? 0.6;
  const releaseDuration = systemInfo?.release_duration ?? 0.25;
  const minConfidence = systemInfo?.min_confidence ?? 0.7;

  return (
    <div className="min-h-screen bg-bg text-ink p-5 md:p-10">
      <div className="max-w-[1240px] mx-auto flex flex-col gap-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-ink flex items-center gap-2.5 mb-2">
            <Activity className="text-brand" size={28} />
            Developer tools
          </h1>
          <p className="text-muted text-base">
            Real-time perception pipeline inspection, FSM state telemetry, and latency metrics.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Card 1: Connection & Host Telemetry */}
          <div className="bg-surface border border-line rounded-card p-6 flex flex-col justify-between">
            <div>
              <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
                <Wifi size={18} className="text-brand" />
                WebSocket & API Connectivity
              </h2>
              <div className="space-y-2.5 font-mono text-xs">
                <div className="flex justify-between items-center py-1">
                  <span className="text-muted font-sans font-medium">WS Channel Status:</span>
                  <span
                    className={`font-semibold px-2.5 py-0.5 rounded-pill text-xs ${
                      ws.connectionStatus === 'connected'
                        ? 'bg-brand-soft text-brand'
                        : ws.connectionStatus === 'connecting'
                        ? 'bg-sand text-ink'
                        : 'bg-coral-soft text-coral-text'
                    }`}
                  >
                    {ws.connectionStatus.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between items-center py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Backend Health:</span>
                  <span
                    className={`font-semibold px-2.5 py-0.5 rounded-pill text-xs ${
                      healthError
                        ? 'bg-coral-soft text-coral-text'
                        : health?.status === 'healthy'
                        ? 'bg-brand-soft text-brand'
                        : 'bg-sand text-ink'
                    }`}
                  >
                    {healthError ? 'OFFLINE' : health?.status.toUpperCase() ?? 'CHECKING...'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Model Weights Loaded:</span>
                  <span className="text-ink font-semibold">
                    {health?.model_loaded ? 'true (PyTorch MLP)' : 'false'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Hand Detector Loaded:</span>
                  <span className="text-ink font-semibold">
                    {health?.detector_loaded ? 'true (MediaPipe)' : 'false'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Active WS Sessions:</span>
                  <span className="text-ink font-semibold">{health?.active_sessions ?? 0}</span>
                </div>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-muted font-mono">
              GET /api/health polled every 5s
            </div>
          </div>

          {/* Card 2: Compute Architecture & Thresholds */}
          <div className="bg-surface border border-line rounded-card p-6 flex flex-col justify-between">
            <div>
              <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
                <Cpu size={18} className="text-brand" />
                Runtime Configuration
              </h2>
              <div className="space-y-2.5 font-mono text-xs">
                <div className="flex justify-between py-1">
                  <span className="text-muted font-sans font-medium">Compute Device:</span>
                  <span className="text-ink font-semibold">{systemInfo?.device ?? 'cpu'}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">PyTorch Version:</span>
                  <span className="text-ink">{systemInfo?.torch_version ?? '—'}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">MediaPipe Version:</span>
                  <span className="text-ink">{systemInfo?.mediapipe_version ?? '—'}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Confidence Threshold:</span>
                  <span className="text-brand font-semibold">{minConfidence.toFixed(2)}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Dwell Duration:</span>
                  <span className="text-brand font-semibold">{dwellDuration.toFixed(2)}s</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Release Debounce:</span>
                  <span className="text-brand font-semibold">{releaseDuration.toFixed(2)}s</span>
                </div>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-muted font-mono truncate">
              {systemInfo?.model_checkpoint ?? 'models/asl_landmark_mlp_best.pt'}
            </div>
          </div>

          {/* Card 3: Frame Timing & Latency */}
          <div className="bg-surface border border-line rounded-card p-6 flex flex-col justify-between">
            <div>
              <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
                <Clock size={18} className="text-brand" />
                Frame Latency & Rate
              </h2>
              <div className="space-y-2.5 font-mono text-xs">
                <div className="flex justify-between py-1">
                  <span className="text-muted font-sans font-medium">Sequence ID:</span>
                  <span className="text-ink font-bold">{rec?.sequence_id ?? '—'}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Backend Duration:</span>
                  <span className="text-brand font-semibold">
                    {rec ? `${rec.processing_duration_ms} ms` : '—'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Observed Frame Rate:</span>
                  <span className="text-ink font-bold">{observedFps} FPS</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Server RX Timestamp:</span>
                  <span className="text-ink">{rec?.server_receive_timestamp_ms ?? '—'}</span>
                </div>
                <div className="flex justify-between py-1 border-t border-line">
                  <span className="text-muted font-sans font-medium">Completion Timestamp:</span>
                  <span className="text-ink">{rec?.processing_completion_timestamp_ms ?? '—'}</span>
                </div>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-muted font-mono">
              Target pipeline throughput: 12 FPS cap
            </div>
          </div>

          {/* Card 4: Perception Telemetry */}
          <div className="bg-surface border border-line rounded-card p-6">
            <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
              <Layers size={18} className="text-brand" />
              MediaPipe Perception Pipeline
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between py-1">
                <span className="text-muted font-sans font-medium">Hand Detected:</span>
                <span
                  className={`font-semibold ${
                    rec?.hand_detected ? 'text-brand' : 'text-muted'
                  }`}
                >
                  {rec?.hand_detected ? 'TRUE' : 'FALSE'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Handedness:</span>
                <span className="text-ink">{rec?.handedness ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Landmarks Normalized:</span>
                <span className={rec?.landmark_valid ? 'text-brand font-semibold' : 'text-muted'}>
                  {rec?.landmark_valid ? 'VALID (63-D)' : 'INVALID'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Coordinates Array:</span>
                <span className="text-ink">
                  {rec?.landmarks ? `${rec.landmarks.length} points` : 'null'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Perception Error Status:</span>
                <span className="text-coral-text truncate max-w-[180px]">
                  {rec?.error_status ?? 'none'}
                </span>
              </div>
            </div>
          </div>

          {/* Card 5: Inference & Probability Metrics */}
          <div className="bg-surface border border-line rounded-card p-6">
            <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
              <Sliders size={18} className="text-brand" />
              PyTorch MLP Inference Metrics
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between py-1">
                <span className="text-muted font-sans font-medium">Raw Top-1 Prediction:</span>
                <span className="text-ink font-bold">
                  {rec?.raw_prediction ?? '—'} ({(rec?.raw_confidence ?? 0).toFixed(4)})
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Smoothed Prediction:</span>
                <span className="text-brand font-bold">
                  {rec?.smoothed_prediction ?? '—'} ({(rec?.smoothed_confidence ?? 0).toFixed(4)})
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Confidence Gate Passed:</span>
                <span
                  className={`font-semibold ${
                    rec?.confidence_gate_passed ? 'text-brand' : 'text-coral-text'
                  }`}
                >
                  {rec?.confidence_gate_passed ? 'PASS (>=0.70)' : 'FILTERED'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Static Fallback Label:</span>
                <span className="text-ink">{rec?.static_fallback_label ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Dynamic Trajectory:</span>
                <span className="text-ink">
                  {rec?.confirmed_dynamic_result
                    ? `Confirmed: ${rec.confirmed_dynamic_result}`
                    : rec?.dynamic_candidate
                    ? `Candidate: ${rec.dynamic_candidate}`
                    : 'none'}
                </span>
              </div>
              {/* Top-3 Probability Candidates */}
              <div className="pt-2 border-t border-line">
                <span className="text-muted font-sans font-medium block mb-1.5">Top-3 Candidates:</span>
                {rec?.top_predictions && rec.top_predictions.length > 0 ? (
                  <div className="space-y-1.5">
                    {rec.top_predictions.map((p, idx) => (
                      <div key={p.label || idx} className="flex items-center justify-between">
                        <span className="text-ink font-bold">#{idx + 1} {p.label}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-track rounded-pill overflow-hidden">
                            <div
                              className="h-full bg-brand rounded-pill"
                              style={{ width: `${Math.round(p.confidence * 100)}%` }}
                            />
                          </div>
                          <span className="text-muted w-12 text-right">
                            {(p.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-muted italic">No prediction stream active</span>
                )}
              </div>
            </div>
          </div>

          {/* Card 6: Gesture State Machine (FSM) Hysteresis */}
          <div className="bg-surface border border-line rounded-card p-6">
            <h2 className="text-base font-semibold text-ink flex items-center gap-2 mb-4 pb-2 border-b border-line">
              <Layers size={18} className="text-brand" />
              Temporal State Machine (FSM)
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between py-1">
                <span className="text-muted font-sans font-medium">Current Candidate:</span>
                <span className="text-ink font-bold text-sm">
                  {rec?.fsm_candidate ?? '—'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Is Dwelling:</span>
                <span className={rec?.fsm_is_dwelling ? 'text-brand font-bold' : 'text-muted'}>
                  {rec?.fsm_is_dwelling ? 'TRUE' : 'FALSE'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Elapsed Dwell Time:</span>
                <span className="text-ink font-bold">
                  {rec
                    ? `${(rec.dwell_progress * dwellDuration).toFixed(2)}s / ${dwellDuration.toFixed(2)}s (${Math.round((rec.dwell_progress ?? 0) * 100)}%)`
                    : `0.00s / ${dwellDuration.toFixed(2)}s (0%)`}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Instantaneous Commit Event:</span>
                <span className="text-brand font-bold">
                  {rec?.committed_label ?? 'none'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-line">
                <span className="text-muted font-sans font-medium">Speech Token:</span>
                <span className="text-ink">{rec?.speech_token ?? 'none'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Collapsible Raw JSON Telemetry Inspector */}
        <div className="bg-surface border border-line rounded-card overflow-hidden">
          <button
            onClick={() => setShowRawJson(!showRawJson)}
            className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-track/50 transition-colors"
          >
            <span className="text-sm font-semibold text-ink flex items-center gap-2 font-mono">
              <span className="text-brand">{'{ }'}</span>
              Raw WebSocket JSON Message Inspector
            </span>
            {showRawJson ? <ChevronUp size={18} className="text-muted" /> : <ChevronDown size={18} className="text-muted" />}
          </button>

          {showRawJson && (
            <div className="p-4 border-t border-line bg-track font-mono text-xs text-ink overflow-x-auto max-h-96">
              <pre>
                {rec
                  ? JSON.stringify(rec, null, 2)
                  : '// No frame recognition response received yet. Start camera in Live Translator to stream frames.'}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
