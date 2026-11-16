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

  return (
    <div className="p-4 lg:p-6 max-w-[1600px] mx-auto min-h-screen">
      <div className="mb-6">
        <h1 className="text-xl md:text-2xl font-bold text-slate-100 flex items-center gap-2.5">
          <Activity className="text-violet-500" size={24} />
          System Diagnostics & Live Telemetry
        </h1>
        <p className="text-slate-400 text-xs md:text-sm mt-1">
          Direct inspect of backend perception, state machine hysteresis, and frame latency
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {/* Card 1: Connection & Host Telemetry */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
              <Wifi size={16} className="text-violet-400" />
              WebSocket & API Connectivity
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-400">WS Channel Status:</span>
                <span
                  className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                    ws.connectionStatus === 'connected'
                      ? 'bg-green-500/10 text-green-400 border border-green-500/30'
                      : ws.connectionStatus === 'connecting'
                      ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                      : 'bg-red-500/10 text-red-400 border border-red-500/30'
                  }`}
                >
                  {ws.connectionStatus.toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Backend Health:</span>
                <span className={healthError ? 'text-red-400' : 'text-green-400'}>
                  {healthError ? 'OFFLINE' : health?.status.toUpperCase() ?? 'CHECKING...'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Model Weights Loaded:</span>
                <span className="text-slate-200">
                  {health?.model_loaded ? 'true (PyTorch MLP)' : 'false'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Hand Detector Loaded:</span>
                <span className="text-slate-200">
                  {health?.detector_loaded ? 'true (MediaPipe)' : 'false'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Active WS Sessions:</span>
                <span className="text-slate-200">{health?.active_sessions ?? 0}</span>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 font-mono">
            GET /api/health polled every 5s
          </div>
        </div>

        {/* Card 2: Compute Architecture & Thresholds */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
              <Cpu size={16} className="text-violet-400" />
              Runtime Configuration
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Compute Device:</span>
                <span className="text-slate-100 font-semibold">{systemInfo?.device ?? 'cpu'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">PyTorch Version:</span>
                <span className="text-slate-200">{systemInfo?.torch_version ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">MediaPipe Version:</span>
                <span className="text-slate-200">{systemInfo?.mediapipe_version ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Confidence Threshold:</span>
                <span className="text-cyan-400 font-semibold">{systemInfo?.min_confidence ?? 0.7}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Dwell Duration:</span>
                <span className="text-cyan-400 font-semibold">{systemInfo?.dwell_duration ?? 0.6}s</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Release Debounce:</span>
                <span className="text-cyan-400 font-semibold">{systemInfo?.release_duration ?? 0.25}s</span>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 font-mono">
            {systemInfo?.model_checkpoint ?? 'models/asl_landmark_mlp_best.pt'}
          </div>
        </div>

        {/* Card 3: Frame Timing & Latency */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
              <Clock size={16} className="text-violet-400" />
              Frame Latency & Rate
            </h2>
            <div className="space-y-2.5 font-mono text-xs">
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Sequence ID:</span>
                <span className="text-slate-100 font-bold">{rec?.sequence_id ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Backend Duration:</span>
                <span className="text-cyan-400 font-semibold">
                  {rec ? `${rec.processing_duration_ms} ms` : '—'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Observed Frame Rate:</span>
                <span className="text-slate-100 font-bold">{observedFps} FPS</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Server RX Timestamp:</span>
                <span className="text-slate-300">{rec?.server_receive_timestamp_ms ?? '—'}</span>
              </div>
              <div className="flex justify-between py-1 border-t border-slate-800/60">
                <span className="text-slate-400">Completion Timestamp:</span>
                <span className="text-slate-300">{rec?.processing_completion_timestamp_ms ?? '—'}</span>
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 font-mono">
            Target pipeline throughput: 12 FPS cap
          </div>
        </div>

        {/* Card 4: Perception Telemetry */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
            <Layers size={16} className="text-violet-400" />
            MediaPipe Perception Pipeline
          </h2>
          <div className="space-y-2.5 font-mono text-xs">
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Hand Detected:</span>
              <span
                className={`font-semibold ${
                  rec?.hand_detected ? 'text-green-400' : 'text-slate-500'
                }`}
              >
                {rec?.hand_detected ? 'TRUE' : 'FALSE'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Handedness:</span>
              <span className="text-slate-200">{rec?.handedness ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Landmarks Normalized:</span>
              <span className={rec?.landmark_valid ? 'text-green-400' : 'text-slate-500'}>
                {rec?.landmark_valid ? 'VALID (63-D)' : 'INVALID'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Coordinates Array:</span>
              <span className="text-slate-200">
                {rec?.landmarks ? `${rec.landmarks.length} points` : 'null'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Perception Error Status:</span>
              <span className="text-amber-400 truncate max-w-[180px]">
                {rec?.error_status ?? 'none'}
              </span>
            </div>
          </div>
        </div>

        {/* Card 5: Inference & Probability Metrics */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
            <Sliders size={16} className="text-violet-400" />
            PyTorch MLP Inference Metrics
          </h2>
          <div className="space-y-2.5 font-mono text-xs">
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Raw Top-1 Prediction:</span>
              <span className="text-slate-200 font-bold">
                {rec?.raw_prediction ?? '—'} ({(rec?.raw_confidence ?? 0).toFixed(4)})
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Smoothed Prediction:</span>
              <span className="text-cyan-400 font-bold">
                {rec?.smoothed_prediction ?? '—'} ({(rec?.smoothed_confidence ?? 0).toFixed(4)})
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Confidence Gate Passed:</span>
              <span
                className={`font-semibold ${
                  rec?.confidence_gate_passed ? 'text-green-400' : 'text-amber-400'
                }`}
              >
                {rec?.confidence_gate_passed ? 'PASS (>=0.70)' : 'FILTERED'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Static Fallback Label:</span>
              <span className="text-slate-200">{rec?.static_fallback_label ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Dynamic Trajectory:</span>
              <span className="text-slate-200">
                {rec?.confirmed_dynamic_result
                  ? `Confirmed: ${rec.confirmed_dynamic_result}`
                  : rec?.dynamic_candidate
                  ? `Candidate: ${rec.dynamic_candidate}`
                  : 'none'}
              </span>
            </div>
            {/* Top-3 Probability Candidates */}
            <div className="pt-2 border-t border-slate-800/60">
              <span className="text-slate-400 block mb-1.5">Top-3 Candidates:</span>
              {rec?.top_predictions && rec.top_predictions.length > 0 ? (
                <div className="space-y-1">
                  {rec.top_predictions.map((p, idx) => (
                    <div key={p.label || idx} className="flex items-center justify-between">
                      <span className="text-slate-300 font-bold">#{idx + 1} {p.label}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-violet-500 rounded-full"
                            style={{ width: `${Math.round(p.confidence * 100)}%` }}
                          />
                        </div>
                        <span className="text-slate-400 w-12 text-right">
                          {(p.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <span className="text-slate-500 italic">No prediction stream active</span>
              )}
            </div>
          </div>
        </div>

        {/* Card 6: Gesture State Machine (FSM) Hysteresis */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2 mb-4 pb-2 border-b border-slate-800">
            <Layers size={16} className="text-violet-400" />
            Temporal State Machine (FSM)
          </h2>
          <div className="space-y-2.5 font-mono text-xs">
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Current Candidate:</span>
              <span className="text-slate-100 font-bold text-sm">
                {rec?.fsm_candidate ?? '—'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Is Dwelling:</span>
              <span className={rec?.fsm_is_dwelling ? 'text-violet-400 font-bold' : 'text-slate-500'}>
                {rec?.fsm_is_dwelling ? 'TRUE' : 'FALSE'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Elapsed Dwell Time:</span>
              <span className="text-cyan-400 font-bold">
                {rec ? `${rec.fsm_dwell_time}s / 0.60s` : '0.0s'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Instantaneous Commit Event:</span>
              <span className="text-cyan-400 font-bold">
                {rec?.committed_label ?? 'none'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-t border-slate-800/60">
              <span className="text-slate-400">Speech Token:</span>
              <span className="text-slate-200">{rec?.speech_token ?? 'none'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Collapsible Raw JSON Telemetry Inspector */}
      <div className="mt-5 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <button
          onClick={() => setShowRawJson(!showRawJson)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors"
        >
          <span className="text-sm font-semibold text-slate-200 flex items-center gap-2 font-mono">
            <span>{'{ }'}</span>
            Raw WebSocket JSON Message Inspector
          </span>
          {showRawJson ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showRawJson && (
          <div className="p-4 border-t border-slate-800 bg-slate-950 font-mono text-xs text-slate-300 overflow-x-auto max-h-96">
            <pre>
              {rec
                ? JSON.stringify(rec, null, 2)
                : '// No frame recognition response received yet. Start camera in Live Translator to stream frames.'}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
