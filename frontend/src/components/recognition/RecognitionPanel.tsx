import { useState, useEffect } from 'react';
import type { RecognitionResponse, SystemInfoResponse } from '@/types/api';
import { Wifi, WifiOff } from 'lucide-react';

interface RecognitionPanelProps {
  recognition: RecognitionResponse | null;
  systemInfo: SystemInfoResponse | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
}

const RING_RADIUS = 44;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export function RecognitionPanel({ recognition, systemInfo, connectionStatus }: RecognitionPanelProps) {
  const dwellProgress = recognition?.dwell_progress ?? 0;

  // Committed flash state
  const [committed, setCommitted] = useState(false);
  useEffect(() => {
    if (recognition?.committed_label) {
      setCommitted(true);
      const t = setTimeout(() => setCommitted(false), 500);
      return () => clearTimeout(t);
    }
  }, [recognition?.committed_label]);

  // Recognition status badge
  const getStatusBadge = () => {
    if (committed) return { label: 'Committed ✓', cls: 'text-cyan-400 bg-cyan-500/10 ring-cyan-500/30' };
    if (!recognition?.hand_detected) return { label: 'No Hand', cls: 'text-slate-500 bg-slate-800 ring-slate-700' };
    if (recognition.fsm_is_dwelling) return { label: 'Stabilizing', cls: 'text-violet-400 bg-violet-500/10 ring-violet-500/30' };
    if (recognition.fsm_candidate) return { label: 'Candidate', cls: 'text-amber-400 bg-amber-500/10 ring-amber-500/30' };
    return { label: 'Searching', cls: 'text-slate-400 bg-slate-800 ring-slate-700' };
  };
  const badge = getStatusBadge();

  const confidence = recognition?.smoothed_confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);
  const confAboveThreshold = confidence >= (systemInfo?.min_confidence ?? 0.70);

  const wsLabel = connectionStatus === 'connected'
    ? 'Connected'
    : connectionStatus === 'connecting'
      ? 'Connecting…'
      : connectionStatus === 'error'
        ? 'WS Error'
        : 'Disconnected';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800">
        <h2 className="text-slate-100 font-semibold text-sm">Recognition</h2>
      </div>

      <div className="flex-1 p-4 flex flex-col gap-5 overflow-y-auto">
        {/* Dwell Ring + Candidate Letter */}
        <div className="flex flex-col items-center gap-2">
          <svg viewBox="0 0 100 100" className="w-32 h-32">
            {/* Background track */}
            <circle
              cx="50" cy="50" r={RING_RADIUS}
              fill="none"
              stroke="#1e293b"
              strokeWidth="8"
            />
            {/* Progress arc */}
            <circle
              cx="50" cy="50" r={RING_RADIUS}
              fill="none"
              stroke={committed ? '#22d3ee' : '#8b5cf6'}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={RING_CIRCUMFERENCE}
              strokeDashoffset={RING_CIRCUMFERENCE * (1 - dwellProgress)}
              transform="rotate(-90 50 50)"
              style={{ transition: 'stroke-dashoffset 0.1s ease-out, stroke 0.3s ease' }}
            />
            {/* Candidate letter in center */}
            <text
              x="50" y="56"
              textAnchor="middle"
              fontSize="28"
              fontWeight="700"
              fill={committed ? '#22d3ee' : '#f1f5f9'}
              fontFamily="Inter, sans-serif"
            >
              {recognition?.fsm_candidate ?? '--'}
            </text>
          </svg>

          {/* Status badge */}
          <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ring-1 ${badge.cls}`}>
            {badge.label}
          </span>
        </div>

        {/* Confidence bar */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-slate-400">Confidence</span>
            <span className={`text-xs font-mono font-semibold ${confAboveThreshold ? 'text-green-400' : 'text-amber-400'}`}>
              {confidencePct}%
            </span>
          </div>
          <div className="relative h-2 bg-slate-800 rounded-full overflow-visible">
            {/* Fill bar */}
            <div
              className={`h-full rounded-full transition-all duration-150 ${confAboveThreshold ? 'bg-green-500' : 'bg-amber-500'}`}
              style={{ width: `${confidencePct}%` }}
            />
            {/* Threshold marker at 70% */}
            <div
              className="absolute top-0 bottom-0 w-px bg-slate-500"
              style={{ left: '70%' }}
            />
          </div>
          <div className="flex justify-end mt-0.5">
            <span className="text-[10px] text-slate-500">0.70 threshold</span>
          </div>
        </div>

        {/* Handedness */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400">Handedness</span>
          <span className="text-xs font-semibold text-slate-200">
            {recognition?.handedness
              ? `${recognition.handedness.charAt(0).toUpperCase() + recognition.handedness.slice(1).toLowerCase()} hand`
              : '--'}
          </span>
        </div>

        {/* Raw vs Smoothed */}
        <div className="bg-slate-950/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Raw</span>
            <span className="text-xs font-mono font-semibold text-slate-300">
              {recognition?.raw_prediction ?? '--'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Smoothed</span>
            <span className="text-xs font-mono font-semibold text-violet-300">
              {recognition?.smoothed_prediction ?? '--'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Dwell</span>
            <span className="text-xs font-mono text-slate-400">
              {(dwellProgress * 0.60).toFixed(2)}s / 0.60s
            </span>
          </div>
        </div>

        {/* Top-3 Predictions */}
        <div className="bg-slate-950/50 rounded-lg p-3">
          <div className="text-xs text-slate-500 mb-2 font-medium">Top Predictions</div>
          <div className="space-y-1.5 h-[64px] flex flex-col justify-center">
            {recognition?.top_predictions && recognition.top_predictions.length > 0 ? (
              recognition.top_predictions.slice(0, 3).map((pred, i) => (
                <div key={pred.label || i} className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-semibold w-8">{pred.label}</span>
                  <div className="flex-1 mx-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-violet-500 rounded-full"
                      style={{ width: `${Math.round(pred.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-slate-400 w-8 text-right text-[11px]">
                    {Math.round(pred.confidence * 100)}%
                  </span>
                </div>
              ))
            ) : (
              <div className="space-y-1.5 opacity-30 select-none">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-500 font-semibold w-8">--</span>
                  <div className="flex-1 mx-2 h-1.5 bg-slate-800 rounded-full" />
                  <span className="text-slate-600 w-8 text-right text-[11px]">0%</span>
                </div>
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-500 font-semibold w-8">--</span>
                  <div className="flex-1 mx-2 h-1.5 bg-slate-800 rounded-full" />
                  <span className="text-slate-600 w-8 text-right text-[11px]">0%</span>
                </div>
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-500 font-semibold w-8">--</span>
                  <div className="flex-1 mx-2 h-1.5 bg-slate-800 rounded-full" />
                  <span className="text-slate-600 w-8 text-right text-[11px]">0%</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* System info (dwell/release durations) */}
        {systemInfo && (
          <div className="bg-slate-950/50 rounded-lg p-3 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">Dwell duration</span>
              <span className="text-xs font-mono text-slate-400">{systemInfo.dwell_duration}s</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">Classes</span>
              <span className="text-xs font-mono text-slate-400">{systemInfo.num_classes}</span>
            </div>
          </div>
        )}
      </div>

      {/* Connection status footer */}
      <div className="px-4 py-3 border-t border-slate-800 flex items-center gap-2">
        {connectionStatus === 'connected' ? (
          <Wifi className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
        ) : (
          <WifiOff className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
        )}
        <span
          className={`text-xs font-medium ${
            connectionStatus === 'connected'
              ? 'text-cyan-400'
              : connectionStatus === 'error'
                ? 'text-red-400'
                : 'text-slate-400'
          }`}
        >
          {wsLabel}
        </span>
        {connectionStatus === 'connected' && (
          <span className="ml-auto w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        )}
      </div>
    </div>
  );
}
