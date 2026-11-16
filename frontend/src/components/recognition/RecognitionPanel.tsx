import { useState, useEffect } from 'react';
import type { RecognitionResponse, SystemInfoResponse } from '@/types/api';

interface RecognitionPanelProps {
  recognition: RecognitionResponse | null;
  systemInfo: SystemInfoResponse | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
}

// SVG ring geometry — 200px viewBox, 12px stroke, r=84
const RING_RADIUS = 84;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS; // ≈ 527.8

/** Three placeholder rows always rendered to prevent layout jump */
const PLACEHOLDER_ROWS = [
  { label: '–', confidence: 0 },
  { label: '–', confidence: 0 },
  { label: '–', confidence: 0 },
];

export function RecognitionPanel({
  recognition,
  systemInfo,
  connectionStatus,
}: RecognitionPanelProps) {
  const dwellProgress = recognition?.dwell_progress ?? 0;
  const dwellDuration = systemInfo?.dwell_duration ?? 0.6;

  // Flash committed state ≤300ms
  const [committed, setCommitted] = useState(false);
  useEffect(() => {
    if (recognition?.committed_label) {
      setCommitted(true);
      const t = setTimeout(() => setCommitted(false), 300);
      return () => clearTimeout(t);
    }
  }, [recognition?.committed_label]);

  const candidate = recognition?.fsm_candidate ?? null;
  const handDetected = recognition?.hand_detected ?? false;
  const isDwelling = recognition?.fsm_is_dwelling ?? false;
  const confidence = recognition?.smoothed_confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);

  // Dwell ring arc — offset from full circumference
  const strokeDashoffset = RING_CIRCUMFERENCE * (1 - dwellProgress);

  // Ring colour: no hand → sand, dwelling → brand, committed → success
  const ringColor = committed
    ? 'var(--color-success)'
    : !handDetected || !candidate
    ? 'var(--color-sand)'
    : isDwelling
    ? 'var(--color-brand)'
    : 'var(--color-sand)';

  // Center character
  const centerChar = committed
    ? '✓'
    : candidate && handDetected
    ? candidate
    : '–';

  // Top-3 predictions — always render 3 rows
  const rawPredictions = recognition?.top_predictions ?? [];
  const rows =
    rawPredictions.length > 0
      ? rawPredictions.slice(0, 3).map((p) => ({ label: p.label, confidence: p.confidence }))
      : PLACEHOLDER_ROWS;

  // Pad to exactly 3 rows
  while (rows.length < 3) {
    rows.push({ label: '–', confidence: 0 });
  }

  const isConnected = connectionStatus === 'connected';

  return (
    <div className="flex flex-col gap-4">
      {/* ── Now reading card ─────────────────────────── */}
      <div
        className="bg-surface border p-5 flex flex-col items-center gap-4"
        style={{ borderColor: 'var(--color-line)', borderRadius: 'var(--radius-card)' }}
      >
        <span
          className="text-xs font-semibold uppercase tracking-widest self-start"
          style={{ color: 'var(--color-muted)', letterSpacing: '0.06em' }}
        >
          Now reading
        </span>

        {/* Dwell ring */}
        <svg
          width="200"
          height="200"
          viewBox="0 0 200 200"
          role="img"
          aria-label={
            candidate && handDetected
              ? `Recognising ${candidate}, ${Math.round(dwellProgress * 100)}% dwell progress`
              : 'No hand detected'
          }
        >
          {/* Track */}
          <circle
            cx="100"
            cy="100"
            r={RING_RADIUS}
            fill="none"
            stroke="var(--color-track)"
            strokeWidth="12"
          />
          {/* Progress arc */}
          <circle
            cx="100"
            cy="100"
            r={RING_RADIUS}
            fill="none"
            stroke={ringColor}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={RING_CIRCUMFERENCE}
            strokeDashoffset={strokeDashoffset}
            transform="rotate(-90 100 100)"
            style={{ transition: 'none' }}
          />
          {/* Center character */}
          <text
            x="100"
            y="130"
            textAnchor="middle"
            fontSize="92"
            fontWeight="700"
            fontFamily="'Inter Variable', system-ui, sans-serif"
            fill={committed ? 'var(--color-success)' : 'var(--color-ink)'}
            style={{ transition: 'fill 80ms linear' }}
          >
            {centerChar}
          </text>
        </svg>

        {/* Dwell label */}
        {handDetected && candidate && (
          <p
            className="text-sm font-medium text-center"
            style={{ color: 'var(--color-brand)' }}
          >
            Hold steady… {(dwellProgress * dwellDuration).toFixed(2)}s of {dwellDuration.toFixed(2)}s
          </p>
        )}

        {/* Confidence bar */}
        <div className="w-full">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm" style={{ color: 'var(--color-muted)' }}>
              Confidence
            </span>
            <span
              className="text-sm font-semibold"
              style={{ color: isConnected && confidence > 0 ? 'var(--color-ink)' : 'var(--color-muted)' }}
            >
              {isConnected && confidence > 0 ? `${confidencePct}%` : '–'}
            </span>
          </div>
          <div
            className="w-full h-2 rounded-full overflow-hidden"
            style={{ backgroundColor: 'var(--color-track)' }}
          >
            <div
              className="h-full rounded-full"
              style={{
                width: `${confidencePct}%`,
                backgroundColor: 'var(--color-brand)',
                transition: 'width 80ms linear',
              }}
            />
          </div>
        </div>
      </div>

      {/* ── Other possibilities card ─────────────────── */}
      <div
        className="bg-surface border p-5"
        style={{ borderColor: 'var(--color-line)', borderRadius: 'var(--radius-card)' }}
      >
        <span
          className="text-xs font-semibold uppercase tracking-widest block mb-4"
          style={{ color: 'var(--color-muted)', letterSpacing: '0.06em' }}
        >
          Other possibilities
        </span>

        <div className="flex flex-col gap-3">
          {rows.map((row, i) => {
            const pct = Math.round(row.confidence * 100);
            const isEmpty = row.label === '–' || pct === 0;
            return (
              <div key={i} className="flex items-center gap-3">
                <span
                  className="text-sm font-semibold w-6 flex-shrink-0"
                  style={{ color: isEmpty ? 'var(--color-sand)' : 'var(--color-ink)' }}
                >
                  {row.label}
                </span>
                <div
                  className="flex-1 h-2 rounded-full overflow-hidden"
                  style={{ backgroundColor: 'var(--color-track)' }}
                >
                  {!isEmpty && (
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: i === 0 ? 'var(--color-brand)' : 'var(--color-sand)',
                        transition: 'width 80ms linear',
                      }}
                    />
                  )}
                </div>
                <span
                  className="text-sm w-9 text-right flex-shrink-0"
                  style={{ color: isEmpty ? 'var(--color-sand)' : 'var(--color-muted)' }}
                >
                  {isEmpty ? '' : `${pct}%`}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── For best results card ────────────────────── */}
      <div
        className="border p-5"
        style={{
          backgroundColor: 'var(--color-coral-soft)',
          borderColor: 'var(--color-coral-line)',
          borderRadius: 'var(--radius-card)',
        }}
      >
        <span
          className="text-xs font-semibold uppercase tracking-widest block mb-3"
          style={{ color: 'var(--color-coral-text)', letterSpacing: '0.06em' }}
        >
          For best results
        </span>
        <ul className="flex flex-col gap-2 list-none m-0 p-0">
          {[
            'Use good, even light on your hand',
            "Keep your hand an arm's length away",
            'J and Z need motion, so they may read as I or D',
          ].map((tip) => (
            <li
              key={tip}
              className="text-sm flex items-start gap-2"
              style={{ color: 'var(--color-ink)' }}
            >
              <span aria-hidden="true">•</span>
              {tip}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
