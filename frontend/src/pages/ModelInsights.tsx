import { useEffect, useState } from 'react';
import { fetchSystemInfo } from '@/services/api';
import type { SystemInfoResponse } from '@/types/api';
import { Brain, GitBranch, AlertTriangle, BookOpen, FlaskConical } from 'lucide-react';

// ─── Shared helpers ────────────────────────────────────────────────────────────

function Card({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-surface border border-line rounded-card p-6 sm:p-8 ${className}`}>
      {children}
    </div>
  );
}

function CardTitle({
  icon,
  children,
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <h2 className="flex items-center gap-2 text-ink text-lg font-semibold mb-5">
      <span className="text-brand">{icon}</span>
      {children}
    </h2>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-line last:border-0">
      <span className="text-muted text-sm">{label}</span>
      <span className="text-ink text-sm font-mono">{value}</span>
    </div>
  );
}

// ─── Pipeline Flow ─────────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  { label: 'Browser Frame', sub: 'getUserMedia()' },
  { label: 'MediaPipe HandLandmarker', sub: 'Python Server-Side' },
  { label: '21 Hand Landmarks', sub: 'x, y, z per point' },
  { label: 'Landmark Normalization', sub: 'wrist-relative, scale-invariant' },
  { label: '63-D Feature Vector', sub: '21 × 3 floats' },
  { label: 'PyTorch MLP', sub: '63 → 128 → 64 → 28' },
  { label: '28 Output Classes', sub: 'A–Z + SPACE + DELETE' },
  { label: 'Gesture State Machine', sub: 'dwell / release debounce' },
  { label: 'TextBuffer', sub: 'committed text output' },
];

function PipelineFlow() {
  return (
    <div className="flex flex-col md:flex-row flex-wrap items-center gap-2 justify-center">
      {PIPELINE_STEPS.map((step, i) => (
        <div key={step.label} className="flex flex-col md:flex-row items-center gap-2">
          <div className="bg-track border border-line rounded-xl px-3 py-2 text-center min-w-[124px]">
            <div className="text-ink text-xs font-semibold leading-tight">
              {step.label}
            </div>
            <div className="text-muted text-[10px] mt-0.5">{step.sub}</div>
          </div>
          {i < PIPELINE_STEPS.length - 1 && (
            <span className="text-brand text-lg font-bold select-none md:rotate-0 rotate-90">
              →
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── MLP Architecture ─────────────────────────────────────────────────────────

const LAYERS = [
  { name: 'Input', size: 63, color: 'bg-sand' },
  { name: 'Hidden 1', size: 128, color: 'bg-brand-soft' },
  { name: 'Hidden 2', size: 64, color: 'bg-brand' },
  { name: 'Output', size: 28, color: 'bg-coral-text' },
];

function MlpDiagram({ paramCount }: { paramCount?: number }) {
  const maxSize = Math.max(...LAYERS.map((l) => l.size));
  return (
    <div>
      <div className="flex items-end justify-around gap-4 mb-4">
        {LAYERS.map((layer) => {
          const heightPct = (layer.size / maxSize) * 100;
          const minH = 32;
          const maxH = 120;
          const h = minH + ((maxH - minH) * heightPct) / 100;
          return (
            <div key={layer.name} className="flex flex-col items-center gap-2">
              <span className="text-muted text-xs font-mono">{layer.size}</span>
              <div
                className={`w-10 rounded-t-lg ${layer.color}`}
                style={{ height: `${h}px` }}
              />
              <span className="text-ink text-xs font-medium text-center">{layer.name}</span>
            </div>
          );
        })}
      </div>
      <div className="mt-4 text-center text-muted text-xs">
        Total parameters:{' '}
        <span className="text-ink font-mono font-bold">
          {paramCount !== undefined ? paramCount.toLocaleString() : '18,524'}
        </span>{' '}
        {paramCount === undefined && (
          <span className="text-muted">(hardcoded fallback — backend offline)</span>
        )}
        <span className="text-muted text-[11px] block mt-0.5">
          (Note: Total of 18,524 parameters includes Linear weights, biases, and BatchNorm layers)
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 mt-4">
        <InfoRow label="Activation (hidden)" value="ReLU" />
        <InfoRow label="Activation (output)" value="Softmax" />
        <InfoRow label="Input dim" value="63" />
        <InfoRow label="Output classes" value="28" />
      </div>
    </div>
  );
}

// ─── Gesture State Machine ────────────────────────────────────────────────────

function GestureStateMachine({ sysInfo }: { sysInfo: SystemInfoResponse | null }) {
  const dwell = sysInfo ? `${sysInfo.dwell_duration.toFixed(2)}s` : '–';
  const release = sysInfo ? `${sysInfo.release_duration.toFixed(2)}s` : '–';
  const minConf = sysInfo ? sysInfo.min_confidence.toFixed(2) : '–';

  const params = [
    { label: 'Dwell duration', value: dwell, desc: 'Gesture must be stable before commit' },
    { label: 'Release duration', value: release, desc: 'Confirmed absence required to reset' },
    { label: 'Min confidence', value: minConf, desc: 'Softmax threshold for a valid prediction' },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {params.map(({ label, value, desc }) => (
          <div
            key={label}
            className="bg-track border border-line rounded-xl p-4 text-center"
          >
            <div className="text-ink font-mono text-2xl font-bold mb-1">{value}</div>
            <div className="text-ink text-sm font-semibold mb-1">{label}</div>
            <div className="text-muted text-xs">{desc}</div>
          </div>
        ))}
      </div>
      <div className="bg-track/50 border border-line rounded-xl p-4 text-sm text-ink space-y-2">
        <p>
          <span className="text-brand font-semibold">Commit:</span> A predicted class must
          remain the top prediction continuously for{' '}
          <span className="font-mono font-bold text-brand">{dwell}</span> before it is committed to
          the text buffer.
        </p>
        <p>
          <span className="text-brand font-semibold">Hold debounce:</span> Once committed, the
          same pose does not re-commit while the hand stays in the same position.
        </p>
        <p>
          <span className="text-brand font-semibold">Release:</span> The dwell timer resets only
          after <span className="font-mono font-bold text-brand">{release}</span> of confirmed hand
          absence or class change.
        </p>
      </div>
    </div>
  );
}

// ─── Sign Class Reference ──────────────────────────────────────────────────────

const SIGN_CLASSES = [
  ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)),
  'SPACE',
  'DELETE',
];
const DYNAMIC = new Set(['J', 'Z']);

function SignClassGrid() {
  return (
    <div className="grid grid-cols-7 sm:grid-cols-9 md:grid-cols-14 gap-2">
      {SIGN_CLASSES.map((cls) => (
        <div
          key={cls}
          className={`relative flex flex-col items-center justify-center rounded-lg p-2 border text-center ${
            DYNAMIC.has(cls)
              ? 'border-coral-line bg-coral-soft'
              : 'border-line bg-track'
          }`}
        >
          <span
            className={`font-bold text-sm ${
              DYNAMIC.has(cls) ? 'text-coral-text' : 'text-ink'
            }`}
          >
            {cls}
          </span>
          {DYNAMIC.has(cls) && (
            <span className="text-[9px] text-coral-text font-semibold uppercase tracking-wider mt-0.5 leading-tight">
              motion
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── MNIST CNN Benchmark ───────────────────────────────────────────────────────

function MnistBenchmark() {
  return (
    <div className="space-y-3">
      <div className="bg-coral-soft border border-coral-line rounded-xl p-4 text-coral-text text-sm font-medium flex items-start gap-2">
        <AlertTriangle size={16} className="shrink-0 mt-0.5" />
        <div>
          This CNN is a <strong>separate model</strong> evaluated on the Sign Language MNIST dataset.
          It is NOT used for live webcam recognition.
        </div>
      </div>
      <div className="grid sm:grid-cols-2 gap-2">
        <InfoRow label="Model type" value="CNN (Sign Language MNIST)" />
        <InfoRow label="Input" value="28 × 28 grayscale" />
        <InfoRow label="Input features" value="784 pixels" />
        <InfoRow label="Dataset" value="Sign Language MNIST" />
        <InfoRow label="Classes" value="24 (excludes J, Z)" />
        <InfoRow label="Purpose" value="Benchmark / research only" />
      </div>
      <p className="text-muted text-xs">
        The live webcam model is a landmark-based MLP (63-D feature vectors from MediaPipe).
        These two models have different inputs, architectures, and evaluation datasets — do not
        compare their metrics directly.
      </p>
    </div>
  );
}

// ─── Known Limitations ────────────────────────────────────────────────────────

const LIMITATIONS = [
  {
    sign: 'J / Z',
    issue:
      'Require motion (arc for J, Z-stroke for Z). The static handshape fallback resembles I (for J) or D (for Z), causing misclassification.',
    severity: 'high',
  },
  {
    sign: 'T / M / N',
    issue:
      'T can be confused with other fist-shaped gestures such as M and N due to subtle thumb placement differences.',
    severity: 'medium',
  },
  {
    sign: 'General',
    issue:
      'Best results at moderate camera distance with the hand centered in the frame and good, even lighting. Extreme angles or poor lighting reduce recognition confidence.',
    severity: 'info',
  },
];

function LimitationsCard() {
  return (
    <div className="space-y-3">
      {LIMITATIONS.map(({ sign, issue, severity }) => (
        <div
          key={sign}
          className={`border rounded-xl p-4 ${
            severity === 'high'
              ? 'bg-coral-soft border-coral-line text-coral-text'
              : 'bg-track border-line text-ink'
          }`}
        >
          <div
            className={`font-semibold text-sm mb-1 ${
              severity === 'high' ? 'text-coral-text' : 'text-ink'
            }`}
          >
            {sign}
          </div>
          <div
            className={`text-xs ${
              severity === 'high' ? 'text-coral-text/90' : 'text-muted'
            }`}
          >
            {issue}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ModelInsights() {
  const [sysInfo, setSysInfo] = useState<SystemInfoResponse | null>(null);

  useEffect(() => {
    fetchSystemInfo()
      .then((d) => setSysInfo(d))
      .catch(() => {/* offline — fallback values used */});
  }, []);

  return (
    <div className="min-h-screen bg-bg text-ink">
      <div className="max-w-[1240px] mx-auto px-5 md:px-10 py-10 flex flex-col gap-10">
        {/* Page heading */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-ink mb-2">
            How it works
          </h1>
          <p className="text-muted text-base">
            From browser camera frame to committed text — architecture, parameters, and design principles.
          </p>
        </div>

        <div className="space-y-6">

          {/* 1. Pipeline Flow */}
          <Card>
            <CardTitle icon={<GitBranch size={18} />}>End-to-End Pipeline</CardTitle>
            <PipelineFlow />
          </Card>

          {/* 2. MLP Architecture */}
          <Card>
            <CardTitle icon={<Brain size={18} />}>MLP Architecture (Live Webcam Model)</CardTitle>
            <MlpDiagram paramCount={sysInfo?.model_param_count} />
          </Card>

          {/* 3. Gesture State Machine */}
          <Card>
            <CardTitle icon={<GitBranch size={18} />}>Gesture State Machine</CardTitle>
            <GestureStateMachine sysInfo={sysInfo} />
          </Card>

          {/* 4. Sign Class Reference */}
          <Card>
            <CardTitle icon={<BookOpen size={18} />}>Sign Class Reference</CardTitle>
            <p className="text-muted text-sm mb-4">
              All 28 output classes. Coral badges indicate signs that require motion and may be
              less reliable with static landmark analysis.
            </p>
            <SignClassGrid />
          </Card>

          {/* 5. MNIST CNN Benchmark */}
          <Card>
            <CardTitle icon={<FlaskConical size={18} />}>
              Separate MNIST CNN Benchmark — Not the Live Webcam Model
            </CardTitle>
            <MnistBenchmark />
          </Card>

          {/* 6. Known Limitations */}
          <Card>
            <CardTitle icon={<AlertTriangle size={18} />}>Known Recognition Limitations</CardTitle>
            <LimitationsCard />
          </Card>

        </div>
      </div>
    </div>
  );
}
