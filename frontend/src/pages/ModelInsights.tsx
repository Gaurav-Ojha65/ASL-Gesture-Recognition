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
    <div className={`bg-slate-900 border border-slate-800 rounded-2xl p-6 ${className}`}>
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
    <h2 className="flex items-center gap-2 text-slate-100 text-lg font-semibold mb-5">
      <span className="text-violet-500">{icon}</span>
      {children}
    </h2>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-slate-800 last:border-0">
      <span className="text-slate-400 text-sm">{label}</span>
      <span className="text-slate-100 text-sm font-mono">{value}</span>
    </div>
  );
}

// ─── Pipeline Flow ─────────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  { label: 'Browser Frame', sub: 'getUserMedia()' },
  { label: 'MediaPipe HandLandmarker', sub: 'WASM model' },
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
    <div className="flex flex-col md:flex-row flex-wrap items-center gap-1 justify-center">
      {PIPELINE_STEPS.map((step, i) => (
        <div key={step.label} className="flex flex-col md:flex-row items-center gap-1">
          <div className="bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-center min-w-[120px]">
            <div className="text-slate-100 text-xs font-semibold leading-tight">
              {step.label}
            </div>
            <div className="text-slate-500 text-[10px] mt-0.5">{step.sub}</div>
          </div>
          {i < PIPELINE_STEPS.length - 1 && (
            <span className="text-violet-500 text-lg font-bold select-none md:rotate-0 rotate-90">
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
  { name: 'Input', size: 63, color: 'bg-violet-700' },
  { name: 'Hidden 1', size: 128, color: 'bg-violet-600' },
  { name: 'Hidden 2', size: 64, color: 'bg-fuchsia-600' },
  { name: 'Output', size: 28, color: 'bg-cyan-500' },
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
              <span className="text-slate-400 text-xs font-mono">{layer.size}</span>
              <div
                className={`w-10 rounded-t-lg ${layer.color} opacity-90`}
                style={{ height: `${h}px` }}
              />
              <span className="text-slate-300 text-xs text-center">{layer.name}</span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 text-center text-slate-400 text-xs">
        Total parameters:{' '}
        <span className="text-cyan-400 font-mono font-semibold">
          {paramCount !== undefined ? paramCount.toLocaleString() : '18,524'}
        </span>{' '}
        {paramCount === undefined && (
          <span className="text-slate-500">(hardcoded fallback — backend offline)</span>
        )}
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
  const dwell = sysInfo ? `${sysInfo.dwell_duration.toFixed(2)}s` : '0.60s';
  const release = sysInfo ? `${sysInfo.release_duration.toFixed(2)}s` : '0.25s';
  const minConf = sysInfo ? sysInfo.min_confidence.toFixed(2) : '0.70';

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
            className="bg-slate-800 border border-slate-700 rounded-xl p-4 text-center"
          >
            <div className="text-cyan-400 font-mono text-2xl font-bold mb-1">{value}</div>
            <div className="text-slate-200 text-sm font-semibold mb-1">{label}</div>
            <div className="text-slate-500 text-xs">{desc}</div>
          </div>
        ))}
      </div>
      <div className="bg-slate-800 rounded-xl p-4 text-sm text-slate-300 space-y-2">
        <p>
          <span className="text-violet-400 font-semibold">Commit:</span> A predicted class must
          remain the top prediction continuously for <span className="text-cyan-400">{dwell}</span>{' '}
          before it is committed to the text buffer.
        </p>
        <p>
          <span className="text-violet-400 font-semibold">Hold debounce:</span> Once committed, the
          same pose does not re-commit while the hand stays in the same position.
        </p>
        <p>
          <span className="text-violet-400 font-semibold">Release:</span> The dwell timer resets
          only after{' '}
          <span className="text-cyan-400">{release}</span> of confirmed hand absence or class
          change.
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
              ? 'border-amber-500 bg-amber-900/20'
              : 'border-slate-700 bg-slate-800'
          }`}
        >
          <span
            className={`font-bold text-sm ${
              DYNAMIC.has(cls) ? 'text-amber-300' : 'text-slate-100'
            }`}
          >
            {cls}
          </span>
          {DYNAMIC.has(cls) && (
            <span className="text-[9px] text-amber-400 font-semibold mt-0.5 leading-tight">
              dynamic
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
      <div className="bg-amber-900/20 border border-amber-700 rounded-xl p-3 text-amber-300 text-sm font-semibold flex items-start gap-2">
        <AlertTriangle size={16} className="shrink-0 mt-0.5" />
        This CNN is a <strong>separate model</strong> evaluated on the Sign Language MNIST dataset.
        It is NOT used for live webcam recognition.
      </div>
      <div className="grid sm:grid-cols-2 gap-2">
        <InfoRow label="Model type" value="CNN (Sign Language MNIST)" />
        <InfoRow label="Input" value="28 × 28 grayscale" />
        <InfoRow label="Input features" value="784 pixels" />
        <InfoRow label="Dataset" value="Sign Language MNIST" />
        <InfoRow label="Classes" value="24 (excludes J, Z)" />
        <InfoRow label="Purpose" value="Benchmark / research only" />
      </div>
      <p className="text-slate-500 text-xs">
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
  const severityColors: Record<string, string> = {
    high: 'text-red-400 border-red-700 bg-red-900/20',
    medium: 'text-amber-400 border-amber-700 bg-amber-900/20',
    info: 'text-slate-400 border-slate-700 bg-slate-800',
  };
  return (
    <div className="space-y-3">
      {LIMITATIONS.map(({ sign, issue, severity }) => (
        <div
          key={sign}
          className={`border rounded-xl p-4 ${severityColors[severity]}`}
        >
          <div className="font-semibold text-sm mb-1">{sign}</div>
          <div className="text-xs opacity-80">{issue}</div>
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
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10">
      <div className="max-w-5xl mx-auto">
        {/* Page heading */}
        <div className="mb-10">
          <h1 className="text-3xl md:text-4xl font-extrabold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent mb-2">
            Model Insights
          </h1>
          <p className="text-slate-400 text-base">
            How ASL Gesture Studio works — from browser frame to committed text.
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
            <p className="text-slate-400 text-sm mb-4">
              All 28 output classes. Amber badges indicate signs that require motion and may be
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
