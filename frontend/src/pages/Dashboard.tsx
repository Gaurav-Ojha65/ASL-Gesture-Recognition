import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchHealth, fetchSystemInfo } from '@/services/api';
import type { HealthResponse, SystemInfoResponse } from '@/types/api';
import {
  Cpu,
  Wifi,
  Camera,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  Zap,
  Video,
} from 'lucide-react';

// ─── Helpers ───────────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: 'healthy' | 'degraded' | 'offline' }) {
  const colors: Record<string, string> = {
    healthy: 'bg-green-400',
    degraded: 'bg-amber-400',
    offline: 'bg-red-500',
  };
  return (
    <span
      className={`inline-block w-3 h-3 rounded-full ${colors[status] ?? 'bg-red-500'} shadow-lg`}
      aria-label={status}
    />
  );
}

function Card({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-slate-900 border border-slate-800 rounded-2xl p-6 ${className}`}
    >
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
    <h2 className="flex items-center gap-2 text-slate-100 text-lg font-semibold mb-4">
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

// ─── Main Component ────────────────────────────────────────────────────────────

export function Dashboard() {
  const navigate = useNavigate();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [sysInfo, setSysInfo] = useState<SystemInfoResponse | null>(null);
  const [sysInfoError, setSysInfoError] = useState(false);

  // Fetch health (polled every 10s)
  const pollHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setHealthError(false);
    } catch {
      setHealth(null);
      setHealthError(true);
    }
  }, []);

  // Fetch system info once on mount
  useEffect(() => {
    fetchSystemInfo()
      .then((data) => setSysInfo(data))
      .catch(() => setSysInfoError(true));
  }, []);

  // Health polling
  useEffect(() => {
    pollHealth();
    const id = setInterval(pollHealth, 10_000);
    return () => clearInterval(id);
  }, [pollHealth]);

  // Derived status label
  const healthStatus: 'healthy' | 'degraded' | 'offline' = healthError
    ? 'offline'
    : health?.status === 'degraded'
    ? 'degraded'
    : health
    ? 'healthy'
    : 'offline';

  const statusLabel = healthError
    ? 'Backend Offline'
    : health?.status === 'healthy'
    ? 'Healthy'
    : health?.status === 'degraded'
    ? 'Degraded'
    : 'Connecting…';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10">
      {/* ── Hero ───────────────────────────────────────────────────────────── */}
      <section className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent mb-4">
          Welcome to ASL Gesture Studio
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto mb-8">
          Transform hand gestures into text in real time using computer vision and
          deep learning.
        </p>
        <button
          onClick={() => navigate('/translator')}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:opacity-90 transition-opacity shadow-lg shadow-violet-900/40"
        >
          <Video size={18} />
          Launch Translator
        </button>
      </section>

      {/* ── Cards grid ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">

        {/* System Status Card */}
        <Card>
          <CardTitle icon={<Wifi size={18} />}>System Status</CardTitle>
          <div className="flex items-center gap-3 mb-4">
            <StatusDot status={healthStatus} />
            <span
              className={
                healthStatus === 'healthy'
                  ? 'text-green-400 font-semibold'
                  : healthStatus === 'degraded'
                  ? 'text-amber-400 font-semibold'
                  : 'text-red-400 font-semibold'
              }
            >
              {statusLabel}
            </span>
            <span className="text-slate-500 text-xs ml-auto">polls every 10s</span>
          </div>
          {healthError ? (
            <div className="flex items-center gap-2 text-red-400 text-sm mt-2">
              <XCircle size={16} />
              Cannot reach backend at <code className="text-xs">/api/health</code>
            </div>
          ) : health ? (
            <>
              <InfoRow
                label="Model Loaded"
                value={
                  health.model_loaded ? (
                    <CheckCircle size={16} className="text-green-400" />
                  ) : (
                    <XCircle size={16} className="text-red-400" />
                  )
                }
              />
              <InfoRow
                label="Detector Loaded"
                value={
                  health.detector_loaded ? (
                    <CheckCircle size={16} className="text-green-400" />
                  ) : (
                    <XCircle size={16} className="text-red-400" />
                  )
                }
              />
              <InfoRow
                label="Active Sessions"
                value={health.active_sessions}
              />
            </>
          ) : (
            <p className="text-slate-500 text-sm">Connecting to backend…</p>
          )}
        </Card>

        {/* Model Info Card */}
        <Card>
          <CardTitle icon={<Cpu size={18} />}>Model Info</CardTitle>
          {sysInfoError ? (
            <div className="flex items-center gap-2 text-red-400 text-sm">
              <XCircle size={16} />
              Could not load system info.
            </div>
          ) : sysInfo ? (
            <>
              <InfoRow label="App" value={sysInfo.app_name} />
              <InfoRow label="Output Classes" value={sysInfo.num_classes} />
              <InfoRow
                label="Parameters"
                value={sysInfo.model_param_count.toLocaleString()}
              />
              <InfoRow label="Input Dim" value={sysInfo.input_dim} />
              <InfoRow label="Device" value={sysInfo.device} />
              <InfoRow label="Torch" value={sysInfo.torch_version} />
            </>
          ) : (
            <p className="text-slate-500 text-sm">Loading model info…</p>
          )}
        </Card>

        {/* Capabilities Card */}
        <Card>
          <CardTitle icon={<Zap size={18} />}>Capabilities</CardTitle>
          <ul className="space-y-3">
            {[
              {
                icon: <Camera size={16} className="text-violet-500 shrink-0" />,
                text: '21 MediaPipe hand landmarks',
              },
              {
                icon: <Info size={16} className="text-violet-500 shrink-0" />,
                text: '63-dimensional feature representation',
              },
              {
                icon: <CheckCircle size={16} className="text-violet-500 shrink-0" />,
                text: '28 output classes: A–Z, SPACE, DELETE',
              },
              {
                icon: <Zap size={16} className="text-violet-500 shrink-0" />,
                text: 'Up to 15 FPS real-time recognition',
              },
            ].map(({ icon, text }) => (
              <li key={text} className="flex items-start gap-3 text-slate-300 text-sm">
                {icon}
                <span>{text}</span>
              </li>
            ))}
          </ul>
        </Card>

        {/* Known Limitations Card */}
        <Card>
          <CardTitle icon={<AlertTriangle size={18} />}>Known Limitations</CardTitle>
          <ul className="space-y-3">
            {[
              'J and Z require motion — static handshape fallback may show I or D respectively.',
              'T can be confused with nearby fist-shaped gestures (M, N).',
              'Best recognition at moderate distance, centered hand, and good lighting.',
              'No claim of perfect recognition accuracy — results vary with hand orientation and lighting conditions.',
            ].map((text) => (
              <li key={text} className="flex items-start gap-3 text-sm text-slate-400">
                <AlertTriangle
                  size={15}
                  className="text-amber-400 shrink-0 mt-0.5"
                />
                <span>{text}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
