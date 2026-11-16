import { useEffect, useRef, useCallback } from 'react';
import { useWebcam } from '@/hooks/useWebcam';
import { useRecognition } from '@/hooks/useRecognition';
import { renderLandmarks, clearCanvas, getContainArea } from '@/utils/landmarkRenderer';
import type { RecognitionResponse, ClientFrameMetadata } from '@/types/api';
import { Camera, CameraOff, Wifi, WifiOff } from 'lucide-react';

interface WebcamPanelProps {
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  sendFrame: (meta: ClientFrameMetadata, jpeg: ArrayBuffer) => boolean;
  lastRecognition: RecognitionResponse | null;
}

export function WebcamPanel({ connectionStatus, sendFrame, lastRecognition }: WebcamPanelProps) {
  const { videoRef, cameraStatus, videoWidth, videoHeight, startCamera, stopCamera, errorMessage } =
    useWebcam();

  const isActive = cameraStatus === 'active' && connectionStatus === 'connected';

  const { isStreaming, fps } = useRecognition({
    videoRef,
    isActive,
    sendFrame,
    lastRecognition,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Render landmarks on each recognition update
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const container = containerRef.current;
    if (!container) return;

    const { clientWidth, clientHeight } = container;
    canvas.width = clientWidth * dpr;
    canvas.height = clientHeight * dpr;

    if (!lastRecognition?.hand_detected || !lastRecognition.landmarks) {
      clearCanvas(canvas, ctx);
      return;
    }

    const area = getContainArea(clientWidth, clientHeight, videoWidth, videoHeight);
    renderLandmarks(ctx, lastRecognition.landmarks, area, dpr);
  }, [lastRecognition, videoWidth, videoHeight]);

  // Resize canvas when container resizes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = container.clientWidth * dpr;
      canvas.height = container.clientHeight * dpr;
      clearCanvas(canvas, ctx);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleToggle = useCallback(() => {
    if (cameraStatus === 'active') {
      stopCamera();
    } else {
      startCamera();
    }
  }, [cameraStatus, startCamera, stopCamera]);

  const handDetected = lastRecognition?.hand_detected ?? false;

  const statusLabel: Record<string, string> = {
    idle: 'Idle',
    requesting: 'Requesting…',
    active: 'Active',
    error: 'Error',
    stopped: 'Stopped',
  };

  const wsLabel =
    connectionStatus === 'connected'
      ? 'Connected'
      : connectionStatus === 'connecting'
        ? 'Connecting…'
        : connectionStatus === 'error'
          ? 'WS Error'
          : 'Disconnected';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <Camera className="w-5 h-5 text-violet-500" />
          <span className="text-slate-100 font-semibold text-sm">Webcam</span>
          {/* Camera status badge */}
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              cameraStatus === 'active'
                ? 'bg-violet-500/20 text-violet-400'
                : cameraStatus === 'error'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-slate-800 text-slate-400'
            }`}
          >
            {statusLabel[cameraStatus] ?? cameraStatus}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* WebSocket status */}
          <div className="flex items-center gap-1.5">
            {connectionStatus === 'connected' ? (
              <Wifi className="w-4 h-4 text-cyan-400" />
            ) : (
              <WifiOff className="w-4 h-4 text-slate-500" />
            )}
            <span className="text-xs text-slate-400">{wsLabel}</span>
          </div>

          {/* FPS indicator */}
          {isStreaming && (
            <span className="text-xs text-slate-400 font-mono">{fps} fps</span>
          )}

          {/* Camera toggle */}
          <button
            onClick={handleToggle}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              cameraStatus === 'active'
                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                : 'bg-violet-500/20 text-violet-400 hover:bg-violet-500/30'
            }`}
          >
            {cameraStatus === 'active' ? (
              <>
                <CameraOff className="w-3.5 h-3.5" />
                Stop
              </>
            ) : (
              <>
                <Camera className="w-3.5 h-3.5" />
                Start
              </>
            )}
          </button>
        </div>
      </div>

      {/* Video area */}
      <div
        ref={containerRef}
        className="relative w-full bg-slate-950"
        style={{ aspectRatio: '16/9' }}
      >
        {/* Mirrored video for natural display */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 w-full h-full"
          style={{ transform: 'scaleX(-1)', objectFit: 'contain' }}
        />

        {/* Canvas overlay — NOT mirrored, coordinates already aligned */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ display: 'block' }}
        />

        {/* Idle/error overlay */}
        {cameraStatus !== 'active' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80">
            {cameraStatus === 'error' ? (
              <>
                <CameraOff className="w-10 h-10 text-red-400 mb-3" />
                <p className="text-red-400 text-sm font-medium text-center px-6">{errorMessage}</p>
              </>
            ) : cameraStatus === 'requesting' ? (
              <>
                <Camera className="w-10 h-10 text-violet-400 mb-3 animate-pulse" />
                <p className="text-slate-400 text-sm">Requesting camera…</p>
              </>
            ) : (
              <>
                <Camera className="w-10 h-10 text-slate-600 mb-3" />
                <p className="text-slate-500 text-sm">Camera not started</p>
                <button
                  onClick={startCamera}
                  className="mt-3 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold rounded-lg transition-colors"
                >
                  Enable Camera
                </button>
              </>
            )}
          </div>
        )}

        {/* Hand detected badge */}
        <div className="absolute bottom-3 left-3">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
              handDetected
                ? 'bg-green-500/20 text-green-400 ring-1 ring-green-500/40'
                : 'bg-slate-800/80 text-slate-400'
            }`}
          >
            {handDetected ? '✋ Hand Detected' : 'No Hand'}
          </span>
        </div>

        {/* Resolution badge */}
        {videoWidth > 0 && videoHeight > 0 && (
          <div className="absolute bottom-3 right-3">
            <span className="text-xs px-2 py-1 rounded bg-slate-900/80 text-slate-400 font-mono">
              {videoWidth}×{videoHeight}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
