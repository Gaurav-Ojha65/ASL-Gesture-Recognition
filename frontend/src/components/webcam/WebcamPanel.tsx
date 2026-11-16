import { useEffect, useRef, useCallback } from 'react';
import { useWebcam } from '@/hooks/useWebcam';
import { useRecognition } from '@/hooks/useRecognition';
import { renderLandmarks, clearCanvas, getContainArea } from '@/utils/landmarkRenderer';
import type { RecognitionResponse, ClientFrameMetadata } from '@/types/api';
import { useCameraStatus } from '@/context/CameraStatusContext';

interface WebcamPanelProps {
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  sendFrame: (meta: ClientFrameMetadata, jpeg: ArrayBuffer) => boolean;
  lastRecognition: RecognitionResponse | null;
}

/** Idle-state hand outline SVG illustration */
function HandIllustration() {
  return (
    <svg
      width="80"
      height="96"
      viewBox="0 0 80 96"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Palm */}
      <rect x="20" y="56" width="40" height="30" rx="12" stroke="#C9BFAE" strokeWidth="2.5" fill="none" />
      {/* Index finger */}
      <rect x="22" y="22" width="9" height="36" rx="4.5" stroke="#C9BFAE" strokeWidth="2.5" fill="none" />
      {/* Middle finger */}
      <rect x="34" y="12" width="9" height="46" rx="4.5" stroke="#C9BFAE" strokeWidth="2.5" fill="none" />
      {/* Ring finger */}
      <rect x="46" y="18" width="9" height="40" rx="4.5" stroke="#C9BFAE" strokeWidth="2.5" fill="none" />
      {/* Pinky */}
      <rect x="58" y="28" width="8" height="30" rx="4" stroke="#C9BFAE" strokeWidth="2.5" fill="none" />
      {/* Joints — coral dots */}
      <circle cx="26.5" cy="40" r="3" fill="#B23E26" />
      <circle cx="38.5" cy="30" r="3" fill="#B23E26" />
      <circle cx="50.5" cy="36" r="3" fill="#B23E26" />
      <circle cx="62" cy="46" r="3" fill="#B23E26" />
      <circle cx="40" cy="68" r="4" fill="#B23E26" />
    </svg>
  );
}

export function WebcamPanel({ connectionStatus, sendFrame, lastRecognition }: WebcamPanelProps) {
  const { videoRef, cameraStatus, videoWidth, videoHeight, startCamera, stopCamera, errorMessage } =
    useWebcam();

  const { setCameraActive } = useCameraStatus();

  const isActive = cameraStatus === 'active' && connectionStatus === 'connected';

  const { isStreaming, fps } = useRecognition({
    videoRef,
    isActive,
    sendFrame,
    lastRecognition,
  });

  // Sync camera state to context for NavBar status pill
  useEffect(() => {
    setCameraActive(cameraStatus === 'active');
  }, [cameraStatus, setCameraActive]);

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
  const handedness = lastRecognition?.handedness ?? null;

  return (
    <div
      className="bg-surface border overflow-hidden"
      style={{ borderColor: 'var(--color-line)', borderRadius: 'var(--radius-card)' }}
    >
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-5 py-3 border-b"
        style={{ borderColor: 'var(--color-line)' }}
      >
        <span className="font-semibold text-sm" style={{ color: 'var(--color-ink)' }}>
          Camera
        </span>

        <div className="flex items-center gap-2 btn-sm">
          {/* FPS pill */}
          {isStreaming && (
            <span
              className="btn-sm text-xs px-3 rounded-pill font-medium flex items-center"
              style={{
                backgroundColor: 'var(--color-track)',
                color: 'var(--color-muted)',
              }}
            >
              {fps} fps
            </span>
          )}

          {/* Stop / Start button */}
          <button
            onClick={handleToggle}
            className="btn-sm text-xs font-semibold px-4 rounded-pill border transition-colors"
            style={
              cameraStatus === 'active'
                ? {
                    color: 'var(--color-coral-text)',
                    borderColor: 'var(--color-coral-line)',
                    backgroundColor: 'var(--color-coral-soft)',
                  }
                : {
                    color: 'var(--color-brand)',
                    borderColor: 'var(--color-brand)',
                    backgroundColor: 'var(--color-brand-soft)',
                  }
            }
            aria-label={cameraStatus === 'active' ? 'Stop camera' : 'Start camera'}
          >
            {cameraStatus === 'active' ? 'Stop' : 'Start'}
          </button>
        </div>
      </div>

      {/* Video area */}
      <div
        ref={containerRef}
        className="relative w-full"
        style={{ height: '420px', backgroundColor: 'var(--color-track)' }}
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

        {/* Idle / error overlay */}
        {cameraStatus !== 'active' && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center gap-4"
            style={{ backgroundColor: 'var(--color-track)' }}
          >
            {cameraStatus === 'error' ? (
              <>
                <HandIllustration />
                <p
                  className="text-sm font-medium text-center px-6"
                  style={{ color: 'var(--color-coral-text)' }}
                >
                  {errorMessage}
                </p>
              </>
            ) : cameraStatus === 'requesting' ? (
              <>
                <HandIllustration />
                <p className="text-sm" style={{ color: 'var(--color-muted)' }}>
                  Requesting camera…
                </p>
              </>
            ) : (
              <>
                <HandIllustration />
                <p className="text-sm text-center px-8" style={{ color: 'var(--color-muted)' }}>
                  Keep your hand centered in the frame
                </p>
                <button
                  onClick={startCamera}
                  className="text-sm font-semibold px-6 py-2.5 transition-colors"
                  style={{
                    backgroundColor: 'var(--color-brand)',
                    color: 'white',
                    borderRadius: 'var(--radius-button)',
                  }}
                  aria-label="Enable camera"
                >
                  Enable camera
                </button>
              </>
            )}
          </div>
        )}

        {/* Hand detected badge — bottom left */}
        {cameraStatus === 'active' && (
          <div className="absolute bottom-3 left-3">
            <span
              className="text-xs px-3 py-1.5 font-medium"
              style={{
                backgroundColor: 'rgba(255,255,255,0.90)',
                color: 'var(--color-ink)',
                borderRadius: 'var(--radius-pill)',
                backdropFilter: 'blur(4px)',
              }}
            >
              {handDetected
                ? `Hand detected · ${handedness?.toLowerCase() ?? 'right'} hand`
                : 'No hand detected'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
