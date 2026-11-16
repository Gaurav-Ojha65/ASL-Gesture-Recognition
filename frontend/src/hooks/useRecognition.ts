import { useCallback, useEffect, useRef, useState } from 'react';
import type { RecognitionResponse, ClientFrameMetadata } from '@/types/api';
import { captureJpegFrame } from '@/utils/frameCapture';

const TARGET_FPS = 12;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

export interface UseRecognitionProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isActive: boolean; // camera active AND websocket connected
  sendFrame: (meta: ClientFrameMetadata, jpeg: ArrayBuffer) => boolean;
  lastRecognition: RecognitionResponse | null;
}

export interface UseRecognitionReturn {
  sequenceId: number;
  isStreaming: boolean;
  fps: number;
}

export function useRecognition({
  videoRef,
  isActive,
  sendFrame,
  lastRecognition,
}: UseRecognitionProps): UseRecognitionReturn {
  const sequenceIdRef = useRef(0);
  const inFlightRef = useRef(false);
  const lastFrameTimeRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const fpsCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [fps, setFps] = useState(0);

  // When server responds, clear in-flight lock
  useEffect(() => {
    if (lastRecognition !== null) {
      inFlightRef.current = false;
    }
  }, [lastRecognition]);

  const streamLoop = useCallback((timestamp: number) => {
    if (!isActive) {
      setIsStreaming(false);
      return;
    }

    rafRef.current = requestAnimationFrame(streamLoop);

    const elapsed = timestamp - lastFrameTimeRef.current;
    if (elapsed < FRAME_INTERVAL_MS) return;
    if (inFlightRef.current) return; // backpressure — wait for server response

    const video = videoRef.current;
    if (!video || video.readyState < video.HAVE_CURRENT_DATA) return;

    lastFrameTimeRef.current = timestamp;
    const seqId = ++sequenceIdRef.current;

    captureJpegFrame(video).then((jpeg) => {
      if (!jpeg || !isActive) return;
      const meta: ClientFrameMetadata = {
        type: 'frame',
        sequence_id: seqId,
        capture_timestamp_ms: Date.now(),
      };
      const sent = sendFrame(meta, jpeg);
      if (sent) {
        inFlightRef.current = true;
        fpsCountRef.current++;
      }
    });
  }, [isActive, videoRef, sendFrame]);

  useEffect(() => {
    if (isActive) {
      setIsStreaming(true);
      inFlightRef.current = false;
      lastFrameTimeRef.current = 0;
      rafRef.current = requestAnimationFrame(streamLoop);
      fpsTimerRef.current = setInterval(() => {
        setFps(fpsCountRef.current);
        fpsCountRef.current = 0;
      }, 1000);
    } else {
      setIsStreaming(false);
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (fpsTimerRef.current) {
        clearInterval(fpsTimerRef.current);
        fpsTimerRef.current = null;
      }
      setFps(0);
    }
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (fpsTimerRef.current) {
        clearInterval(fpsTimerRef.current);
        fpsTimerRef.current = null;
      }
    };
  }, [isActive, streamLoop]);

  return {
    sequenceId: sequenceIdRef.current,
    isStreaming,
    fps,
  };
}
