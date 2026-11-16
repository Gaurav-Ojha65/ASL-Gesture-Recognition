import { useCallback, useEffect, useRef, useState } from 'react';
import type { CameraStatus } from '@/types/api';

export interface UseWebcamReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraStatus: CameraStatus;
  videoWidth: number;
  videoHeight: number;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  errorMessage: string | null;
}

export function useWebcam(): UseWebcamReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setVideoWidth(0);
    setVideoHeight(0);
    setCameraStatus('stopped');
  }, []);

  const startCamera = useCallback(async () => {
    setErrorMessage(null);
    setCameraStatus('requesting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setVideoWidth(videoRef.current.videoWidth);
        setVideoHeight(videoRef.current.videoHeight);
      }
      setCameraStatus('active');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg.includes('Permission') || msg.includes('NotAllowed')
        ? 'Camera permission denied. Please allow camera access and try again.'
        : `Camera error: ${msg}`);
      setCameraStatus('error');
    }
  }, []);

  // Update video dimensions after metadata loads
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const handleMeta = () => {
      setVideoWidth(video.videoWidth);
      setVideoHeight(video.videoHeight);
    };
    video.addEventListener('loadedmetadata', handleMeta);
    return () => video.removeEventListener('loadedmetadata', handleMeta);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => { stopCamera(); };
  }, [stopCamera]);

  return { videoRef, cameraStatus, videoWidth, videoHeight, startCamera, stopCamera, errorMessage };
}
