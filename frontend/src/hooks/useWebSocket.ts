import { useCallback, useEffect, useRef, useState } from 'react';
import { RecognitionWebSocket } from '@/services/websocket';
import type {
  WebSocketServerMessage,
  RecognitionResponse,
  ControlAckResponse,
  ClientControlMessage,
  ClientFrameMetadata,
} from '@/types/api';

export interface UseWebSocketReturn {
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  lastRecognition: RecognitionResponse | null;
  lastControlAck: ControlAckResponse | null;
  textBuffer: string;
  speechEnabled: boolean;
  connect: () => void;
  disconnect: () => void;
  sendControl: (msg: ClientControlMessage) => void;
  sendFrame: (meta: ClientFrameMetadata, jpeg: ArrayBuffer) => boolean;
}

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<RecognitionWebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [lastRecognition, setLastRecognition] = useState<RecognitionResponse | null>(null);
  const [lastControlAck, setLastControlAck] = useState<ControlAckResponse | null>(null);
  const [textBuffer, setTextBuffer] = useState('');
  const [speechEnabled, setSpeechEnabled] = useState(true);

  const handleMessage = useCallback((msg: WebSocketServerMessage) => {
    if (msg.type === 'recognition_result') {
      setLastRecognition(msg);
      setTextBuffer(msg.text_buffer);
    } else if (msg.type === 'control_ack') {
      setLastControlAck(msg);
      setTextBuffer(msg.text_buffer);
      if (msg.speech_enabled !== null && msg.speech_enabled !== undefined) {
        setSpeechEnabled(msg.speech_enabled);
      }
    }
    // 'error' type: surface to parent if needed; for now connection status handles it
  }, []);

  const handleStatus = useCallback((status: 'connected' | 'disconnected' | 'error') => {
    setConnectionStatus(status);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.disconnect();
    };
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.disconnect();
    setConnectionStatus('connecting');
    const ws = new RecognitionWebSocket(handleMessage, handleStatus);
    wsRef.current = ws;
    ws.connect();
  }, [handleMessage, handleStatus]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
    setConnectionStatus('disconnected');
  }, []);

  const sendControl = useCallback((msg: ClientControlMessage) => {
    wsRef.current?.sendControl(msg);
  }, []);

  const sendFrame = useCallback((meta: ClientFrameMetadata, jpeg: ArrayBuffer): boolean => {
    return wsRef.current?.sendFrame(meta, jpeg) ?? false;
  }, []);

  return {
    connectionStatus,
    lastRecognition,
    lastControlAck,
    textBuffer,
    speechEnabled,
    connect,
    disconnect,
    sendControl,
    sendFrame,
  };
}
