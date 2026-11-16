import type {
  ClientFrameMetadata,
  ClientControlMessage,
  WebSocketServerMessage,
} from '@/types/api';

function getWsBase(): string {
  const envBase = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  if (envBase && envBase.trim()) return envBase.trim().replace(/\/$/, '');
  if (typeof window !== 'undefined' && window.location) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    return `${proto}//${host}`;
  }
  return 'ws://127.0.0.1:8000';
}

export type WsMessageHandler = (msg: WebSocketServerMessage) => void;
export type WsStatusHandler = (status: 'connected' | 'disconnected' | 'error') => void;

export class RecognitionWebSocket {
  private ws: WebSocket | null = null;
  private onMessage: WsMessageHandler;
  private onStatus: WsStatusHandler;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private shouldReconnect = false;

  constructor(onMessage: WsMessageHandler, onStatus: WsStatusHandler) {
    this.onMessage = onMessage;
    this.onStatus = onStatus;
  }

  connect(): void {
    this.shouldReconnect = true;
    this._connect();
  }

  private _connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const url = `${getWsBase()}/ws/recognition`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.onStatus('connected');

    this.ws.onopen = () => {
      this.onStatus('connected');
    };

    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const parsed = JSON.parse(event.data) as WebSocketServerMessage;
          this.onMessage(parsed);
        } catch {
          // Malformed JSON — ignore silently
        }
      }
      // Binary messages from server not expected; ignore
    };

    this.ws.onerror = () => {
      this.onStatus('error');
    };

    this.ws.onclose = () => {
      this.onStatus('disconnected');
      if (this.shouldReconnect) {
        this.reconnectTimer = setTimeout(() => this._connect(), 3000);
      }
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendControl(msg: ClientControlMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  /**
   * Sends a webcam frame using the two-message interleaved protocol:
   * 1. JSON text metadata
   * 2. Binary JPEG bytes (ArrayBuffer or Blob)
   *
   * IMPORTANT: Never insert any other message between these two sends.
   */
  sendFrame(metadata: ClientFrameMetadata, jpegBytes: ArrayBuffer): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(metadata));
    this.ws.send(jpegBytes);
    return true;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
