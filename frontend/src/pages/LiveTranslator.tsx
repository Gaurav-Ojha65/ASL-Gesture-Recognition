import { useEffect, useState } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { WebcamPanel } from '@/components/webcam/WebcamPanel';
import { RecognitionPanel } from '@/components/recognition/RecognitionPanel';
import { TextBufferPanel } from '@/components/text-buffer/TextBufferPanel';
import { fetchSystemInfo } from '@/services/api';
import type { SystemInfoResponse } from '@/types/api';

export function LiveTranslator() {
  const ws = useWebSocket();
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);

  // Auto-connect WebSocket on page mount
  useEffect(() => {
    ws.connect();
    return () => {
      ws.disconnect();
    };
  }, []);

  // Send init control command upon WebSocket connection
  useEffect(() => {
    if (ws.connectionStatus === 'connected') {
      ws.sendControl({ type: 'init', speech_enabled: ws.speechEnabled });
    }
  }, [ws.connectionStatus]);

  // Fetch static model and perception metadata once
  useEffect(() => {
    fetchSystemInfo()
      .then((info) => setSystemInfo(info))
      .catch(() => {
        // Fallback or offline state handled gracefully
      });
  }, []);

  return (
    <div className="max-w-[1240px] mx-auto px-5 md:px-10 py-8 min-h-[calc(100vh-3.5rem)] flex flex-col justify-start">
      {/* Page Header */}
      <div className="mb-6">
        <h1
          className="text-[32px] font-bold leading-tight"
          style={{ color: 'var(--color-ink)', letterSpacing: '-0.02em' }}
        >
          Live translator
        </h1>
        <p className="text-base mt-1" style={{ color: 'var(--color-muted)' }}>
          Hold a letter steady in front of your camera and it appears as text.
        </p>
      </div>

      {/* Responsive Two-Column Layout (1.9fr / 1fr, stacked on small screens) */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.9fr_1fr] gap-6 items-start flex-1">
        {/* Left Column: Camera Feed & Text Composition */}
        <div className="flex flex-col gap-6 min-w-0">
          <WebcamPanel
            connectionStatus={ws.connectionStatus}
            sendFrame={ws.sendFrame}
            lastRecognition={ws.lastRecognition}
          />
          <TextBufferPanel
            textBuffer={ws.textBuffer}
            lastRecognition={ws.lastRecognition}
            speechEnabled={ws.speechEnabled}
            sendControl={ws.sendControl}
            connectionStatus={ws.connectionStatus}
          />
        </div>

        {/* Right Column: Now Reading, Other Possibilities, For Best Results */}
        <div className="flex flex-col gap-6 min-w-0">
          <RecognitionPanel
            recognition={ws.lastRecognition}
            systemInfo={systemInfo}
            connectionStatus={ws.connectionStatus}
          />
        </div>
      </div>
    </div>
  );
}
