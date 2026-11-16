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
    <div className="p-4 lg:p-6 max-w-[1600px] mx-auto min-h-[calc(100vh-3.5rem)] lg:min-h-screen flex flex-col justify-start">
      {/* Page Header */}
      <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-slate-100 tracking-tight">
            Live ASL Translator
          </h1>
          <p className="text-slate-400 text-xs md:text-sm mt-0.5">
            Real-time hand landmark recognition with temporal hysteresis confirmation
          </p>
        </div>
      </div>

      {/* Responsive Two-Column Layout */}
      <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 items-stretch flex-1">
        {/* Left Column (Webcam Feed & Text Composition) ~70% */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">
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

        {/* Right Column (Candidate Gesture, Progress Ring & Telemetry) ~30% */}
        <div className="lg:w-80 xl:w-96 flex-shrink-0 flex flex-col">
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
