import { useCallback, useEffect, useState } from 'react';
import { Copy, Trash2, Space, Volume2, VolumeX, Delete, Check } from 'lucide-react';
import type { ClientControlMessage, RecognitionResponse } from '@/types/api';

interface TextBufferPanelProps {
  textBuffer: string;
  lastRecognition: RecognitionResponse | null;
  speechEnabled: boolean;
  sendControl: (msg: ClientControlMessage) => void;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
}

export function TextBufferPanel({
  textBuffer,
  lastRecognition,
  speechEnabled,
  sendControl,
  connectionStatus,
}: TextBufferPanelProps) {
  const [copied, setCopied] = useState(false);
  const [lastCommitEffect, setLastCommitEffect] = useState<string | null>(null);

  // Transient visual feedback for character commit (does NOT accumulate locally)
  useEffect(() => {
    if (lastRecognition?.committed_label) {
      setLastCommitEffect(lastRecognition.committed_label);
      const timer = setTimeout(() => {
        setLastCommitEffect(null);
      }, 700);
      return () => clearTimeout(timer);
    }
  }, [lastRecognition?.committed_label]);

  // Browser speech synthesis: speak only discrete speech_token emitted by backend
  useEffect(() => {
    const token = lastRecognition?.speech_token;
    if (!token || !speechEnabled) return;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    try {
      window.speechSynthesis.cancel(); // cancel pending speech
      const utterance = new SpeechSynthesisUtterance(token);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch {
      // Speech synthesis fallback failure ignored gracefully
    }
  }, [lastRecognition?.speech_token, speechEnabled]);

  const handleCopy = useCallback(async () => {
    if (!textBuffer) return;
    try {
      await navigator.clipboard.writeText(textBuffer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API is blocked
      const textArea = document.createElement('textarea');
      textArea.value = textBuffer;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [textBuffer]);

  const handleSpace = useCallback(() => {
    sendControl({ type: 'space' });
  }, [sendControl]);

  const handleDelete = useCallback(() => {
    sendControl({ type: 'delete' });
  }, [sendControl]);

  const handleClear = useCallback(() => {
    sendControl({ type: 'clear' });
  }, [sendControl]);

  const handleToggleSpeech = useCallback(() => {
    sendControl({
      type: 'speech_pref',
      speech_enabled: !speechEnabled,
    });
  }, [sendControl, speechEnabled]);

  const isConnected = connectionStatus === 'connected';
  const wordCount = textBuffer.trim() ? textBuffer.trim().split(/\s+/).length : 0;
  const charCount = textBuffer.length;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/50">
        <div className="flex items-center gap-3">
          <h2 className="text-slate-100 font-semibold text-sm">Translated Text</h2>
          {lastCommitEffect && (
            <span className="text-xs px-2 py-0.5 rounded-full font-mono font-bold bg-cyan-400/20 text-cyan-400 ring-1 ring-cyan-400/40 animate-pulse">
              + {lastCommitEffect === 'SPACE' ? '␠' : lastCommitEffect === 'DELETE' ? '⌫' : lastCommitEffect}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
          <span>{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
          <span>•</span>
          <span>{charCount} chars</span>
        </div>
      </div>

      {/* Main Text Display Area */}
      <div className="p-4 flex-1 min-h-[140px] flex flex-col justify-between">
        <div
          className="w-full min-h-[90px] font-mono text-lg md:text-xl text-slate-100 whitespace-pre-wrap break-words leading-relaxed selection:bg-violet-600 selection:text-white"
          data-testid="text-buffer-display"
        >
          {textBuffer ? (
            <span>
              {textBuffer}
              <span className="inline-block w-2 h-5 ml-0.5 bg-violet-400 animate-pulse align-middle" />
            </span>
          ) : (
            <span className="text-slate-500 italic text-sm md:text-base font-sans select-none">
              Signs recognized through the camera will appear here...
            </span>
          )}
        </div>

        {/* Toolbar Controls */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-slate-800/80">
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleSpace}
              disabled={!isConnected}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-40 disabled:pointer-events-none transition-colors border border-slate-700"
              title="Add word space"
            >
              <Space size={14} />
              <span>Space</span>
            </button>
            <button
              onClick={handleDelete}
              disabled={!isConnected || charCount === 0}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-40 disabled:pointer-events-none transition-colors border border-slate-700"
              title="Backspace character"
            >
              <Delete size={14} />
              <span>Delete</span>
            </button>
            <button
              onClick={handleClear}
              disabled={!isConnected || charCount === 0}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-red-950/40 text-slate-200 hover:text-red-300 disabled:opacity-40 disabled:pointer-events-none transition-colors border border-slate-700 hover:border-red-800/50"
              title="Clear text buffer"
            >
              <Trash2 size={14} />
              <span>Clear</span>
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={handleToggleSpeech}
              disabled={!isConnected}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                speechEnabled
                  ? 'bg-violet-950/50 text-violet-300 border-violet-800 hover:bg-violet-900/60'
                  : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
              } disabled:opacity-40 disabled:pointer-events-none`}
              title={speechEnabled ? 'Mute Speech Output' : 'Enable Speech Output'}
            >
              {speechEnabled ? <Volume2 size={14} className="text-violet-400" /> : <VolumeX size={14} />}
              <span>{speechEnabled ? 'Speech On' : 'Muted'}</span>
            </button>

            <button
              onClick={handleCopy}
              disabled={charCount === 0}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                copied
                  ? 'bg-cyan-950/50 text-cyan-300 border-cyan-700'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
              } disabled:opacity-40 disabled:pointer-events-none`}
              title="Copy text to clipboard"
            >
              {copied ? <Check size={14} className="text-cyan-400" /> : <Copy size={14} />}
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

