import { useCallback, useEffect, useState } from 'react';
import { Copy, Check } from 'lucide-react';
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

  // Browser speech synthesis: speak only discrete speech_token emitted by backend
  useEffect(() => {
    const token = lastRecognition?.speech_token;
    if (!token || !speechEnabled) return;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
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

  const handleSpace = useCallback(() => sendControl({ type: 'space' }), [sendControl]);
  const handleDelete = useCallback(() => sendControl({ type: 'delete' }), [sendControl]);
  const handleClear = useCallback(() => sendControl({ type: 'clear' }), [sendControl]);

  const handleSpeak = useCallback(() => {
    if (!textBuffer) return;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textBuffer);
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    } catch {
      // ignored
    }
  }, [textBuffer]);

  const isConnected = connectionStatus === 'connected';
  const wordCount = textBuffer.trim() ? textBuffer.trim().split(/\s+/).length : 0;
  const charCount = textBuffer.length;

  return (
    <div
      className="bg-surface border overflow-hidden flex flex-col"
      style={{ borderColor: 'var(--color-line)', borderRadius: 'var(--radius-card)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3 border-b"
        style={{ borderColor: 'var(--color-line)' }}
      >
        <span
          className="text-xs font-semibold uppercase tracking-widest"
          style={{ color: 'var(--color-muted)', letterSpacing: '0.06em' }}
        >
          Your text
        </span>
        <span className="text-xs" style={{ color: 'var(--color-muted)' }}>
          {charCount} {charCount === 1 ? 'character' : 'characters'} · {wordCount}{' '}
          {wordCount === 1 ? 'word' : 'words'}
        </span>
      </div>

      {/* Text display */}
      <div className="px-5 pt-5 pb-4 flex-1 min-h-[120px]">
        <div
          className="min-h-[72px] break-words leading-tight text-3xl md:text-[44px] font-bold"
          data-testid="text-buffer-display"
          aria-live="polite"
          aria-label="Translated text output"
          style={{
            letterSpacing: '0.04em',
            color: 'var(--color-ink)',
          }}
        >
          {textBuffer ? (
            <span>
              {textBuffer}
              {/* Coral caret */}
              <span
                className="inline-block w-[3px] h-7 md:h-[40px] ml-1 align-middle animate-pulse"
                style={{ backgroundColor: 'var(--color-coral-text)' }}
                aria-hidden="true"
              />
            </span>
          ) : (
            <span
              className="text-base font-normal not-italic"
              style={{ color: 'var(--color-muted)' }}
            >
              Signs you make will appear here
            </span>
          )}
        </div>
      </div>

      {/* Controls */}
      <div
        className="flex items-center justify-between px-5 py-3 border-t gap-2 flex-wrap"
        style={{ borderColor: 'var(--color-line)' }}
      >
        {/* Left: editing buttons */}
        <div className="flex items-center gap-2">
          {(['Space', 'Delete', 'Clear'] as const).map((label) => {
            const disabled =
              !isConnected || (label !== 'Space' && charCount === 0);
            const handler =
              label === 'Space' ? handleSpace : label === 'Delete' ? handleDelete : handleClear;
            return (
              <button
                key={label}
                onClick={handler}
                disabled={disabled}
                className="btn-sm text-sm font-medium px-4 border transition-colors disabled:opacity-40 disabled:pointer-events-none"
                style={{
                  borderColor: 'var(--color-line)',
                  backgroundColor: 'var(--color-surface)',
                  color: 'var(--color-ink)',
                  borderRadius: 'var(--radius-button)',
                }}
                aria-label={label}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Right: Copy + Speak aloud */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            disabled={charCount === 0}
            className="btn-sm text-sm font-medium px-4 border transition-colors disabled:opacity-40 disabled:pointer-events-none flex items-center gap-1.5"
            style={{
              borderColor: 'var(--color-line)',
              backgroundColor: 'var(--color-surface)',
              color: 'var(--color-ink)',
              borderRadius: 'var(--radius-button)',
            }}
            aria-label="Copy text to clipboard"
          >
            {copied ? <Check size={14} /> : <Copy size={14} aria-hidden="true" />}
            {copied ? 'Copied' : 'Copy'}
          </button>

          <button
            onClick={handleSpeak}
            disabled={charCount === 0}
            className="btn-sm text-sm font-semibold px-5 transition-colors disabled:opacity-40 disabled:pointer-events-none"
            style={{
              backgroundColor: 'var(--color-brand)',
              color: 'var(--color-on-brand)',
              borderRadius: 'var(--radius-button)',
            }}
            aria-label="Speak text aloud"
          >
            Speak aloud
          </button>
        </div>
      </div>
    </div>
  );
}
