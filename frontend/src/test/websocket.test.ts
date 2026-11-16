import { describe, it, expect } from 'vitest';
import type { RecognitionResponse, ControlAckResponse, ErrorResponse } from '@/types/api';

describe('WebSocket Protocol & Schema Parsing', () => {
  it('parses valid recognition_result with all required telemetry fields', () => {
    const raw: RecognitionResponse = {
      type: 'recognition_result',
      sequence_id: 101,
      server_receive_timestamp_ms: 1794778700140,
      processing_completion_timestamp_ms: 1794778700152,
      processing_duration_ms: 11.45,
      hand_detected: true,
      handedness: 'Right',
      landmark_valid: true,
      landmarks: Array(21).fill({ x: 0.5, y: 0.5, z: 0.0 }),
      raw_prediction: 'L',
      raw_confidence: 0.9996,
      smoothed_prediction: 'L',
      smoothed_confidence: 0.9998,
      confidence_gate_passed: true,
      static_fallback_label: 'L',
      dynamic_candidate: null,
      confirmed_dynamic_result: null,
      fsm_candidate: 'L',
      fsm_is_dwelling: true,
      fsm_dwell_time: 0.35,
      committed_label: null,
      text_buffer: 'HELLO ',
      speech_token: null,
      error_status: null,
    };

    expect(raw.type).toBe('recognition_result');
    expect(raw.sequence_id).toBe(101);
    expect(raw.hand_detected).toBe(true);
    expect(raw.landmarks).toHaveLength(21);
    expect(raw.smoothed_prediction).toBe('L');
    expect(raw.fsm_dwell_time).toBe(0.35);
    expect(raw.text_buffer).toBe('HELLO ');
    expect(raw.committed_label).toBeNull();
  });

  it('handles NO_HAND frame with null landmarks and zero confidence', () => {
    const raw: RecognitionResponse = {
      type: 'recognition_result',
      sequence_id: 102,
      server_receive_timestamp_ms: 1794778700170,
      processing_completion_timestamp_ms: 1794778700174,
      processing_duration_ms: 4.12,
      hand_detected: false,
      handedness: null,
      landmark_valid: false,
      landmarks: null,
      raw_prediction: null,
      raw_confidence: 0.0,
      smoothed_prediction: null,
      smoothed_confidence: 0.0,
      confidence_gate_passed: false,
      static_fallback_label: null,
      dynamic_candidate: null,
      confirmed_dynamic_result: null,
      fsm_candidate: null,
      fsm_is_dwelling: false,
      fsm_dwell_time: 0.0,
      committed_label: null,
      text_buffer: 'HELLO ',
      speech_token: null,
      error_status: null,
    };

    expect(raw.hand_detected).toBe(false);
    expect(raw.landmarks).toBeNull();
    expect(raw.fsm_candidate).toBeNull();
    expect(raw.fsm_is_dwelling).toBe(false);
  });

  it('parses control_ack for space with speech token', () => {
    const raw: ControlAckResponse = {
      type: 'control_ack',
      action: 'space',
      status: 'ok',
      text_buffer: 'HELLO WORLD ',
      speech_token: 'WORLD',
      speech_enabled: true,
    };

    expect(raw.type).toBe('control_ack');
    expect(raw.action).toBe('space');
    expect(raw.text_buffer).toBe('HELLO WORLD ');
    expect(raw.speech_token).toBe('WORLD');
  });

  it('parses control_ack for clear with empty text_buffer', () => {
    const raw: ControlAckResponse = {
      type: 'control_ack',
      action: 'clear',
      status: 'ok',
      text_buffer: '',
      speech_token: null,
      speech_enabled: true,
    };

    expect(raw.type).toBe('control_ack');
    expect(raw.text_buffer).toBe('');
  });

  it('parses BACKPRESSURE_DROP error response', () => {
    const raw: ErrorResponse = {
      type: 'error',
      sequence_id: 105,
      message: 'Previous frame is still processing. Dropped to maintain real-time responsiveness.',
      code: 'BACKPRESSURE_DROP',
    };

    expect(raw.type).toBe('error');
    expect(raw.code).toBe('BACKPRESSURE_DROP');
    expect(raw.sequence_id).toBe(105);
  });

  it('parses STALE_SEQUENCE_DROP error response', () => {
    const raw: ErrorResponse = {
      type: 'error',
      sequence_id: 101,
      message: 'Sequence ID 101 is stale (last processed: 104).',
      code: 'STALE_SEQUENCE_DROP',
    };

    expect(raw.type).toBe('error');
    expect(raw.code).toBe('STALE_SEQUENCE_DROP');
  });
});

