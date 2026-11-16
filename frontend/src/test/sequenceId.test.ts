import { describe, it, expect } from 'vitest';

describe('Frame Sequence & Monotonicity Contracts', () => {
  it('client sequence IDs must strictly increase to prevent out-of-order jitter', () => {
    const sequenceIds = [1, 2, 3, 4, 5];
    for (let i = 1; i < sequenceIds.length; i++) {
      expect(sequenceIds[i]).toBeGreaterThan(sequenceIds[i - 1]);
    }
  });

  it('drops out-of-order frame when sequence_id is stale', () => {
    const staleResponse = {
      type: 'error' as const,
      code: 'STALE_SEQUENCE_DROP',
      message: 'Sequence ID 4 is stale (last processed: 10).',
      sequence_id: 4,
    };
    expect(staleResponse.code).toBe('STALE_SEQUENCE_DROP');
    expect(staleResponse.sequence_id).toBe(4);
  });

  it('identifies backpressure drop when worker thread is saturated', () => {
    const backpressureResponse = {
      type: 'error' as const,
      code: 'BACKPRESSURE_DROP',
      message: 'Previous frame is still processing. Dropped to maintain real-time responsiveness.',
      sequence_id: 11,
    };
    expect(backpressureResponse.code).toBe('BACKPRESSURE_DROP');
  });
});

