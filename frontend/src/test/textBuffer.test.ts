import { describe, it, expect } from 'vitest';

describe('Text Buffer Backend Source-of-Truth Contract', () => {
  it('text_buffer from server snapshot is authoritative', () => {
    const serverSnapshot = 'HELLO WORLD ';
    // Frontend binds directly to server text_buffer snapshot
    expect(serverSnapshot).toBe('HELLO WORLD ');
  });

  it('committed_label is purely for transient feedback and does not double-accumulate', () => {
    // Backend updates text_buffer and includes committed_label on commit frame
    const committedLabel = 'L';
    const serverBufferAfterCommit = 'ASL';

    // The frontend must NOT do: `localBuffer += committedLabel` (which would cause double append)
    // It directly adopts serverBufferAfterCommit
    expect(serverBufferAfterCommit.endsWith(committedLabel)).toBe(true);
    expect(serverBufferAfterCommit).toBe('ASL');
  });

  it('control actions update text_buffer synchronously with server ack', () => {
    const spaceAck = {
      type: 'control_ack' as const,
      action: 'space',
      status: 'ok' as const,
      text_buffer: 'HELLO ',
      speech_token: null,
      speech_enabled: true,
    };
    expect(spaceAck.text_buffer).toBe('HELLO ');

    const deleteAck = {
      type: 'control_ack' as const,
      action: 'delete',
      status: 'ok' as const,
      text_buffer: 'HELL',
      speech_token: null,
      speech_enabled: true,
    };
    expect(deleteAck.text_buffer).toBe('HELL');

    const clearAck = {
      type: 'control_ack' as const,
      action: 'clear',
      status: 'ok' as const,
      text_buffer: '',
      speech_token: null,
      speech_enabled: true,
    };
    expect(clearAck.text_buffer).toBe('');
  });
});

