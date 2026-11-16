import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('HTTP REST Client (Health & System Info)', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetchHealth returns validated HealthResponse on HTTP 200', async () => {
    const mockHealth = {
      status: 'healthy',
      timestamp_utc: 1794778709.15,
      model_loaded: true,
      detector_loaded: true,
      active_sessions: 1,
    };

    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    });

    const { fetchHealth } = await import('@/services/api');
    const res = await fetchHealth();
    expect(res.status).toBe('healthy');
    expect(res.model_loaded).toBe(true);
    expect(res.detector_loaded).toBe(true);
    expect(res.active_sessions).toBe(1);
  });

  it('fetchHealth throws descriptive error on network failure', async () => {
    (fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('Failed to fetch'));

    const { fetchHealth } = await import('@/services/api');
    await expect(fetchHealth()).rejects.toThrow('Failed to fetch');
  });

  it('fetchSystemInfo returns hardware and model architecture metadata', async () => {
    const mockSystemInfo = {
      app_name: 'ASL Gesture Studio API',
      version: '1.0.0',
      python_version: '3.12.10',
      torch_version: '2.14.0+cu132',
      cuda_available: true,
      device: 'cuda',
      model_checkpoint: 'models/asl_landmark_mlp_best.pt',
      model_param_count: 18524,
      mediapipe_version: '1.0.1',
      opencv_version: '5.0.0',
      input_dim: 63,
      num_classes: 28,
      classes: ['A', 'B', 'C', 'D', 'E', 'DELETE', 'SPACE'],
      min_confidence: 0.70,
      dwell_duration: 0.60,
      release_duration: 0.25,
    };

    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockSystemInfo,
    });

    const { fetchSystemInfo } = await import('@/services/api');
    const info = await fetchSystemInfo();
    expect(info.app_name).toBe('ASL Gesture Studio API');
    expect(info.model_param_count).toBe(18524);
    expect(info.dwell_duration).toBe(0.60);
    expect(info.min_confidence).toBe(0.70);
  });
});

