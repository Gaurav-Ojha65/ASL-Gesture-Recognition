import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Navbar } from '@/components/layout/Navbar';
import { TextBufferPanel } from '@/components/text-buffer/TextBufferPanel';
import { RecognitionPanel } from '@/components/recognition/RecognitionPanel';
import { WebcamPanel } from '@/components/webcam/WebcamPanel';
import { NotFound } from '@/pages/NotFound';
import * as webcamHook from '@/hooks/useWebcam';

describe('Frontend Component Tests', () => {
  it('Navbar renders branding and all navigation routes with active state', () => {
    render(
      <MemoryRouter initialEntries={['/translator']}>
        <Navbar cameraActive={true} />
      </MemoryRouter>
    );

    expect(screen.getAllByText(/ASL Gesture/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Home')).toBeInTheDocument();
    expect(screen.getByText('Translate')).toBeInTheDocument();
    expect(screen.getByText('How it works')).toBeInTheDocument();
    expect(screen.getByText('Developer tools')).toBeInTheDocument();

    const activeLink = screen.getByRole('link', { name: 'Translate' });
    expect(activeLink).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Camera on')).toBeInTheDocument();
  });

  it('TextBufferPanel renders text_buffer and triggers control actions', () => {
    const mockSendControl = vi.fn();

    render(
      <TextBufferPanel
        textBuffer="TEST MESSAGE"
        lastRecognition={null}
        speechEnabled={true}
        sendControl={mockSendControl}
        connectionStatus="connected"
      />
    );

    expect(screen.getByText(/TEST MESSAGE/)).toBeInTheDocument();

    const spaceBtn = screen.getByRole('button', { name: /Space/i });
    fireEvent.click(spaceBtn);
    expect(mockSendControl).toHaveBeenCalledWith({ type: 'space' });

    const deleteBtn = screen.getByRole('button', { name: /Delete/i });
    fireEvent.click(deleteBtn);
    expect(mockSendControl).toHaveBeenCalledWith({ type: 'delete' });

    const clearBtn = screen.getByRole('button', { name: /Clear/i });
    fireEvent.click(clearBtn);
    expect(mockSendControl).toHaveBeenCalledWith({ type: 'clear' });
  });

  it('RecognitionPanel renders candidate, formatted dwell duration, and confidence', () => {
    const mockRecognition = {
      type: 'recognition_result' as const,
      sequence_id: 1,
      server_receive_timestamp_ms: 1000,
      processing_completion_timestamp_ms: 1012,
      processing_duration_ms: 12.0,
      hand_detected: true,
      handedness: 'Right',
      landmark_valid: true,
      landmarks: Array(21).fill({ x: 0.5, y: 0.5, z: 0.0 }),
      raw_prediction: 'A',
      raw_confidence: 0.95,
      smoothed_prediction: 'A',
      smoothed_confidence: 0.98,
      confidence_gate_passed: true,
      static_fallback_label: 'A',
      dynamic_candidate: null,
      confirmed_dynamic_result: null,
      fsm_candidate: 'A',
      fsm_is_dwelling: true,
      fsm_dwell_time: 0.45,
      dwell_progress: 0.75,
      committed_label: null,
      text_buffer: '',
      speech_token: null,
      error_status: null,
    };

    const mockSystemInfo = {
      app_name: 'ASL Gesture Studio API',
      version: '1.0.0',
      python_version: '3.12.10',
      torch_version: '2.14.0',
      cuda_available: true,
      device: 'cuda',
      model_checkpoint: 'models/asl_landmark_mlp_best.pt',
      model_param_count: 18524,
      mediapipe_version: '1.0.1',
      opencv_version: '5.0.0',
      input_dim: 63,
      num_classes: 28,
      classes: ['A'],
      min_confidence: 0.70,
      dwell_duration: 0.60,
      release_duration: 0.25,
    };

    render(
      <RecognitionPanel
        recognition={mockRecognition}
        systemInfo={mockSystemInfo}
        connectionStatus="connected"
      />
    );

    expect(screen.getAllByText('A').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Hold steady… 0.45s of 0.60s/)).toBeInTheDocument();
    expect(screen.getByText(/98%/)).toBeInTheDocument();
  });

  it('RecognitionPanel renders 3 placeholder rows when top_predictions is missing or empty', () => {
    const mockRecognition = {
      type: 'recognition_result' as const,
      sequence_id: 1,
      server_receive_timestamp_ms: 1000,
      processing_completion_timestamp_ms: 1012,
      processing_duration_ms: 12.0,
      hand_detected: false,
      handedness: null,
      landmark_valid: false,
      landmarks: null,
      raw_prediction: null,
      raw_confidence: 0,
      smoothed_prediction: null,
      smoothed_confidence: 0,
      confidence_gate_passed: false,
      static_fallback_label: null,
      dynamic_candidate: null,
      confirmed_dynamic_result: null,
      fsm_candidate: null,
      fsm_is_dwelling: false,
      fsm_dwell_time: 0,
      dwell_progress: 0,
      committed_label: null,
      text_buffer: '',
      speech_token: null,
      error_status: null,
      top_predictions: [],
    };

    render(
      <RecognitionPanel
        recognition={mockRecognition}
        systemInfo={null}
        connectionStatus="connected"
      />
    );

    // Three placeholder dashes rendered in Other possibilities
    const dashes = screen.getAllByText('–');
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });

  it('WebcamPanel renders idle fallback overlay with hand guidance', () => {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;

    render(
      <WebcamPanel
        connectionStatus="connected"
        sendFrame={() => true}
        lastRecognition={null}
      />
    );
    expect(screen.getByText(/Keep your hand centered in the frame/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Enable camera/i })).toBeInTheDocument();
  });

  it('WebcamPanel renders "No hand detected" badge when camera is active and no hand in frame', () => {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;

    vi.spyOn(webcamHook, 'useWebcam').mockReturnValue({
      videoRef: { current: null },
      cameraStatus: 'active',
      videoWidth: 640,
      videoHeight: 480,
      startCamera: vi.fn(),
      stopCamera: vi.fn(),
      errorMessage: null,
    });

    render(
      <WebcamPanel
        connectionStatus="connected"
        sendFrame={() => true}
        lastRecognition={{
          type: 'recognition_result',
          sequence_id: 2,
          server_receive_timestamp_ms: 1000,
          processing_completion_timestamp_ms: 1012,
          processing_duration_ms: 12.0,
          hand_detected: false,
          handedness: null,
          landmark_valid: false,
          landmarks: null,
          raw_prediction: null,
          raw_confidence: 0,
          smoothed_prediction: null,
          smoothed_confidence: 0,
          confidence_gate_passed: false,
          static_fallback_label: null,
          dynamic_candidate: null,
          confirmed_dynamic_result: null,
          fsm_candidate: null,
          fsm_is_dwelling: false,
          fsm_dwell_time: 0,
          dwell_progress: 0,
          committed_label: null,
          text_buffer: '',
          speech_token: null,
          error_status: null,
        }}
      />
    );

    expect(screen.getByText('No hand detected')).toBeInTheDocument();
  });

  it('NotFound page renders 404 status and Return to Home link', () => {
    render(
      <MemoryRouter>
        <NotFound />
      </MemoryRouter>
    );

    expect(screen.getByText('404')).toBeInTheDocument();
    expect(screen.getByText('Page not found')).toBeInTheDocument();
    const homeLink = screen.getByRole('link', { name: /Return to Home/i });
    expect(homeLink).toBeInTheDocument();
    expect(homeLink).toHaveAttribute('href', '/');
  });
});
