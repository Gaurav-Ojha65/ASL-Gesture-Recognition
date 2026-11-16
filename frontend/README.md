# ASL Gesture Studio — React Frontend

React + Vite + TypeScript frontend for the **ASL Gesture Studio** real-time American Sign Language recognition system.

The application interfaces directly with the asynchronous FastAPI + Uvicorn backend (`src/api/`) over a low-latency WebSocket connection, streaming live webcam frames and rendering interactive skeletal overlays, continuous text composition, and telemetry metrics.

---

## 1. System Requirements

- **Node.js**: `>= 20.x` (tested and verified on `v24.20.0`)
- **npm**: `>= 10.x` (tested and verified on `11.19.0`)
- **Browser**: Modern Chromium, Firefox, or Safari supporting:
  - `navigator.mediaDevices.getUserMedia` (Webcam video stream)
  - Standard HTML5 `<canvas>` 2D context
  - Standard WebSocket API with binary `arraybuffer` support
  - Web Speech API (`window.speechSynthesis`) for auditory speech output
- **FastAPI Backend**: Python 3.12 environment with CUDA or CPU PyTorch runtime

---

## 2. Quickstart & Installation

From the project root:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install
```

---

## 3. Environment Configuration

Copy `.env.example` to `.env.local` to override backend endpoints if needed:

```bash
cp .env.example .env.local
```

### Environment Variables

| Variable | Description | Default (Local Dev) |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Base URL for FastAPI REST endpoints (`/api/health`, `/api/system/info`) | `""` (Empty — utilizes Vite proxy) |
| `VITE_WS_BASE_URL` | Base URL for WebSocket streaming (`/ws/recognition`) | `""` (Empty — utilizes Vite proxy) |

> **Development Proxy Note:** In development, Vite's dev server automatically proxies `/api` to `http://127.0.0.1:8000` and `/ws` to `ws://127.0.0.1:8000` with WebSocket upgrade support enabled.

---

## 4. Running the Complete System

### Step A: Start the FastAPI Backend
In your primary terminal, activate the Python virtual environment and start Uvicorn:

```powershell
# From C:\project\Computer_Vision
.\.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify backend health at: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Step B: Start the React Frontend
In a separate terminal:

```bash
# From C:\project\Computer_Vision\frontend
npm run dev
```

Open your browser to: **[http://localhost:5173](http://localhost:5173)**

---

## 5. Production Build & Preview

```bash
# Type check and build production bundle
npm run build

# Preview production build locally
npm run preview
```

Production assets will be emitted into `frontend/dist/`.

---

## 6. Frontend Test Suite

The frontend test suite is powered by **Vitest** and **React Testing Library**:

```bash
# Run all tests once
npm test

# Run tests in interactive watch mode
npm run test:watch
```

### Test Coverage Highlights
- `src/test/websocket.test.ts`: Validates `recognition_result`, `control_ack`, `BACKPRESSURE_DROP`, and `STALE_SEQUENCE_DROP` parsing.
- `src/test/textBuffer.test.ts`: Verifies backend source-of-truth text synchronization and double-append prevention.
- `src/test/landmarkRenderer.test.ts`: Tests 4:3 letterboxing, aspect ratio math, and normalized `[0, 1]` landmark coordinate projection.
- `src/test/sequenceId.test.ts`: Validates monotonic sequence ID validation and jitter drops.
- `src/test/apiHealth.test.ts`: Verifies `fetchHealth` and `fetchSystemInfo` REST clients and error handling.
- `src/test/components.test.tsx`: Validates component rendering for `Sidebar`, `TextBufferPanel`, and `RecognitionPanel`.

---

## 7. WebSocket Streaming Protocol Specification

The frontend connects to `ws://127.0.0.1:8000/ws/recognition` using a dual-message interleaved protocol:

### Outbound Client Frame Protocol
For every captured video frame, the client emits two consecutive messages:
1. **Metadata Frame (JSON Text):**
   ```json
   {
     "type": "frame",
     "sequence_id": 101,
     "capture_timestamp_ms": 1794778700120
   }
   ```
2. **Binary Frame Payload (ArrayBuffer):**
   - Raw binary JPEG bytes generated from the un-mirrored `<video>` element.
   - Enforced backend limits: Payload size $\le 2\text{ MB}$, Dimensions $\le 1280 \times 720$.

### Outbound Control Commands (JSON Text)
- `{"type": "init", "speech_enabled": true}` — Initialize session preferences.
- `{"type": "space"}` — Append word space and trigger speech output for completed word.
- `{"type": "delete"}` — Backspace character.
- `{"type": "clear"}` — Clear session text buffer and reset FSM dwell state.
- `{"type": "speech_pref", "speech_enabled": false}` — Toggle auditory speech output.

### Inbound Server Responses
- `recognition_result`: Emitted once per processed frame containing `landmarks` (21 normalized 3D coordinates), `smoothed_prediction`, `smoothed_confidence`, `fsm_candidate`, `fsm_dwell_time`, `committed_label`, and authoritative `text_buffer`.
- `control_ack`: Emitted in response to control commands carrying updated `text_buffer` snapshot.
- `error`: Emitted on backpressure (`BACKPRESSURE_DROP`) or out-of-order frames (`STALE_SEQUENCE_DROP`).

---

## 8. Mirroring & Coordinate Alignment Architecture

To achieve intuitive user interaction, video and skeleton overlays follow a strict coordinate contract:
1. **Display Video:** Mirrored horizontally via CSS (`transform: scaleX(-1)`).
2. **Outbound Frame Capture:** Captured directly from the un-mirrored `<video>` element without applying canvas transforms (`frameCapture.ts`).
3. **Backend Processing:** The FastAPI backend mirrors the frame horizontally before running MediaPipe hand landmark detection.
4. **Canvas Landmark Overlay:** Landmark normalized $x$ coordinates are already aligned with the mirrored video feed. The HTML5 `<canvas>` is drawn in the unmirrored coordinate space (`landmarkRenderer.ts`), perfectly matching the user's displayed hand.

---

## 9. Troubleshooting & FAQ

- **Backend Offline Warning:** Ensure Uvicorn is active on port 8000. Verify with `curl http://127.0.0.1:8000/api/health`.
- **Camera Permission Denied:** Allow camera access in your browser settings (look for the camera icon in your address bar) and restart the camera.
- **Backpressure Drops Observed in Diagnostics:** If your device sends frames faster than the backend can process, `BACKPRESSURE_DROP` protects against queue latency buildup. The frontend automatically waits for the preceding frame response before transmitting the next.
- **Why are there no Top-3 probability charts on Diagnostics?** The backend currently returns Top-1 prediction in `RecognitionResponse`. To ensure 100% honesty with zero fabricated data, multi-class distribution graphs are omitted until a future backend schema update.

---

## 10. Project Directory Layout

```
frontend/
├── .env.example              # Template environment variables
├── index.html                # HTML entry point with Inter & JetBrains Mono fonts
├── package.json              # Dependencies and build/test scripts
├── tsconfig.app.json         # TypeScript configuration with @/* path aliases
├── vite.config.ts            # Vite bundler, Tailwind CSS v4, proxy & Vitest setup
├── src/
│   ├── App.tsx               # Root component with React Router layout
│   ├── index.css             # Tailwind CSS v4 theme variables and global styles
│   ├── main.tsx              # React DOM render entrypoint
│   ├── types/
│   │   └── api.ts            # TypeScript interfaces matching FastAPI schemas
│   ├── services/
│   │   ├── api.ts            # REST API client (fetchHealth, fetchSystemInfo)
│   │   └── websocket.ts      # Bi-directional RecognitionWebSocket service
│   ├── hooks/
│   │   ├── useWebSocket.ts   # WebSocket lifecycle and message dispatcher hook
│   │   ├── useWebcam.ts      # getUserMedia stream and track cleanup hook
│   │   └── useRecognition.ts # 10–15 FPS frame capture and backpressure loop
│   ├── utils/
│   │   ├── frameCapture.ts   # Raw un-mirrored JPEG frame encoder
│   │   └── landmarkRenderer.ts # 21-point hand skeleton canvas renderer
│   ├── components/
│   │   ├── layout/
│   │   │   └── Sidebar.tsx   # Persistent desktop sidebar & mobile drawer
│   │   ├── webcam/
│   │   │   └── WebcamPanel.tsx # Video viewport & Canvas skeleton overlay
│   │   ├── recognition/
│   │   │   └── RecognitionPanel.tsx # Candidate letter, SVG dwell ring & confidence meter
│   │   └── text-buffer/
│   │       └── TextBufferPanel.tsx  # Authoritative text display, toolbar & TTS
│   ├── pages/
│   │   ├── Dashboard.tsx     # Landing dashboard with real health & model info
│   │   ├── LiveTranslator.tsx# Main real-time translation studio (70/30 layout)
│   │   ├── ModelInsights.tsx # Pipeline explanation & MLP architecture
│   │   └── Diagnostics.tsx   # Full live telemetry inspector (no fake data)
│   └── test/
│       ├── setup.ts          # Vitest Jest-DOM matchers setup
│       ├── websocket.test.ts # Schema parsing unit tests
│       ├── landmarkRenderer.test.ts # Aspect ratio math tests
│       ├── textBuffer.test.ts # Single source of truth sync tests
│       ├── sequenceId.test.ts # Monotonic sequence ID tests
│       ├── apiHealth.test.ts # REST endpoint client tests
│       └── components.test.tsx # Component rendering tests
```
