"""
Browser-Based ASL Gesture Recognition Application (Gradio Web Interface).

Integrates:
- Real-time browser webcam streaming via Gradio (gr.Image streaming)
- Reusable post-perception recognition pipeline (ASLRealTimePipeline)
- Shared PyTorch MLP inference engine & MediaPipe HandLandmarker
- Invariant anatomical geometric disambiguator (resolving M/N/T, V/U/K, F/B, X/D, P/Q, Y)
- Dynamic trajectory kinematic detector for moving ASL letters 'J' and 'Z'
- Temporal gesture hysteresis state machine (0.40s dwell confirmation)
- Client-side browser Web Speech API synthesis (replacing Windows SAPI5 for cross-platform/cloud)
- Multi-session isolation with dedicated state per browser tab
- Modern, accessible UI designed for academic/college demonstrations and Hugging Face Spaces
"""

import os
import sys
import time
import base64
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.realtime.app import ASLRealTimePipeline, FrameProcessingResult
from src.realtime.inference import ASLInferenceEngine
from src.realtime.gesture_state_machine import CommitEvent
from src.landmarks.labels import SEMANTIC_CLASSES


# ─── Global Shared Model Engine ───────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "models", "asl_landmark_mlp_best.pt")
TASK_MODEL_PATH = os.path.join(REPO_ROOT, "hand_landmarker.task")

# Shared inference engine: Thread-safe, evaluated with torch.no_grad()
SHARED_ENGINE = ASLInferenceEngine(checkpoint_path=CHECKPOINT_PATH)


# ─── Per-Session State ────────────────────────────────────────────────────────
class WebSessionState:
    """Encapsulates independent pipeline and interaction state for one browser tab."""

    def __init__(self):
        self.pipeline = ASLRealTimePipeline(
            checkpoint_path=CHECKPOINT_PATH,
            task_model_path=TASK_MODEL_PATH,
            dwell_duration=0.40,
            release_duration=0.25,
            min_confidence=0.70,
            inference_engine=SHARED_ENGINE,
            enable_dynamic=True,
            enable_speech=False,            # Server-side SAPI5 disabled for web
            enable_disambiguation=True,
            mirror_feed=True,
        )
        self.speech_enabled: bool = True
        self.last_emitted_speech_token: str = ""
        self.last_speech_timestamp: float = 0.0
        self.pause_check_timestamp: float = 0.0
        self.last_spoken_word: str = ""

    def reset(self):
        """Clears text buffer and resets state machine."""
        self.pipeline.text_buffer.clear()
        self.pipeline.state_machine.reset()
        self.last_spoken_word = ""


def create_session() -> WebSessionState:
    """Factory creating an isolated session instance per user connection."""
    return WebSessionState()


# ─── Client-Side Web Speech & WebCam JavaScript ──────────────────────────────
BROWSER_SPEECH_HEAD_JS = """
<script>
window.aslSpeechQueue = [];
window.aslSpeaking = false;
window.aslSessionId = 'sess_' + Math.random().toString(36).substring(2, 12);
window.aslCameraActive = false;
window.aslStream = null;
window.aslFrameInFlight = false;

function processASLSpeechQueue() {
    if (window.aslSpeaking || window.aslSpeechQueue.length === 0) return;
    if (!('speechSynthesis' in window)) return;

    const text = window.aslSpeechQueue.shift();
    if (!text || text.trim() === '') {
        processASLSpeechQueue();
        return;
    }

    window.aslSpeaking = true;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onend = function() {
        window.aslSpeaking = false;
        setTimeout(processASLSpeechQueue, 60);
    };
    utterance.onerror = function() {
        window.aslSpeaking = false;
        setTimeout(processASLSpeechQueue, 60);
    };

    window.speechSynthesis.speak(utterance);
}

function speakASL(text) {
    if (!text || !('speechSynthesis' in window)) return;
    window.aslSpeechQueue.push(text);
    processASLSpeechQueue();
}
window.speakASL = speakASL;

// ─── Browser-Native Webcam Capture & Streaming ───────────────────────────────
async function startASLCamera() {
    console.log("[ASL WebCam] User clicked Start Camera. Requesting getUserMedia...");
    if (window.aslCameraActive) {
        console.log("[ASL WebCam] Camera is already active.");
        return;
    }

    const startBtn = document.querySelector("#btn-start-camera button") || document.querySelector("#btn-start-camera");
    if (startBtn) startBtn.innerText = "⏳ Initializing...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 24, max: 30 }
            },
            audio: false
        });

        console.log("[ASL WebCam] Camera permission granted successfully.");
        window.aslStream = stream;
        window.aslCameraActive = true;

        let video = document.getElementById("asl-hidden-video");
        if (!video) {
            video = document.createElement("video");
            video.id = "asl-hidden-video";
            video.autoplay = true;
            video.playsInline = true;
            video.muted = true;
            video.style.display = "none";
            document.body.appendChild(video);
        }
        video.srcObject = stream;
        await video.play();

        console.log(`[ASL WebCam] Video active: ${video.videoWidth}x${video.videoHeight}px. Starting render loop...`);

        if (startBtn) startBtn.innerText = "● Camera Active";

        const canvas = document.getElementById("asl-live-display");
        const placeholder = document.getElementById("asl-camera-placeholder");
        if (canvas) canvas.style.display = "block";
        if (placeholder) placeholder.style.display = "none";

        startFrameLoop();
    } catch (err) {
        console.error("[ASL WebCam] Camera initialization error:", err);
        if (startBtn) startBtn.innerText = "📷 Start Camera";
        alert("Camera error: " + err.name + " - " + err.message + "\\nPlease ensure camera permissions are allowed in your browser.");
    }
}

function stopASLCamera() {
    console.log("[ASL WebCam] Stop Camera clicked.");
    window.aslCameraActive = false;
    if (window.aslStream) {
        window.aslStream.getTracks().forEach(track => track.stop());
        window.aslStream = null;
    }
    const startBtn = document.querySelector("#btn-start-camera button") || document.querySelector("#btn-start-camera");
    if (startBtn) startBtn.innerText = "📷 Start Camera";

    const canvas = document.getElementById("asl-live-display");
    const placeholder = document.getElementById("asl-camera-placeholder");
    if (canvas) canvas.style.display = "none";
    if (placeholder) placeholder.style.display = "block";
}

function startFrameLoop() {
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = 640;
    offscreenCanvas.height = 480;
    const offCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

    const displayCanvas = document.getElementById("asl-live-display");
    const displayCtx = displayCanvas ? displayCanvas.getContext("2d") : null;

    let lastFrameTime = 0;
    const targetInterval = 55; // ~18 FPS target

    async function captureStep(now) {
        if (!window.aslCameraActive) return;

        const video = document.getElementById("asl-hidden-video");
        if (video && video.readyState >= 2) {
            if (now - lastFrameTime >= targetInterval && !window.aslFrameInFlight) {
                lastFrameTime = now;
                window.aslFrameInFlight = true;

                offCtx.drawImage(video, 0, 0, 640, 480);
                offscreenCanvas.toBlob(async (blob) => {
                    if (!blob || !window.aslCameraActive) {
                        window.aslFrameInFlight = false;
                        return;
                    }
                    try {
                        const resp = await fetch(`/api/process_frame?session_id=${window.aslSessionId}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/octet-stream" },
                            body: blob
                        });
                        if (resp.ok) {
                            const data = await resp.json();
                            if (data.annotated_frame && displayCtx) {
                                const img = new Image();
                                img.onload = () => {
                                    displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
                                };
                                img.src = data.annotated_frame;
                            }
                            updateUI(data);
                        }
                    } catch (e) {
                        console.error("[ASL WebCam] Frame post error:", e);
                    } finally {
                        window.aslFrameInFlight = false;
                    }
                }, "image/jpeg", 0.70);
            }
        }
        requestAnimationFrame(captureStep);
    }
    requestAnimationFrame(captureStep);
}

function updateUI(data) {
    const charBadge = document.querySelector("#char-display input") || document.querySelector("#char-display textarea");
    if (charBadge && data.pred_label !== undefined) charBadge.value = data.pred_label;

    const confBadge = document.querySelector("#conf-display input");
    if (confBadge && data.confidence !== undefined) confBadge.value = data.confidence;

    const statusBadge = document.querySelector("#status-display input");
    if (statusBadge && data.status !== undefined) statusBadge.value = data.status;

    const textDisplay = document.querySelector("#text-display textarea") || document.querySelector("#text-display input");
    if (textDisplay && data.text !== undefined) textDisplay.value = data.text;

    if (data.speech && data.speech.trim() !== '') {
        window.speakASL(data.speech.trim());
    }
}

async function sendASLAction(action) {
    try {
        const resp = await fetch(`/api/action?action=${action}&session_id=${window.aslSessionId}`, { method: "POST" });
        if (resp.ok) {
            const data = await resp.json();
            const textDisplay = document.querySelector("#text-display textarea") || document.querySelector("#text-display input");
            if (textDisplay && data.text !== undefined) textDisplay.value = data.text;
            if (data.speech) window.speakASL(data.speech);
            if (data.speech_toggle_text) {
                const btn = document.querySelector("#speech-toggle-btn button") || document.querySelector("#speech-toggle-btn");
                if (btn) btn.innerText = data.speech_toggle_text;
            }
        }
    } catch (err) {
        console.error("[ASL WebCam] Action error:", err);
    }
}

window.startASLCamera = startASLCamera;
window.stopASLCamera = stopASLCamera;
window.sendASLAction = sendASLAction;

// Bind physical keyboard controls matching desktop OpenCV app
window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;

    if (e.code === 'Space') {
        e.preventDefault();
        sendASLAction('space');
    } else if (e.code === 'Backspace' || e.code === 'Delete') {
        e.preventDefault();
        sendASLAction('delete');
    } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        sendASLAction('clear');
    } else if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        sendASLAction('toggle_speech');
    }
});
</script>
"""

CUSTOM_CSS = """
/* Responsive College Project Demonstration Theme */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

#header-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

#header-banner h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

#header-banner p {
    margin: 6px 0 0 0;
    font-size: 15px;
    opacity: 0.9;
}

.prediction-box {
    background: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.char-badge {
    font-size: 48px;
    font-weight: 800;
    color: #2563eb;
    margin: 8px 0;
}

.char-badge input, .char-badge textarea {
    font-size: 44px !important;
    font-weight: 800 !important;
    color: #2563eb !important;
    text-align: center !important;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
}

#asl-single-window-container {
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

#asl-single-window-container:hover {
    border-color: #60a5fa !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
}
"""


# ─── Frame Processing Callback ────────────────────────────────────────────────
_FRAME_COUNTER: int = 0


def resolve_session(session: Any) -> WebSessionState:
    """Ensures session is a valid WebSessionState instance, instantiating if callable or None."""
    if callable(session):
        return session()
    if isinstance(session, WebSessionState):
        return session
    return create_session()


def process_webcam_frame(
    rgb_frame: Optional[np.ndarray],
    session: Any
) -> Tuple[Optional[np.ndarray], str, str, str, str, str, WebSessionState]:
    """
    Receives an RGB webcam frame from the browser, executes the recognition pipeline,
    and returns annotated video, telemetry, text, and browser speech tokens.
    """
    global _FRAME_COUNTER
    _FRAME_COUNTER += 1
    fid = _FRAME_COUNTER

    session = resolve_session(session)

    try:
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
            current_text = session.pipeline.text_buffer.get_text()
            print(f"[WebCam #{fid}] Input: None or empty frame. Waiting for browser webcam...")
            return None, "-", "0.0%", "WAITING FOR WEBCAM", current_text, "", session

        h, w = rgb_frame.shape[:2]
        c = rgb_frame.shape[2] if rgb_frame.ndim > 2 else 1
        print(f"[WebCam #{fid}] Input frame: shape=({h}, {w}, {c}), dtype={rgb_frame.dtype}")

        now = time.time()

        # 1. Convert incoming Gradio RGB frame to OpenCV BGR format
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 2. Execute existing, verified recognition pipeline
        t0 = time.time()
        result: FrameProcessingResult = session.pipeline.process_frame(
            frame_bgr,
            timestamp_sec=now
        )
        proc_ms = (time.time() - t0) * 1000

        # 3. Convert annotated result frame back to RGB for web browser display
        annotated_rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)

        # 4. Telemetry data
        pred_label = result.prediction_label or "-"
        conf_str = f"{result.confidence * 100:.1f}%" if result.prediction_label else "0.0%"
        status = result.status
        current_text = session.pipeline.text_buffer.get_text()

        print(
            f"[WebCam #{fid}] Pipeline ({proc_ms:.1f}ms, {result.fps:.1f} FPS): "
            f"Hand={result.hand_detected} | "
            f"Pred='{pred_label}' ({conf_str}) | "
            f"Status={status} | "
            f"Out shape={annotated_rgb.shape}"
        )

        # 5. Dual-Mode Speech Token Arbitration
        speech_token = ""
        if session.speech_enabled:
            # Event A: A character or dynamic gesture was committed this frame
            if result.commit_event is not None and result.commit_event.label:
                committed_label = result.commit_event.label
                if committed_label == "SPACE":
                    # Finalize and speak completed word on SPACE
                    words = current_text.strip().split()
                    if words:
                        speech_token = words[-1]
                        session.last_spoken_word = speech_token
                        print(f"[WebCam #{fid}] Speech: Word completed via SPACE -> '{speech_token}'")
                elif committed_label == "DELETE":
                    pass  # Do not speak deletes
                else:
                    # Speak committed individual letter
                    speech_token = committed_label
                    session.last_speech_timestamp = now
                    print(f"[WebCam #{fid}] Speech: Letter committed -> '{speech_token}'")

            # Event B: Natural conversational pause (2.5s) on completed unspaced word
            elif current_text.strip():
                words = current_text.strip().split()
                if words:
                    candidate_word = words[-1]
                    if candidate_word != session.last_spoken_word:
                        if session.last_speech_timestamp > 0 and (now - session.last_speech_timestamp) >= 2.5:
                            speech_token = candidate_word
                            session.last_spoken_word = candidate_word
                            print(f"[WebCam #{fid}] Speech: Word completed via Pause -> '{speech_token}'")

        return annotated_rgb, pred_label, conf_str, status, current_text, speech_token, session

    except Exception as exc:
        import traceback
        print(f"[WebCam #{fid}] EXCEPTION in process_webcam_frame: {exc}")
        traceback.print_exc()
        current_text = session.pipeline.text_buffer.get_text() if hasattr(session, "pipeline") else ""
        return None, "-", "0.0%", f"ERROR: {exc}", current_text, "", session


# ─── Toolbar Action Callbacks ──────────────────────────────────────────────────
def on_space_clicked(session: Any) -> Tuple[str, str, WebSessionState]:
    """Appends SPACE, finalizes word, and speaks completed word."""
    session = resolve_session(session)
    now = time.time()
    current_text = session.pipeline.text_buffer.get_text()
    words = current_text.strip().split()
    last_word = words[-1] if words else ""

    session.pipeline.text_buffer.apply("SPACE")
    updated_text = session.pipeline.text_buffer.get_text()

    speech_token = ""
    if session.speech_enabled and last_word:
        speech_token = last_word
        session.last_spoken_word = last_word

    return updated_text, speech_token, session


def on_delete_clicked(session: Any) -> Tuple[str, WebSessionState]:
    """Deletes previous character from text buffer."""
    session = resolve_session(session)
    session.pipeline.text_buffer.apply("DELETE")
    return session.pipeline.text_buffer.get_text(), session


def on_clear_clicked(session: Any) -> Tuple[str, WebSessionState]:
    """Clears text buffer."""
    session = resolve_session(session)
    session.reset()
    return "", session


def on_toggle_speech(session: Any) -> Tuple[str, WebSessionState]:
    """Toggles browser audio synthesis on/off."""
    session = resolve_session(session)
    session.speech_enabled = not session.speech_enabled
    state_str = "🔊 Speech Audio: ON" if session.speech_enabled else "🔇 Speech Audio: MUTED"
    return state_str, session


# ─── Global Multi-Session State Registry ───────────────────────────────────────
SESSION_REGISTRY: Dict[str, WebSessionState] = {}


def get_web_session(session_id: str) -> WebSessionState:
    """Retrieves or creates an isolated session state for a browser connection."""
    if not session_id:
        session_id = "default"
    if session_id not in SESSION_REGISTRY:
        SESSION_REGISTRY[session_id] = create_session()
    return SESSION_REGISTRY[session_id]


# ─── Gradio App Construction ──────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    """Constructs the interactive Gradio web application."""
    with gr.Blocks(title="ASL Gesture Recognition | AI Sign Language Interpreter") as demo:
        # Per-session pipeline state for compatibility with direct Gradio callbacks
        session_state = gr.State(create_session)

        # Invisible carrier transporting spoken text to client-side Web Speech JS
        speech_carrier = gr.Textbox(visible=False, elem_id="speech_carrier")

        # Header Banner
        gr.HTML("""
        <div id="header-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1>ASL Gesture Recognition & Speech Translation System</h1>
                    <p>Real-Time American Sign Language Fingerspelling & Dynamic Gesture Recognition via MediaPipe & PyTorch MLP</p>
                </div>
                <div style="margin-top: 10px;">
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px; font-size: 13px; margin-right: 6px;">26 Letters (A-Z)</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px; font-size: 13px; margin-right: 6px;">Dynamic J & Z</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px; font-size: 13px;">Web Speech Audio</span>
                </div>
            </div>
        </div>
        """)

        # Main Interface Row
        with gr.Row():
            # Left Column: Single Live Video Feed & Wireframe HUD
            with gr.Column(scale=5):
                with gr.Row():
                    gr.Markdown("### 📷 Live Camera Feed & Landmark HUD")
                    start_cam_btn = gr.Button(
                        "📷 Start Camera",
                        variant="primary",
                        size="sm",
                        elem_id="btn-start-camera",
                        scale=0,
                        min_width=120
                    )
                    stop_cam_btn = gr.Button(
                        "⏹ Stop Camera",
                        variant="secondary",
                        size="sm",
                        elem_id="btn-stop-camera",
                        scale=0,
                        min_width=120
                    )

                # Exactly ONE visible video display panel
                gr.HTML("""
                <div id="asl-single-window-container" style="position: relative; width: 100%; height: 480px; background: #0f172a; border-radius: 12px; overflow: hidden; border: 2px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center;">
                    <canvas id="asl-live-display" width="640" height="480" style="width: 100%; height: 100%; object-fit: contain; display: none;"></canvas>
                    <div id="asl-camera-placeholder" style="color: #94a3b8; text-align: center; padding: 24px;">
                        <div style="font-size: 56px; margin-bottom: 12px; opacity: 0.85;">📷</div>
                        <div style="font-size: 20px; font-weight: 700; color: #f8fafc;">Camera Inactive</div>
                        <div style="font-size: 14px; margin-top: 8px; color: #94a3b8; max-width: 340px; line-height: 1.4;">Click <b>Start Camera</b> above to begin real-time ASL recognition and landmark tracking</div>
                    </div>
                </div>
                """)

            # Right Column: Recognition Telemetry & Text Assembly
            with gr.Column(scale=4):
                gr.Markdown("### 🔍 Real-Time Classification")

                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**Predicted Sign:**")
                            char_display = gr.Textbox(
                                value="-",
                                interactive=False,
                                show_label=False,
                                elem_id="char-display",
                                elem_classes=["char-badge"],
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("**Confidence:**")
                            conf_display = gr.Textbox(
                                value="0.0%",
                                interactive=False,
                                show_label=False,
                                elem_id="conf-display"
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("**Status:**")
                            status_display = gr.Textbox(
                                value="WAITING",
                                interactive=False,
                                show_label=False,
                                elem_id="status-display"
                            )

                gr.Markdown("### 📝 Accumulated Text Buffer")
                text_display = gr.Textbox(
                    value="",
                    label="Recognized Words & Sentences",
                    placeholder="Accumulated recognized letters will appear here...",
                    lines=3,
                    interactive=False,
                    elem_id="text-display",
                )

                # Action Buttons Toolbar
                with gr.Row():
                    space_btn = gr.Button("␣ SPACE (Complete Word)", variant="primary", scale=2, elem_id="space-btn")
                    delete_btn = gr.Button("⌫ DELETE", variant="secondary", scale=1, elem_id="delete-btn")
                    clear_btn = gr.Button("🗑 CLEAR", variant="secondary", scale=1, elem_id="clear-btn")

                with gr.Row():
                    speech_toggle_btn = gr.Button("🔊 Speech Audio: ON", variant="secondary", scale=1, elem_id="speech-toggle-btn")

                gr.Markdown("---")
                with gr.Accordion("📖 Quick ASL Fingerspelling Reference & Guide", open=False):
                    gr.Markdown("""
                    **Static Letters (Hold for 0.40s):**
                    - **A to Z:** Show hand clearly facing the camera.
                    - **F:** Touch index and thumb (circle), other 3 fingers upright.
                    - **V vs. U:** Spread index and middle wide for **V**; keep them touching for **U**.
                    - **M / N / T:** Thumb under 3 fingers (**M**), under 2 (**N**), under 1 (**T**).
                    - **P / Q:** Hand pointed downward.
                    - **X:** Index finger hooked like a pirate hook.
                    - **Y:** Shaka sign (thumb and pinky extended, middle 3 curled).

                    **Dynamic Gestures (Move in air):**
                    - **J:** Start in pinky upright ('I' shape) and scoop a downward 'J' hook.
                    - **Z:** Start with index finger ('1/D' shape) and trace a 3-stroke 'Z' zig-zag.

                    **Voice & Audio:**
                    - Committed letters speak immediately (e.g. "C", "A", "T").
                    - Press **SPACE** or pause for 2.5s to speak the completed word (e.g. "CAT").
                    """)

        # ─── Event Wiring ─────────────────────────────────────────────────────

        # Client-side camera controls triggered by explicit user clicks
        start_cam_btn.click(fn=None, js="() => { window.startASLCamera(); }")
        stop_cam_btn.click(fn=None, js="() => { window.stopASLCamera(); }")

        # Toolbar controls (connected to both Python state and client JS)
        space_btn.click(
            fn=on_space_clicked,
            inputs=[session_state],
            outputs=[text_display, speech_carrier, session_state],
            js="() => { window.sendASLAction('space'); }"
        )

        delete_btn.click(
            fn=on_delete_clicked,
            inputs=[session_state],
            outputs=[text_display, session_state],
            js="() => { window.sendASLAction('delete'); }"
        )

        clear_btn.click(
            fn=on_clear_clicked,
            inputs=[session_state],
            outputs=[text_display, session_state],
            js="() => { window.sendASLAction('clear'); }"
        )

        speech_toggle_btn.click(
            fn=on_toggle_speech,
            inputs=[session_state],
            outputs=[speech_toggle_btn, session_state],
            js="() => { window.sendASLAction('toggle_speech'); }"
        )

        # Browser Client-Side Audio Hook: Triggers window.speakASL(val) whenever carrier updates
        speech_carrier.change(
            fn=None,
            inputs=[speech_carrier],
            js="(val) => { if (val && val.trim() !== '') { window.speakASL(val.trim()); } }"
        )

    # ─── FastAPI Streaming & Action Endpoints (Registered outside with gr.Blocks) ───
    @demo.app.post("/api/process_frame")
    async def api_process_frame(request: Request):
        session_id = request.query_params.get("session_id", "default")
        session = get_web_session(session_id)
        raw_bytes = await request.body()

        if not raw_bytes or len(raw_bytes) < 32:
            current_text = session.pipeline.text_buffer.get_text()
            return JSONResponse({
                "annotated_frame": "",
                "pred_label": "-",
                "confidence": "0.0%",
                "status": "WAITING FOR WEBCAM",
                "text": current_text,
                "speech": "",
                "fps": 0.0,
            })

        arr = np.frombuffer(raw_bytes, np.uint8)
        bgr_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr_frame is None or bgr_frame.size == 0:
            current_text = session.pipeline.text_buffer.get_text()
            return JSONResponse({
                "annotated_frame": "",
                "pred_label": "-",
                "confidence": "0.0%",
                "status": "INVALID_IMAGE",
                "text": current_text,
                "speech": "",
                "fps": 0.0,
            })

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        annotated_rgb, pred_label, conf_str, status, current_text, speech_token, session = process_webcam_frame(
            rgb_frame, session
        )

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64_frame = base64.b64encode(enc).decode("ascii")

        return JSONResponse({
            "annotated_frame": f"data:image/jpeg;base64,{b64_frame}",
            "pred_label": pred_label,
            "confidence": conf_str,
            "status": status,
            "text": current_text,
            "speech": speech_token,
            "fps": round(getattr(session.pipeline, "_fps_filtered", 0.0), 1),
        })

    @demo.app.post("/api/action")
    async def api_action(request: Request):
        action = request.query_params.get("action", "")
        session_id = request.query_params.get("session_id", "default")
        session = get_web_session(session_id)

        if action == "space":
            text, speech, session = on_space_clicked(session)
            return JSONResponse({"text": text, "speech": speech})
        elif action == "delete":
            text, session = on_delete_clicked(session)
            return JSONResponse({"text": text})
        elif action == "clear":
            text, session = on_clear_clicked(session)
            return JSONResponse({"text": text})
        elif action == "toggle_speech":
            btn_text, session = on_toggle_speech(session)
            return JSONResponse({"speech_toggle_text": btn_text, "enabled": session.speech_enabled})

        return JSONResponse({"status": "unknown_action"})

    from gradio.routes import App
    demo.app = App.create_app(demo, app=demo.app)
    return demo


def main():
    """Local application entrypoint."""
    demo = build_app()
    print("=" * 70)
    print("Starting ASL Gesture Recognition Web Interface (Gradio)...")
    print("Local URL: http://127.0.0.1:7860")
    print("=" * 70)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        head=BROWSER_SPEECH_HEAD_JS,
        css=CUSTOM_CSS,
        share=False,
        _app=demo.app,
    )


if __name__ == "__main__":
    main()
