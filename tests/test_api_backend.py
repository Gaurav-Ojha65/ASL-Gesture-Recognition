"""
Comprehensive Test Suite for FastAPI ASL Gesture Studio Backend.

Verifies:
1. Health and readiness endpoints (/api/health).
2. System information endpoint (/api/system/info).
3. JPEG payload validation and size limits (>2MB rejected, corrupt rejected, >1280x720 rejected).
4. Correct 63-feature and 28-output contracts for stateless inference.
5. No-hand handling and non-retention of stale predictions.
6. WebSocket connection initialization, control actions (SPACE, DELETE, CLEAR, speech toggle).
7. Frame sequence handling and monotonic sequence ID validation.
8. Multi-session isolation: two simultaneous WebSocket sessions with independent state.
9. Independent smoothing histories between sessions.
10. Independent FSM dwell and TextBuffers across sessions.
11. Disconnect cleanup and bounded session registry.
12. Backpressure handling and non-blocking worker execution.
13. Real hand image recognition over WebSocket.
14. Graceful resource cleanup on application shutdown.
"""

import time
import json
from pathlib import Path
import numpy as np
import cv2
import pytest
from starlette.testclient import TestClient

from src.api.main import app, create_app
from src.api.schemas import (
    HealthResponse,
    SystemInfoResponse,
    RecognitionResponse,
    ControlAckResponse,
    ErrorResponse
)
from src.api.model_service import SharedModelService
from src.api.session_manager import SessionManager, UserSessionContext
from src.api.recognition_service import RecognitionService
from src.landmarks.labels import SEMANTIC_CLASSES


# ─── 1. Health Endpoint Tests ─────────────────────────────────────────────────

def test_health_endpoint():
    """Verify /api/health returns 200 with model, detector, and session telemetry."""
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        validated = HealthResponse.model_validate(data)
        assert validated.status == "healthy"
        assert validated.model_loaded is True
        assert validated.detector_loaded is True
        assert validated.active_sessions >= 0


# ─── 2. System Information Endpoint Tests ─────────────────────────────────────

def test_system_info_endpoint():
    """Verify /api/system/info returns accurate hardware, model, and vocabulary metadata."""
    with TestClient(app) as client:
        resp = client.get("/api/system/info")
        assert resp.status_code == 200
        data = resp.json()
        validated = SystemInfoResponse.model_validate(data)
        assert validated.app_name == "ASL Gesture Studio API"
        assert validated.num_classes == 28
        assert validated.classes == SEMANTIC_CLASSES
        assert validated.input_dim == 63
        assert validated.model_param_count == 18524
        assert validated.min_confidence == 0.70
        assert validated.dwell_duration == 0.60
        assert validated.release_duration == 0.25


# ─── 3. JPEG Validation and Size Limit Tests ──────────────────────────────────

def test_jpeg_validation_and_size_limits():
    """Verify recognition service rejects >2MB payloads, corrupt data, and oversized dimensions."""
    model_service = SharedModelService()
    model_service.load()
    session = UserSessionContext("test_validation_sess")

    try:
        # A. Empty payload
        with pytest.raises(ValueError, match="Empty frame payload"):
            RecognitionService.process_frame(b"", session, model_service, 1, int(time.time() * 1000))

        # B. Payload exceeding 2MB
        oversized_bytes = b"x" * (2 * 1024 * 1024 + 10)
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            RecognitionService.process_frame(oversized_bytes, session, model_service, 1, int(time.time() * 1000))

        # C. Corrupt non-JPEG bytes
        corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_BYTES_1234567890"
        with pytest.raises(ValueError, match="Corrupt or invalid JPEG"):
            RecognitionService.process_frame(corrupt_bytes, session, model_service, 1, int(time.time() * 1000))

        # D. Oversized dimensions (>1280x720, e.g. 1920x1080)
        oversized_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, enc_oversized = cv2.imencode(".jpg", oversized_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        with pytest.raises(ValueError, match="exceed maximum allowed cap"):
            RecognitionService.process_frame(enc_oversized.tobytes(), session, model_service, 1, int(time.time() * 1000))

    finally:
        session.cleanup()
        model_service.close()


# ─── 4. Correct 63-Feature and 28-Output Contracts ────────────────────────────

def test_stateless_model_service_contracts():
    """Verify stateless inference preserves 63-D input and 28-class output contracts."""
    model_service = SharedModelService()
    model_service.load()

    try:
        dummy_feat = np.zeros((63,), dtype=np.float32)
        raw_probs, logits, top1_idx, top1_label, top1_conf = model_service.predict_stateless(dummy_feat)

        assert raw_probs.shape == (28,)
        assert logits.shape == (28,)
        assert np.isclose(raw_probs.sum(), 1.0, atol=1e-5)
        assert 0 <= top1_idx < 28
        assert top1_label in SEMANTIC_CLASSES
        assert 0.0 <= top1_conf <= 1.0

        # Verify invalid dimensions raise ValueError
        with pytest.raises(ValueError):
            model_service.predict_stateless(np.zeros((62,), dtype=np.float32))

        # Verify non-finite values raise ValueError
        nan_feat = np.zeros((63,), dtype=np.float32)
        nan_feat[5] = np.nan
        with pytest.raises(ValueError):
            model_service.predict_stateless(nan_feat)

    finally:
        model_service.close()


# ─── 5. No-Hand Handling ──────────────────────────────────────────────────────

def test_websocket_no_hand_handling():
    """Verify blank frame returns hand_detected=False and does not retain stale predictions."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws:
            # Metadata
            ws.send_text(json.dumps({"type": "frame", "sequence_id": 1, "capture_timestamp_ms": 1000}))

            # Binary blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, enc = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            ws.send_bytes(enc.tobytes())

            resp_text = ws.receive_text()
            data = json.loads(resp_text)
            assert data["type"] == "recognition_result"
            assert data["sequence_id"] == 1
            assert data["hand_detected"] is False
            assert data["landmark_valid"] is False
            assert data["raw_prediction"] is None
            assert data["smoothed_prediction"] is None
            assert data["fsm_is_dwelling"] is False
            assert data["dwell_progress"] == 0.0
            assert data.get("top_predictions") is None or data.get("top_predictions") == []


# ─── 6. WebSocket Initialization and Control Actions ──────────────────────────

def test_websocket_control_actions():
    """Verify WebSocket control commands: init, space, delete, clear, speech_pref."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws:
            # 1. Init
            ws.send_text(json.dumps({"type": "init", "speech_enabled": True}))
            r1 = json.loads(ws.receive_text())
            assert r1["type"] == "control_ack"
            assert r1["action"] == "init"
            assert r1["speech_enabled"] is True

            # 2. Space on empty
            ws.send_text(json.dumps({"type": "space"}))
            r2 = json.loads(ws.receive_text())
            assert r2["type"] == "control_ack"
            assert r2["action"] == "space"

            # 3. Delete
            ws.send_text(json.dumps({"type": "delete"}))
            r3 = json.loads(ws.receive_text())
            assert r3["type"] == "control_ack"
            assert r3["action"] == "delete"

            # 4. Clear
            ws.send_text(json.dumps({"type": "clear"}))
            r4 = json.loads(ws.receive_text())
            assert r4["type"] == "control_ack"
            assert r4["action"] == "clear"
            assert r4["text_buffer"] == ""

            # 5. Toggle speech pref
            ws.send_text(json.dumps({"type": "speech_pref", "speech_enabled": False}))
            r5 = json.loads(ws.receive_text())
            assert r5["type"] == "control_ack"
            assert r5["speech_enabled"] is False


# ─── 7. Frame Sequence Handling & Monotonicity ────────────────────────────────

def test_websocket_sequence_monotonicity():
    """Verify stale or duplicate sequence IDs are dropped with STALE_SEQUENCE_DROP."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, enc = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            raw_bytes = enc.tobytes()

            # Send sequence_id = 10
            ws.send_text(json.dumps({"type": "frame", "sequence_id": 10}))
            ws.send_bytes(raw_bytes)
            res1 = json.loads(ws.receive_text())
            assert res1["type"] == "recognition_result"
            assert res1["sequence_id"] == 10

            # Send sequence_id = 5 (stale!)
            ws.send_text(json.dumps({"type": "frame", "sequence_id": 5}))
            ws.send_bytes(raw_bytes)
            res2 = json.loads(ws.receive_text())
            assert res2["type"] == "error"
            assert res2["code"] == "STALE_SEQUENCE_DROP"


# ─── 8. Multi-Session Isolation (Two Simultaneous WebSockets) ─────────────────

def test_two_simultaneous_websocket_sessions():
    """Verify two concurrent WebSocket connections get distinct sessions and isolated state."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws1:
            with client.websocket_connect("/ws/recognition") as ws2:
                # Check active sessions count
                resp_health = client.get("/api/health")
                assert resp_health.json()["active_sessions"] == 2

                # Session 1: SPACE action
                ws1.send_text(json.dumps({"type": "space"}))
                r1 = json.loads(ws1.receive_text())
                assert r1["action"] == "space"

                # Session 2: Clear action
                ws2.send_text(json.dumps({"type": "clear"}))
                r2 = json.loads(ws2.receive_text())
                assert r2["action"] == "clear"


# ─── 9. Smoothing History Isolation ───────────────────────────────────────────

def test_session_smoothing_history_isolation():
    """Verify SessionContexts do not share or leak _prob_history deques."""
    mgr = SessionManager()
    s1 = mgr.create_session("sess_iso_1")
    s2 = mgr.create_session("sess_iso_2")

    try:
        assert s1.prob_history is not s2.prob_history

        # Append to s1
        s1.prob_history.append(np.ones((28,), dtype=np.float32) * 0.1)
        assert len(s1.prob_history) == 1
        assert len(s2.prob_history) == 0  # s2 completely unaffected!

        # Reset s1
        s1.reset_smoothing()
        assert len(s1.prob_history) == 0
        assert len(s2.prob_history) == 0

    finally:
        mgr.close_all()


# ─── 10. Independent FSMs and TextBuffers ─────────────────────────────────────

def test_independent_fsms_and_text_buffers():
    """Verify TextBuffer and GestureStateMachine do not leak across sessions."""
    mgr = SessionManager()
    s1 = mgr.create_session("sess_fsm_1")
    s2 = mgr.create_session("sess_fsm_2")

    try:
        s1.text_buffer.apply("A")
        s1.text_buffer.apply("S")
        s1.text_buffer.apply("L")

        assert s1.text_buffer.get_text() == "ASL"
        assert s2.text_buffer.get_text() == ""

        s2.text_buffer.apply("X")
        assert s1.text_buffer.get_text() == "ASL"
        assert s2.text_buffer.get_text() == "X"

    finally:
        mgr.close_all()


# ─── 11. Disconnect Cleanup ───────────────────────────────────────────────────

def test_websocket_disconnect_cleanup():
    """Verify session is removed and active_sessions decrements upon WebSocket close."""
    with TestClient(app) as client:
        initial_active = client.get("/api/health").json()["active_sessions"]

        with client.websocket_connect("/ws/recognition") as ws:
            during_active = client.get("/api/health").json()["active_sessions"]
            assert during_active == initial_active + 1
            ws.send_text(json.dumps({"type": "init"}))
            ws.receive_text()

        # After context exit, websocket disconnected
        after_active = client.get("/api/health").json()["active_sessions"]
        assert after_active == initial_active


# ─── 12. Real Hand Recognition over WebSocket ─────────────────────────────────

def test_real_hand_websocket_recognition():
    """Verify physical 'L' gesture image produces valid prediction over WebSocket channel."""
    img_l = cv2.imread("data/raw/asl_alphabet_external/L/L0001_test.jpg")
    assert img_l is not None, "Physical test image for L missing"

    _, enc = cv2.imencode(".jpg", img_l, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw_bytes = enc.tobytes()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws:
            ws.send_text(json.dumps({"type": "frame", "sequence_id": 1}))
            ws.send_bytes(raw_bytes)

            resp_data = json.loads(ws.receive_text())
            assert resp_data["type"] == "recognition_result"
            assert resp_data["sequence_id"] == 1
            assert resp_data["hand_detected"] is True
            assert resp_data["landmark_valid"] is True
            assert resp_data["raw_prediction"] == "L"
            assert resp_data["raw_confidence"] > 0.70
            assert resp_data["processing_duration_ms"] > 0.0

            # Real hand contract assertions:
            top_preds = resp_data.get("top_predictions")
            assert isinstance(top_preds, list)
            assert len(top_preds) == 3
            for item in top_preds:
                assert isinstance(item["label"], str)
                assert 0.0 <= item["confidence"] <= 1.0
            assert top_preds[0]["confidence"] >= top_preds[1]["confidence"] >= top_preds[2]["confidence"]
            top_label = resp_data.get("smoothed_prediction") or resp_data.get("raw_prediction")
            assert top_preds[0]["label"] == top_label
            assert isinstance(resp_data["dwell_progress"], (float, int))
            assert 0.0 <= resp_data["dwell_progress"] <= 1.0


def test_websocket_committed_letter_then_no_hand_resets_dwell():
    """Verify holding a sign commits letter, and subsequent blank frame resets dwell_progress to 0.0."""
    img_l = cv2.imread("data/raw/asl_alphabet_external/L/L0001_test.jpg")
    assert img_l is not None, "Physical test image for L missing"
    _, enc_l = cv2.imencode(".jpg", img_l, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw_l_bytes = enc_l.tobytes()

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc_blank = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
    raw_blank_bytes = enc_blank.tobytes()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition") as ws:
            seq = 1
            committed = False
            # Stream physical L frames over >= 0.7s to achieve dwell commitment
            for _ in range(8):
                ws.send_text(json.dumps({"type": "frame", "sequence_id": seq}))
                ws.send_bytes(raw_l_bytes)
                resp = json.loads(ws.receive_text())
                if resp.get("committed_label") == "L":
                    committed = True
                    break
                seq += 1
                time.sleep(0.12)

            assert committed is True, "Expected 'L' to commit after holding steady"

            # Now send no-hand frame
            seq += 1
            ws.send_text(json.dumps({"type": "frame", "sequence_id": seq}))
            ws.send_bytes(raw_blank_bytes)
            no_hand_resp = json.loads(ws.receive_text())

            assert no_hand_resp["hand_detected"] is False
            assert no_hand_resp["dwell_progress"] == 0.0
            assert no_hand_resp.get("top_predictions") is None or no_hand_resp.get("top_predictions") == []


# ─── 13. Resource Cleanup and Application Shutdown ────────────────────────────

def test_application_lifespan_shutdown():
    """Verify lifespan startup loads resources and shutdown closes them cleanly."""
    model_service = SharedModelService()
    model_service.load()
    assert model_service.is_ready() is True

    session_mgr = SessionManager()
    session_mgr.create_session("test_sess")
    assert session_mgr.active_count() == 1

    # Execute shutdown
    session_mgr.close_all()
    assert session_mgr.active_count() == 0

    model_service.close()
    assert model_service.is_ready() is False


# ─── 14. SPA Static File Serving & Client-Side Route Fallback ──────────────────

def test_spa_static_serving_and_fallback(tmp_path: Path):
    """
    Verify SPA static file serving with client-side route fallback:
    - Root / serves index.html.
    - Deep client-side routes (/translator) fall back to index.html.
    - API routes (/api/health) return JSON and are not shadowed.
    - Missing API routes (/api/nonexistent) return 404, not index.html.
    - JavaScript assets return application/javascript MIME type.
    - Missing static assets with extensions (/assets/missing.js) return 404.
    """
    # Setup test static folder structure
    (tmp_path / "index.html").write_text("<!DOCTYPE html><html><body><div id='root'>SPA Root</div></body></html>", encoding="utf-8")
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "index.js").write_text("console.log('asl studio bundle');", encoding="utf-8")

    # Create test app with custom dist path and lifespan bypassed
    spa_app = create_app(frontend_dist=tmp_path, enable_lifespan=False)

    with TestClient(spa_app) as client:
        # A. Root path serves index.html
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "<div id='root'>SPA Root</div>" in resp_root.text
        assert resp_root.headers["content-type"].startswith("text/html")

        # B. Client-side route fallback (/translator) serves index.html
        resp_route = client.get("/translator")
        assert resp_route.status_code == 200
        assert "<div id='root'>SPA Root</div>" in resp_route.text
        assert resp_route.headers["content-type"].startswith("text/html")

        # C. Another client-side route (/diagnostics/telemetry) serves index.html
        resp_deep = client.get("/diagnostics/telemetry")
        assert resp_deep.status_code == 200
        assert "<div id='root'>SPA Root</div>" in resp_deep.text

        # D. API route (/api/health) returns JSON and is NOT shadowed by SPA
        resp_api = client.get("/api/health")
        assert resp_api.status_code == 200
        assert "application/json" in resp_api.headers["content-type"]
        assert "status" in resp_api.json()

        # E. Missing API route returns 404, NEVER index.html
        resp_missing_api = client.get("/api/nonexistent")
        assert resp_missing_api.status_code == 404
        assert "SPA Root" not in resp_missing_api.text

        # F. JS asset returns valid javascript MIME type (application/javascript or text/javascript)
        resp_js = client.get("/assets/index.js")
        assert resp_js.status_code == 200
        js_media_type = resp_js.headers["content-type"].split(";")[0].strip()
        assert js_media_type in ("application/javascript", "text/javascript")
        assert "asl studio bundle" in resp_js.text

        # G. Missing asset with file extension returns 404, not index.html
        resp_missing_asset = client.get("/assets/missing_bundle.js")
        assert resp_missing_asset.status_code == 404


def test_spa_real_dist_smoke_test():
    """
    Smoke test: Serves the real frontend/dist production bundle
    and verifies that root / returns HTTP 200 with the React root element.
    """
    real_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if not real_dist.exists() or not (real_dist / "index.html").exists():
        pytest.skip("frontend/dist build directory does not exist")

    real_app = create_app(frontend_dist=real_dist, enable_lifespan=False)

    with TestClient(real_app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert '<div id="root">' in resp.text or '<div id="root"' in resp.text
        assert resp.headers["content-type"].startswith("text/html")

        # Verify client-side route fallback on real build
        resp_trans = client.get("/translator")
        assert resp_trans.status_code == 200
        assert '<div id="root">' in resp_trans.text or '<div id="root"' in resp_trans.text

