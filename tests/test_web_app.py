"""
Unit and Integration Tests for Browser-Based ASL Recognition Web Application.

Verifies:
1. WebSessionState initialization and session independence.
2. Handling of null or empty webcam frames gracefully.
3. End-to-end processing of synthetic webcam RGB frames through process_webcam_frame.
4. Toolbar controls: SPACE completes word and emits speech token.
5. Toolbar controls: DELETE modifies text buffer without speech.
6. Toolbar controls: CLEAR completely flushes session state and text buffer.
7. Toolbar controls: Speech toggle switches speech state.
8. build_app constructs a valid, renderable Gradio Blocks application.
"""

import time
import numpy as np
import pytest
import gradio as gr

from src.web.app import (
    WebSessionState,
    create_session,
    resolve_session,
    process_webcam_frame,
    on_space_clicked,
    on_delete_clicked,
    on_clear_clicked,
    on_toggle_speech,
    build_app,
)


def test_web_session_initialization():
    """Verify that a session creates an isolated pipeline without SAPI5 speech."""
    session = create_session()
    assert isinstance(session, WebSessionState)
    assert session.pipeline is not None
    assert session.pipeline.speech_engine is None  # Server SAPI5 disabled
    assert session.speech_enabled is True
    assert session.pipeline.text_buffer.get_text() == ""


def test_session_state_isolation():
    """Verify that two browser sessions do not leak text buffer or state machine."""
    s1 = create_session()
    s2 = create_session()

    s1.pipeline.text_buffer.apply("A")
    s1.pipeline.text_buffer.apply("B")

    assert s1.pipeline.text_buffer.get_text() == "AB"
    assert s2.pipeline.text_buffer.get_text() == ""

    s2.pipeline.text_buffer.apply("C")
    assert s1.pipeline.text_buffer.get_text() == "AB"
    assert s2.pipeline.text_buffer.get_text() == "C"


def test_process_webcam_frame_none_input():
    """Verify that None or empty frame returns waiting status without crashing."""
    session = create_session()
    rgb_out, label, conf, status, text, speech, updated_session = process_webcam_frame(None, session)

    assert rgb_out is None
    assert label == "-"
    assert conf == "0.0%"
    assert status == "WAITING FOR WEBCAM"
    assert speech == ""


def test_process_webcam_frame_valid_dimensions():
    """Verify that a valid uint8 RGB frame produces an annotated RGB output."""
    session = create_session()
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    rgb_out, label, conf, status, text, speech, updated_session = process_webcam_frame(dummy_rgb, session)

    assert isinstance(rgb_out, np.ndarray)
    assert rgb_out.shape == (480, 640, 3)
    assert status == "NO_HAND"
    assert label == "-"


def test_toolbar_controls_space_word_speech():
    """Verify that SPACE completes the current word and emits speech token."""
    session = create_session()
    session.pipeline.text_buffer.apply("C")
    session.pipeline.text_buffer.apply("A")
    session.pipeline.text_buffer.apply("T")

    text, speech, session = on_space_clicked(session)
    assert text == "CAT "
    assert speech == "CAT"  # Emitted for browser speech synthesis


def test_toolbar_controls_delete():
    """Verify that DELETE removes the last character without emitting speech."""
    session = create_session()
    session.pipeline.text_buffer.apply("A")
    session.pipeline.text_buffer.apply("B")

    text, session = on_delete_clicked(session)
    assert text == "A"


def test_toolbar_controls_clear():
    """Verify that CLEAR completely wipes the text buffer."""
    session = create_session()
    session.pipeline.text_buffer.apply("H")
    session.pipeline.text_buffer.apply("I")

    text, session = on_clear_clicked(session)
    assert text == ""
    assert session.pipeline.text_buffer.get_text() == ""


def test_toolbar_speech_toggle():
    """Verify toggling speech ON and OFF."""
    session = create_session()
    assert session.speech_enabled is True

    btn_text, session = on_toggle_speech(session)
    assert session.speech_enabled is False
    assert "MUTED" in btn_text

    btn_text, session = on_toggle_speech(session)
    assert session.speech_enabled is True
    assert "ON" in btn_text


def test_build_app_structure():
    """Verify that build_app returns a valid Gradio Blocks instance."""
    app = build_app()
    assert isinstance(app, gr.Blocks)


def test_resolve_session_with_callable_and_none():
    """Verify resolve_session correctly executes callables and instantiates defaults."""
    # 1. Callable input (Gradio 6.x gr.State behavior)
    session_from_fn = resolve_session(create_session)
    assert isinstance(session_from_fn, WebSessionState)

    # 2. None input
    session_from_none = resolve_session(None)
    assert isinstance(session_from_none, WebSessionState)

    # 3. Already instantiated
    existing = create_session()
    resolved = resolve_session(existing)
    assert resolved is existing


def test_process_webcam_frame_with_callable_session():
    """Verify process_webcam_frame does not crash with AttributeError when session is a callable."""
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # Pass create_session function directly, simulating Gradio 6.x state injection
    rgb_out, label, conf, status, text, speech, updated_session = process_webcam_frame(
        dummy_rgb,
        create_session
    )
    assert isinstance(updated_session, WebSessionState)
    assert isinstance(rgb_out, np.ndarray)
    assert rgb_out.shape == (480, 640, 3)
    assert status == "NO_HAND"


def test_toolbar_callbacks_with_callable_session():
    """Verify toolbar callbacks handle callable session gracefully."""
    text, speech, s1 = on_space_clicked(create_session)
    assert isinstance(s1, WebSessionState)

    text, s2 = on_delete_clicked(create_session)
    assert isinstance(s2, WebSessionState)

    text, s3 = on_clear_clicked(create_session)
    assert isinstance(s3, WebSessionState)

    state_str, s4 = on_toggle_speech(create_session)
    assert isinstance(s4, WebSessionState)


def test_process_webcam_frame_diagnostic_logging(capsys):
    """Verify diagnostic logging prints frame information and pipeline execution."""
    session = create_session()
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    process_webcam_frame(dummy_rgb, session)
    captured = capsys.readouterr()

    assert "[WebCam #" in captured.out
    assert "Input frame: shape=(480, 640, 3)" in captured.out
    assert "Pipeline (" in captured.out
    assert "Hand=False" in captured.out


def test_api_process_frame_endpoint():
    """Verify /api/process_frame endpoint decodes JPEG and returns annotated frame + telemetry."""
    import cv2
    from starlette.testclient import TestClient

    app = build_app()
    client = TestClient(app.app)

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    raw_bytes = enc.tobytes()

    response = client.post(
        "/api/process_frame?session_id=test_session_1",
        content=raw_bytes,
        headers={"Content-Type": "application/octet-stream"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "annotated_frame" in data
    assert data["annotated_frame"].startswith("data:image/jpeg;base64,")
    assert "pred_label" in data
    assert "confidence" in data
    assert "status" in data
    assert "fps" in data


def test_api_action_endpoints():
    """Verify /api/action handles space, delete, clear, and toggle_speech."""
    from starlette.testclient import TestClient

    app = build_app()
    client = TestClient(app.app)

    # 1. Action: space
    resp_space = client.post("/api/action?action=space&session_id=test_act_1")
    assert resp_space.status_code == 200
    assert "text" in resp_space.json()

    # 2. Action: delete
    resp_del = client.post("/api/action?action=delete&session_id=test_act_1")
    assert resp_del.status_code == 200
    assert "text" in resp_del.json()

    # 3. Action: clear
    resp_clear = client.post("/api/action?action=clear&session_id=test_act_1")
    assert resp_clear.status_code == 200
    assert resp_clear.json()["text"] == ""

    # 4. Action: toggle_speech
    resp_toggle = client.post("/api/action?action=toggle_speech&session_id=test_act_1")
    assert resp_toggle.status_code == 200
    assert "speech_toggle_text" in resp_toggle.json()



