"""
WebSocket Endpoint for Real-Time ASL Gesture Recognition.
Handles low-latency streaming of binary JPEG frames and control events.
"""

import time
import json
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.schemas import (
    ClientFrameMetadata,
    ClientControlMessage,
    RecognitionResponse,
    ControlAckResponse,
    ErrorResponse
)
from src.api.recognition_service import RecognitionService
from src.api.session_manager import UserSessionContext, SessionManager
from src.api.model_service import SharedModelService


router = APIRouter()


@router.websocket("/ws/recognition")
async def websocket_recognition(websocket: WebSocket):
    """
    Bi-directional streaming recognition endpoint.
    Maintains 100% session isolation, backpressure management, and non-blocking worker execution.
    """
    await websocket.accept()

    app_state = websocket.app.state
    model_service: SharedModelService = app_state.model_service
    session_manager: SessionManager = app_state.session_manager

    # 1. Establish session context for this WebSocket connection
    session: UserSessionContext = session_manager.create_session()
    print(f"[WebSocket] Client connected. Assigned session_id='{session.session_id}'. Active: {session_manager.active_count()}")

    current_meta: Optional[ClientFrameMetadata] = None

    try:
        while True:
            # Receive next message from client (can be JSON text or binary bytes)
            raw_msg = await websocket.receive()
            server_rx_ms = int(time.time() * 1000)

            if "text" in raw_msg:
                text_payload = raw_msg["text"]
                try:
                    data = json.loads(text_payload)
                except Exception:
                    await websocket.send_text(
                        ErrorResponse(code="INVALID_JSON", message="Failed to parse JSON text payload").model_dump_json()
                    )
                    continue

                msg_type = data.get("type", "")

                # Case A: Frame metadata preceding a binary frame
                if msg_type == "frame":
                    try:
                        current_meta = ClientFrameMetadata.model_validate(data)
                    except Exception as e:
                        await websocket.send_text(
                            ErrorResponse(code="INVALID_METADATA", message=f"Invalid frame metadata: {e}").model_dump_json()
                        )
                        current_meta = None
                    continue

                # Case B: Action / Control commands
                try:
                    ctrl = ClientControlMessage.model_validate(data)
                except Exception as e:
                    await websocket.send_text(
                        ErrorResponse(code="INVALID_CONTROL", message=f"Invalid control message: {e}").model_dump_json()
                    )
                    continue

                # Execute action on session-isolated context
                if ctrl.type == "init":
                    if ctrl.speech_enabled is not None:
                        session.speech_enabled = ctrl.speech_enabled
                    ack = ControlAckResponse(
                        action="init",
                        status="ok",
                        text_buffer=session.text_buffer.get_text(),
                        speech_enabled=session.speech_enabled
                    )
                    await websocket.send_text(ack.model_dump_json())

                elif ctrl.type == "space":
                    words = session.text_buffer.get_text().strip().split()
                    last_word = words[-1] if words else ""
                    session.text_buffer.apply("SPACE")
                    ack = ControlAckResponse(
                        action="space",
                        status="ok",
                        text_buffer=session.text_buffer.get_text(),
                        speech_token=last_word if session.speech_enabled and last_word else None
                    )
                    await websocket.send_text(ack.model_dump_json())

                elif ctrl.type == "delete":
                    session.text_buffer.apply("DELETE")
                    ack = ControlAckResponse(
                        action="delete",
                        status="ok",
                        text_buffer=session.text_buffer.get_text()
                    )
                    await websocket.send_text(ack.model_dump_json())

                elif ctrl.type == "clear":
                    session.reset_state()
                    ack = ControlAckResponse(
                        action="clear",
                        status="ok",
                        text_buffer=""
                    )
                    await websocket.send_text(ack.model_dump_json())

                elif ctrl.type == "speech_pref":
                    if ctrl.speech_enabled is not None:
                        session.speech_enabled = ctrl.speech_enabled
                    ack = ControlAckResponse(
                        action="speech_pref",
                        status="ok",
                        text_buffer=session.text_buffer.get_text(),
                        speech_enabled=session.speech_enabled
                    )
                    await websocket.send_text(ack.model_dump_json())

                elif ctrl.type == "shutdown":
                    break

            elif "bytes" in raw_msg:
                # Binary JPEG payload received
                raw_bytes = raw_msg["bytes"]

                seq_id = current_meta.sequence_id if current_meta else (session.last_sequence_id + 1)
                current_meta = None  # Consume metadata association

                # Backpressure & Monotonic Sequence Check
                if session.is_processing:
                    # Drop incoming frame to prevent queue buildup and latency accumulation
                    await websocket.send_text(
                        ErrorResponse(
                            sequence_id=seq_id,
                            code="BACKPRESSURE_DROP",
                            message="Previous frame is still processing. Dropped to maintain real-time responsiveness."
                        ).model_dump_json()
                    )
                    continue

                if seq_id <= session.last_sequence_id and session.last_sequence_id >= 0:
                    await websocket.send_text(
                        ErrorResponse(
                            sequence_id=seq_id,
                            code="STALE_SEQUENCE_DROP",
                            message=f"Sequence ID {seq_id} is stale (last processed: {session.last_sequence_id})."
                        ).model_dump_json()
                    )
                    continue

                session.is_processing = True
                session.last_sequence_id = seq_id

                try:
                    # Offload CPU-bound perception & inference to worker thread
                    # Keeps the main asyncio event loop completely free and responsive!
                    response: RecognitionResponse = await asyncio.to_thread(
                        RecognitionService.process_frame,
                        raw_bytes,
                        session,
                        model_service,
                        seq_id,
                        server_rx_ms,
                        True  # mirror_feed
                    )
                    await websocket.send_text(response.model_dump_json())

                except ValueError as val_err:
                    await websocket.send_text(
                        ErrorResponse(
                            sequence_id=seq_id,
                            code="VALIDATION_ERROR",
                            message=str(val_err)
                        ).model_dump_json()
                    )
                except Exception as exc:
                    await websocket.send_text(
                        ErrorResponse(
                            sequence_id=seq_id,
                            code="INTERNAL_PROCESSING_ERROR",
                            message=f"Recognition failed: {str(exc)}"
                        ).model_dump_json()
                    )
                finally:
                    session.is_processing = False

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected cleanly (session_id='{session.session_id}').")
    except Exception as e:
        print(f"[WebSocket] Exception in session '{session.session_id}': {e}")
    finally:
        session_manager.remove_session(session.session_id)
        print(f"[WebSocket] Session '{session.session_id}' cleaned up. Remaining active: {session_manager.active_count()}")

