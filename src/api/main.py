"""
FastAPI Main Application Entry Point for ASL Gesture Studio.
Defines REST routes, WebSocket streaming, and application lifecycle.
Includes SPA static file serving with Windows Registry MIME type fixes.
"""

import sys
import time
import os
import mimetypes
import logging
from pathlib import Path
from typing import Optional

# Fix Windows Registry MIME type bugs where .js and .mjs may be registered as text/plain
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

import torch
import cv2
import mediapipe as mp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.lifespan import lifespan
from src.api.schemas import HealthResponse, SystemInfoResponse
from src.api.websocket import router as ws_router
from src.landmarks.labels import SEMANTIC_CLASSES

logger = logging.getLogger("uvicorn")


class SPAStaticFiles(StaticFiles):
    """
    Custom StaticFiles handler that serves index.html for client-side SPA routes
    while preserving real 404s for missing assets with file extensions,
    and never shadowing /api or /ws endpoints (handles cross-platform backslashes).
    """
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            norm_path = path.replace("\\", "/").lstrip("/")
            last = norm_path.rsplit("/", 1)[-1]
            if exc.status_code == 404 and "." not in last and not (norm_path in ("api", "ws") or norm_path.startswith(("api/", "ws/"))):
                return await super().get_response("index.html", scope)
            raise


def create_app(frontend_dist: Optional[Path] = None, enable_lifespan: bool = True) -> FastAPI:
    """
    Application factory for ASL Gesture Studio.
    Registers /api and /ws routes FIRST before mounting SPA static serving.
    """
    application = FastAPI(
        title="ASL Gesture Studio API",
        description="High-performance asynchronous backend for real-time American Sign Language recognition.",
        version="1.0.0",
        lifespan=lifespan if enable_lifespan else None
    )

    # Enable CORS for React + Vite development and cross-origin clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register WebSocket and REST routes FIRST to prevent route shadowing
    application.include_router(ws_router)

    @application.get("/api/health", response_model=HealthResponse)
    async def get_health(request: Request):
        """
        Liveness and readiness probe for containerized deployments and health monitoring.
        """
        model_service = getattr(request.app.state, "model_service", None)
        session_manager = getattr(request.app.state, "session_manager", None)

        is_model_ready = bool(model_service and model_service.is_ready())
        active_count = session_manager.active_count() if session_manager else 0

        return HealthResponse(
            status="healthy" if is_model_ready else "degraded",
            timestamp_utc=time.time(),
            model_loaded=is_model_ready,
            detector_loaded=is_model_ready,
            active_sessions=active_count,
        )

    @application.get("/api/system/info", response_model=SystemInfoResponse)
    async def get_system_info(request: Request):
        """
        Returns hardware, model metadata, class vocabulary, and perception configurations.
        """
        model_service = getattr(request.app.state, "model_service", None)
        device_str = str(model_service.device) if model_service else ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_rel = model_service.checkpoint_path if model_service else "models/asl_landmark_mlp_best.pt"
        param_count = getattr(model_service.engine, "param_count", 18524) if (model_service and model_service.engine) else 18524

        return SystemInfoResponse(
            app_name="ASL Gesture Studio API",
            version="1.0.0",
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device=device_str,
            model_checkpoint=checkpoint_rel,
            model_param_count=param_count,
            mediapipe_version=getattr(mp, "__version__", "1.0.1"),
            opencv_version=cv2.__version__,
            input_dim=63,
            num_classes=len(SEMANTIC_CLASSES),
            classes=SEMANTIC_CLASSES,
            min_confidence=0.70,
            dwell_duration=0.60,
            release_duration=0.25,
        )

    # Resolve default frontend dist path: C:\project\Computer_Vision\frontend\dist
    default_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    dist_path = frontend_dist if frontend_dist is not None else default_dist

    if dist_path.exists() and dist_path.is_dir():
        application.mount("/", SPAStaticFiles(directory=str(dist_path), html=True), name="frontend")
    else:
        logger.warning(
            f"[FastAPI SPA] Frontend dist directory not found at '{dist_path.resolve()}'. "
            "Static SPA serving is disabled. Build frontend using 'npm run build' in frontend/ directory."
        )

    return application


# Top-level application instance for uvicorn runtime
app = create_app()


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Convenience launcher function."""
    import uvicorn
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
