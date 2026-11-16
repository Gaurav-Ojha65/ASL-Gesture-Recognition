"""
Lifespan Management for FastAPI Backend.
Handles application startup and graceful shutdown:
- Loads shared model weights and MediaPipe detector once during startup.
- Initializes bounded SessionManager.
- Safely releases GPU/CPU memory and TFLite delegates on shutdown.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI

from src.api.model_service import SharedModelService
from src.api.session_manager import SessionManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manages the lifecycle of shared backend resources.
    Guarantees models load before requests are accepted and close on shutdown.
    """
    print("[FastAPI Lifespan] Starting ASL Gesture Studio Backend...")

    # 1. Initialize and load shared model service
    model_service = SharedModelService(
        checkpoint_path="models/asl_landmark_mlp_best.pt",
        task_model_path="hand_landmarker.task"
    )
    model_service.load()
    print("[FastAPI Lifespan] SharedModelService initialized and ready.")

    # 2. Initialize bounded session manager
    session_manager = SessionManager()
    print("[FastAPI Lifespan] SessionManager initialized.")

    # Attach to application state for dependency injection across route handlers
    app.state.model_service = model_service
    app.state.session_manager = session_manager

    yield

    # 3. Graceful shutdown cleanup
    print("[FastAPI Lifespan] Shutting down ASL Gesture Studio Backend...")
    session_manager.close_all()
    model_service.close()
    print("[FastAPI Lifespan] Resources cleanly released.")

