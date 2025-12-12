# api/__init__.py

import os
import sys
import torch
from fastapi import FastAPI
from .consts import MODEL_PATH, APP_TITLE, APP_DESCRIPTION, PredictionLabel
from .api import api_router

# --- We assume the partner is using this class name in src/model.py ---
try:
    # Adding parent directory to path to find src/model.py
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.model import SimpleCNN # We stick to SimpleCNN for now
    MODEL_CLASS = SimpleCNN
except ImportError as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to import model class from src.model. {e}")


def create_app() -> FastAPI:
    """Application Factory: Creates and configures the FastAPI application."""
    
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        # Add the custom exception handler here if needed (e.g., for ModelLoadError)
    )
    
    # 1. Initialize and Load the AI Model
    try:
        model = MODEL_CLASS() 
        # The number of output classes must match the LABELS_MAP size
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        app.state.model = model  # Attach model to app state for access in api.py
        print(f"Model {MODEL_CLASS.__name__} loaded successfully from {MODEL_PATH}")
    except Exception as e:
        # CRITICAL FAILURE: If the model cannot load, the service cannot start.
        print(f"CRITICAL ERROR: Failed to load model from {MODEL_PATH}. {e}")
        # Raising the error will prevent Uvicorn from starting the service.
        raise RuntimeError(f"Model initialization failed: {e}")

    # 2. Register Routers (API Endpoints)
    # Prefixing the API routes with /api/v1 is a common best practice
    app.include_router(api_router, prefix="/api/v1")
    
    return app