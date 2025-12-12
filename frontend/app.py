import streamlit as st
import requests
from PIL import Image
import io
import time

# --- Configuration ---
# CRITICAL: This MUST point to your structured FastAPI endpoint
API_URL = "http://127.0.0.1:8000/api/v1/predict/" 

# Constants for visual feedback (must match API labels: Normal, Benign, Malignant)
COLOR_MAP = {"Normal": "green", "Benign": "orange", "Malignant": "red"}
LABEL_DISPLAY_MAP = {"Normal": "✅ NORMAL", "Benign": "🟡 BENIGN", "Malignant": "❌ MALIGNANT"}

# --- Project Setup for Launch Instructions (For Managers/Reviewers) ---
LAUNCH_INSTRUCTIONS = """
**1. Launch Backend (API Server):**
```bash
uvicorn api.__init__:create_app --factory --reload --port 8000