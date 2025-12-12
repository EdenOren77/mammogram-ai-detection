# api/core.py

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from fastapi import HTTPException
from .consts import MODEL_INPUT_SIZE

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Converts raw image bytes into a preprocessed PyTorch tensor ready for the model."""
    
    # 1. Decode image bytes using OpenCV
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or decoding failed.")

    # 2. Resize and normalize
    image = cv2.resize(image, MODEL_INPUT_SIZE)
    
    # 3. Convert to Tensor (dtype=float32), normalize by 255.0, add channel (0) and batch (0) dimensions
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    
    return image_tensor

def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """Executes the forward pass on the model and computes probabilities."""
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get final probabilities
        probabilities = F.softmax(outputs, dim=1)
        
    return probabilities