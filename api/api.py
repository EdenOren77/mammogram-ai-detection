# api/api.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from .models.prediction import PredictionResponse 
from .core import preprocess_image, run_inference
from .consts import LABELS_MAP

# 1. Initialize the Router
api_router = APIRouter()

@api_router.post(
    "/predict/",
    response_model=PredictionResponse, # <-- Ensures output quality and documentation
    summary="Predict Mammogram Classification"
)
async def predict(request: Request, file: UploadFile = File(...)):
    """Handles file upload and returns prediction results using the AI model."""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Get the pre-loaded model from the application state (set in __init__.py)
        model = request.app.state.model 

        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Run inference using the core function
        probabilities = run_inference(model, image_tensor) 

        # Post-processing and formatting
        predicted_index = probabilities.argmax(dim=1).item()
        predicted_label = LABELS_MAP[predicted_index]
        
        # Format probabilities dictionary using StrEnum keys
        probs_dict = {
            label.value: probabilities[0, index].item()
            for index, label in LABELS_MAP.items()
        }
        
        return {
            "prediction": predicted_label,
            "probabilities": probs_dict,
            "model_used": model.__class__.__name__ 
        }

    except Exception as e:
        # Catch any inference or preprocessing errors
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")