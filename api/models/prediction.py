# api/models/prediction.py

from pydantic import BaseModel, Field
from typing import Dict
from ..consts import PredictionLabel

class PredictionResponse(BaseModel):
    """Defines the structure for the API's successful prediction output."""
    
    # Ensures the prediction field is one of the valid enum values
    prediction: PredictionLabel = Field(..., description="The model's final classification label.")
    
    # Probabilities map label strings (from the Enum) to float values
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each class.")
    
    # Information about the model used
    model_used: str = Field("ResNet-50", description="Name of the model architecture used for inference.")

# __init__.py is needed in the models/ folder to make it a Python package
# Create an empty file: api/models/__init__.py