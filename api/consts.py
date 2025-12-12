# api/consts.py

from enum import StrEnum

# 1. Prediction Labels (The StrEnum)
class PredictionLabel(StrEnum):
    """Fixed set of possible output classifications."""
    NORMAL = "Normal"
    BENIGN = "Benign"
    MALIGNANT = "Malignant"

# 2. Label Index Mapping (Used in Core/API)
# Maps the model's output index (0, 1, 2) to the StrEnum value
LABELS_MAP = {
    0: PredictionLabel.NORMAL,
    1: PredictionLabel.BENIGN,
    2: PredictionLabel.MALIGNANT,
}

# 3. Application and Model Configuration
APP_TITLE = "Mammogram AI Prediction API (Full Stack)"
APP_DESCRIPTION = "High-performance FastAPI serving the ResNet-50 AI model."

# Model Configuration
MODEL_PATH = "saved_models/mammogram_cnn.pth"
MODEL_INPUT_SIZE = (224, 224)