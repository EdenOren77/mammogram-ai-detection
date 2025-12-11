from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import torch
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import BreastCancerModel
import numpy as np
import cv2

MODEL_PATH="saved_models/resnet50_mammogram.pth"
LABELS_MAP={0:"Normal",1:"Benign",2:"Malignant"}

#server
app=FastAPI(titel="Breast Cancer Detection API")
model=None
device=torch.device("cpu")

@app.on_event("startup")
def startup_event():
    global model
    try:
        model=BreastCancerModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        #predict mode
        model.eval()
    except Exception as e:
        print(f"FATAL ERROR: Could not load model. {e}")
        raise e

#Decodes the image bytes and prepares them for the ResNet model
def preprocess_image(image_bytes):
    nparr=np.frombuffer(image_bytes,np.uint8)
    img=cv2.imdecode(nparr,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Is this a valid file?")
    
    img=cv2.resize(img,(224,224))
    
    # Transform to Tensor:
    # 1. unsqueeze(0).unsqueeze(0): Adds Batch and Channel dimensions -> [1, 1, 224, 224]
    # 2. Divide by 255.0: Normalizes pixel values from range [0, 255] to [0, 1]
    img_tensor=torch.tensor(img,dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
    return img_tensor

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    if model is None:
        raise HTTPException(status_code=500,detail="Model not loaded yet")
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400,detail="File must be an image.")
    try:
        contents=await file.read()
        input_tensor=preprocess_image(contents).to(device)
        with torch.no_grad(): # Disable gradient calculation for speed
            outputs=model(input_tensor)
            probabilities=F.softmax(outputs,dim=1)
            prediction_idx=torch.argmax(probabilities,dim=1).item()
        return {
            "filename": file.filename,
            "prediction": LABELS_MAP[prediction_idx],
            "confidence": f"{probabilities[0][prediction_idx].item() * 100:.2f}%",
            "probabilities": {
                "normal": float(probabilities[0][0]),
                "benign": float(probabilities[0][1]),
                "malignant": float(probabilities[0][2])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)