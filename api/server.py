from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import sys
import os
import shutil  #Used to save the temporary image file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import predictor

app=FastAPI(title="Breast Cancer Detection API")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and return the diagnosis.
    Args:
        image (UploadFile): The image file sent by the client.
    Returns:
        JSON object containing the label, confidence score, and probability breakdown.
    """
    temp_filename = f"temp_{image.filename}"
    
    try:
        #the predictor expects a file path, we write the upload stream to disk
        with open(temp_filename,"wb") as buffer:
            shutil.copyfileobj(image.file,buffer)

        #Run Inference
        #Pass the file path to the predictor to get the model's output
        result=predictor.predict(temp_filename)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if result:
            return result
        else:
            raise HTTPException(status_code=500, detail="Prediction failed inside the model")

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    print("Server starting on http://127.0.0.1:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)