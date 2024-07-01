from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import base64
import numpy as np
from ultralytics import YOLO
from make_cubeSIde import prediction_function

app = FastAPI()

# Load your YOLOv8 model
model = YOLO('best.pt')

class ImageRequest(BaseModel):
    image: str

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        detections = prediction_function(model,base64_image=request.image)
        
        return {"success": True, "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
