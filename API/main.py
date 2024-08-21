from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Annotated

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am done"}

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return np.array(image)

@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
    data = await file.read()
    image = read_file_as_image(data)
    # Log the shape of the image for debugging
    print(f"Image shape: {image.shape}")
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_shape": image.shape.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
