from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Prepare the data in JSON format for TensorFlow Serving
    json_data = {
        "instances": img_batch.tolist()
    }

    # Send the POST request to TensorFlow Serving
    response = requests.post(endpoint, json=json_data)

    # Print the entire response for debugging
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)

    try:
        # Check if 'predictions' key is in the response JSON
        response_json = response.json()
        if "predictions" in response_json:
            prediction = np.array(response_json["predictions"][0])
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction)

            return {
                "class": predicted_class,
                "confidence": confidence
            }
        else:
            return {"error": "'predictions' key not found in response"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
