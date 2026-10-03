from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import json


app = FastAPI()


# ==================================================
# CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://10.14.21.188:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run backend/model.py first to train and save the model."
    )

MODEL = tf.keras.models.load_model(
    MODEL_PATH
)


# ==================================================
# LOAD CLASS NAMES
# ==================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}\n"
        "Run backend/model.py first."
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    CLASS_NAMES = json.load(file)


print("Model loaded successfully.")
print("Class names:", CLASS_NAMES)


# ==================================================
# PING ENDPOINT
# ==================================================

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# ==================================================
# IMAGE READING FUNCTION
# ==================================================

def read_file_as_image(data) -> np.ndarray:

    image = Image.open(
        BytesIO(data)
    ).convert("RGB")

    return np.array(image)


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_data = await file.read()

    # Convert image into NumPy array
    image = read_file_as_image(
        image_data
    )

    # Add batch dimension
    img_batch = np.expand_dims(
        image,
        axis=0
    )

    # Run CNN inference
    predictions = MODEL.predict(
        img_batch,
        verbose=0
    )

    # Find class with highest probability
    predicted_index = int(
        np.argmax(
            predictions[0]
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Get confidence
    confidence = float(
        np.max(
            predictions[0]
        )
    )


    # ==================================================
    # DISPLAY NAME MAPPING
    # ==================================================

    display_names = {
        "Potato___Early_blight": "Early Blight",
        "Potato___Late_blight": "Late Blight",
        "Potato___healthy": "Healthy"
    }

    predicted_class_display = display_names.get(
        predicted_class,
        predicted_class
    )


    # ==================================================
    # RETURN JSON RESPONSE
    # ==================================================

    return {
        "class": predicted_class_display,
        "confidence": confidence
    }


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )