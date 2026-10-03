import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "PLANT_DISEASES",
    "PlantVillage"
)


# Load model
model = tf.keras.models.load_model(
    MODEL_PATH
)


# Load class names
with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


# Select one image from each class
test_images = []

for class_name in class_names:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ])

    image_path = os.path.join(
        class_dir,
        files[0]
    )

    test_images.append(
        (class_name, image_path)
    )


# Predict
for actual_class, image_path in test_images:

    image = Image.open(
        image_path
    ).convert("RGB")

    image_array = np.array(
        image
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_batch,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    print("\n------------------------------")

    print(
        "Actual:",
        actual_class
    )

    print(
        "Predicted:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print(
        "Image:",
        image_path
    )

    print(
        "All probabilities:"
    )

    for i, probability in enumerate(
        predictions
    ):

        print(
            f"  {class_names[i]}: "
            f"{probability * 100:.2f}%"
        )