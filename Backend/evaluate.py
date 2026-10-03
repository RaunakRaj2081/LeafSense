import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 123

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "PLANT_DISEASES",
    "PlantVillage"
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
# 2. LOAD TRAINED MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==================================================
# 3. LOAD CLASS NAMES
# ==================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


print("Class mapping:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ==================================================
# 4. LOAD DATASET
# ==================================================

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    seed=SEED,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)


# ==================================================
# 5. RECREATE SAME TEST SPLIT
# ==================================================

dataset_size = len(dataset)

train_size = int(
    0.8 * dataset_size
)

val_size = int(
    0.1 * dataset_size
)

train_ds = dataset.take(
    train_size
)

remaining_ds = dataset.skip(
    train_size
)

val_ds = remaining_ds.take(
    val_size
)

test_ds = remaining_ds.skip(
    val_size
)


# ==================================================
# 6. GET TRUE LABELS AND PREDICTIONS
# ==================================================

true_labels = []
predicted_labels = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)


# ==================================================
# 7. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")

print(cm)


# ==================================================
# 8. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)