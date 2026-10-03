import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
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
# 2. LOAD MODEL
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

for index, name in enumerate(class_names):

    print(
        f"{index}: {name}"
    )


# ==================================================
# 4. COLLECT IMAGE PATHS
# ==================================================

image_paths = []
labels = []


for class_index, class_name in enumerate(
    class_names
):

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    for filename in sorted(
        os.listdir(class_dir)
    ):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):

            image_paths.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )

            labels.append(
                class_index
            )


# ==================================================
# 5. RECREATE EXACT STRATIFIED SPLIT
# ==================================================

from sklearn.model_selection import train_test_split


train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)


val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )
)


# ==================================================
# 6. IMAGE LOADING
# ==================================================

def load_image(
    path,
    label
):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ==================================================
# 7. CREATE TEST DATASET
# ==================================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(
    BATCH_SIZE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ==================================================
# 8. PREDICTIONS
# ==================================================

true_labels = []
predicted_labels = []


for images, labels_batch in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels_batch.numpy()
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
# 9. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")

print(cm)


# ==================================================
# 10. CLASSIFICATION REPORT
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