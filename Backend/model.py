import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers


# ==================================================
# 1. CONFIGURATION
# ==================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
SEED = 123

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==================================================
# 2. PATH CONFIGURATION
# ==================================================

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
# 3. CLASS CONFIGURATION
# ==================================================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

num_classes = len(class_names)


# ==================================================
# 4. CHECK DATASET
# ==================================================

if not os.path.isdir(DATASET_DIR):

    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )


# ==================================================
# 5. COLLECT IMAGE PATHS AND LABELS
# ==================================================

image_paths = []
labels = []


for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.isdir(class_dir):

        raise FileNotFoundError(
            f"Class directory not found: {class_dir}"
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


print("Total images:", len(image_paths))

print("\nOriginal class distribution:")

for index, name in enumerate(class_names):

    print(
        f"{name}: {labels.count(index)}"
    )


# ==================================================
# 6. STRATIFIED TRAIN / TEMP SPLIT
#    80% TRAIN
#    20% TEMP
# ==================================================

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)


# ==================================================
# 7. STRATIFIED VALIDATION / TEST SPLIT
#    10% VALIDATION
#    10% TEST
# ==================================================

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
# 8. PRINT SPLIT DISTRIBUTION
# ==================================================

def print_distribution(
    name,
    labels_list
):

    print(f"\n{name} distribution:")

    for index, class_name in enumerate(class_names):

        print(
            f"{class_name}: "
            f"{labels_list.count(index)}"
        )


print("\nSplit sizes:")

print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

print_distribution(
    "Training",
    train_labels
)

print_distribution(
    "Validation",
    val_labels
)

print_distribution(
    "Testing",
    test_labels
)


# ==================================================
# 9. CREATE TF.DATA DATASETS
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


def create_dataset(
    paths,
    labels,
    shuffle=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    shuffle=True
)

val_ds = create_dataset(
    val_paths,
    val_labels
)

test_ds = create_dataset(
    test_paths,
    test_labels
)


# ==================================================
# 10. DATA AUGMENTATION
# ==================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal_and_vertical"
    ),

    layers.RandomRotation(
        0.2
    )

])


# ==================================================
# 11. OPTIMIZE DATA PIPELINE
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.map(
    lambda images, labels: (
        data_augmentation(
            images,
            training=True
        ),
        labels
    ),
    num_parallel_calls=AUTOTUNE
)


train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)


# ==================================================
# 12. BUILD CNN
# ==================================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    layers.Resizing(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    layers.Rescaling(
        1.0 / 255
    ),


    # ---------- Convolution Block 1 ----------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),


    # ---------- Convolution Block 2 ----------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),


    # ---------- Convolution Block 3 ----------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),


    # ---------- Classification Head ----------

    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# ==================================================
# 13. COMPILE
# ==================================================

model.compile(

    optimizer="adam",

    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),

    metrics=[
        "accuracy"
    ]

)


print("\nCNN architecture:")

model.summary()


# ==================================================
# 14. TRAIN
# ==================================================

print("\nStarting training...")


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    verbose=1

)


# ==================================================
# 15. FINAL TEST EVALUATION
# ==================================================

print(
    "\nEvaluating on the test dataset..."
)


test_loss, test_accuracy = model.evaluate(

    test_ds,

    verbose=1

)


print(
    f"\nTest loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ==================================================
# 16. SAVE MODEL
# ==================================================

model.save(
    MODEL_PATH
)


# ==================================================
# 17. SAVE CLASS MAPPING
# ==================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print("\nTraining completed.")

print(
    "Trained model saved to:",
    MODEL_PATH
)

print(
    "Class mapping saved to:",
    CLASS_NAMES_PATH
)