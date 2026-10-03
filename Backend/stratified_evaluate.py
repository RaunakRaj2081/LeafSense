import os
from collections import Counter

from sklearn.model_selection import train_test_split


# ==================================================
# 1. DATASET PATH
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


# ==================================================
# 2. COLLECT IMAGE PATHS AND LABELS
# ==================================================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

image_paths = []
labels = []


for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):

            image_paths.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )

            labels.append(class_index)


print("Total images:", len(image_paths))


# ==================================================
# 3. ORIGINAL CLASS DISTRIBUTION
# ==================================================

print("\nOriginal distribution:")

for class_index, class_name in enumerate(class_names):

    count = labels.count(class_index)

    print(
        f"{class_name}: {count}"
    )


# ==================================================
# 4. FIRST SPLIT
#    80% TRAIN
#    20% TEMPORARY
# ==================================================

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=123,
        stratify=labels
    )
)


# ==================================================
# 5. SECOND SPLIT
#    10% VALIDATION
#    10% TEST
# ==================================================

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=123,
        stratify=temp_labels
    )
)


# ==================================================
# 6. PRINT SPLIT SIZES
# ==================================================

print("\nSplit sizes:")

print(
    "Training:",
    len(train_paths)
)

print(
    "Validation:",
    len(val_paths)
)

print(
    "Testing:",
    len(test_paths)
)


# ==================================================
# 7. PRINT CLASS DISTRIBUTION
# ==================================================

def print_distribution(
    name,
    labels_list
):

    counts = Counter(
        labels_list
    )

    print(f"\n{name} distribution:")

    for index, class_name in enumerate(class_names):

        print(
            f"{class_name}: "
            f"{counts[index]}"
        )


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