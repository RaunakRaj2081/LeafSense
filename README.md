# LeafSense

**LeafSense** is a web-based plant disease detection application that uses a Convolutional Neural Network (CNN) to classify potato leaf images into three categories:

* Early Blight
* Late Blight
* Healthy

The application provides a simple React interface where users can upload an image, drag and drop an image, or capture a leaf image using their camera. The image is then sent to a FastAPI backend, where the trained CNN performs inference and returns the predicted class along with its confidence score.

---

## Features

* Upload potato leaf images
* Drag-and-drop image upload
* Capture images directly using the device camera
* Image preview before prediction
* CNN-based plant disease classification
* Confidence score for predictions
* FastAPI REST API for model inference
* React-based interactive frontend
* Responsive user interface
* Prediction result with visual confidence bar
* Clear/reset functionality

---

## Tech Stack

### Frontend

* React.js
* JavaScript
* CSS

### Backend

* Python
* FastAPI
* Uvicorn
* Pillow
* NumPy

### Machine Learning

* TensorFlow
* Keras
* Convolutional Neural Network (CNN)
* Scikit-learn

---

## System Architecture

```text
                User
                  |
                  v
        React Frontend
                  |
        Upload / Camera
                  |
                  v
            FormData
                  |
                  v
       POST /predict API
                  |
                  v
         FastAPI Backend
                  |
          Image Processing
                  |
                  v
        NumPy Image Array
                  |
                  v
       Trained CNN Model
          (model.keras)
                  |
                  v
       Class Prediction
        + Confidence Score
                  |
                  v
          JSON Response
                  |
                  v
          React Frontend
                  |
                  v
        Prediction Display
```

---

## End-to-End Workflow

1. The user uploads a potato leaf image or captures one using the camera.
2. React stores the selected image as a `File` object.
3. The frontend creates a `FormData` object and appends the image under the `file` field.
4. React sends a `POST` request to the FastAPI `/predict` endpoint.
5. FastAPI receives the image through `UploadFile`.
6. The backend reads the uploaded bytes and converts the image into RGB format using Pillow.
7. The image is converted into a NumPy array.
8. A batch dimension is added before passing the image to the CNN.
9. The trained Keras model performs inference.
10. The class with the highest predicted probability is selected.
11. The corresponding confidence score is calculated.
12. FastAPI returns the predicted class and confidence as JSON.
13. React receives the response and displays the prediction and confidence percentage.

---

## CNN Architecture

The trained model uses the following architecture:

```text
Input Image
224 × 224 × 3
       |
       v
Resizing(224, 224)
       |
       v
Rescaling(1/255)
       |
       v
Conv2D(32, 3×3, ReLU)
       |
       v
MaxPooling2D
       |
       v
Conv2D(64, 3×3, ReLU)
       |
       v
MaxPooling2D
       |
       v
Conv2D(64, 3×3, ReLU)
       |
       v
MaxPooling2D
       |
       v
Flatten
       |
       v
Dense(64, ReLU)
       |
       v
Dense(3, Softmax)
```

### Model Configuration

* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Output activation: Softmax
* Epochs: 50
* Batch size: 32
* Input size: 224 × 224
* Pixel scaling: 1/255
* Training augmentation:

  * Random horizontal flip
  * Random vertical flip
  * Random rotation

The original dataset images are 256 × 256 RGB images and are resized to 224 × 224 before being processed by the CNN.

---

## Dataset

The model was trained using the **PlantVillage potato leaf dataset**.

The three classes used in the project are:

| Class        | Number of Images |
| ------------ | ---------------: |
| Early Blight |             1000 |
| Late Blight  |             1000 |
| Healthy      |              152 |
| **Total**    |         **2152** |

The dataset contains a class imbalance because the Healthy class has fewer images than the two disease classes.

The dataset is not included in this repository.

---

## Train / Validation / Test Split

A stratified image-level split was used to preserve class proportions across the datasets.

| Dataset    | Images |
| ---------- | -----: |
| Training   |   1721 |
| Validation |    215 |
| Test       |    216 |

Test-set distribution:

| Class        | Test Images |
| ------------ | ----------: |
| Early Blight |         100 |
| Late Blight  |         100 |
| Healthy      |          16 |

---

## Model Evaluation

The final model achieved:

**Test Accuracy: 96.30%**

### Classification Report

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Early Blight |    1.0000 | 0.9900 |   0.9950 |
| Late Blight  |    0.9792 | 0.9400 |   0.9592 |
| Healthy      |    0.7143 | 0.9375 |   0.8108 |

### Confusion Matrix

```text
                 Predicted
              Early  Late  Healthy

Actual Early    99     1      0
Actual Late      0    94      6
Actual Healthy   0     1     15
```

The evaluation shows particularly strong recognition of Early Blight. The model correctly identified 15 of the 16 Healthy test images. Some Late Blight images were classified as Healthy, which is reflected in the class-wise precision and recall values.

---

## API

### `GET /ping`

Used to verify that the FastAPI server is running.

### `POST /predict`

Accepts an uploaded image and returns the predicted disease class and confidence.

Example response:

```json
{
  "class": "Early Blight",
  "confidence": 0.98
}
```

---

## Project Structure

```text
LeafSense/
│
├── Backend/
│   ├── main.py
│   ├── model.py
│   ├── evaluate.py
│   ├── evaluate_stratified.py
│   ├── stratified_evaluate.py
│   ├── predict_test.py
│   ├── model.keras
│   ├── class_names.json
│   └── requirments.txt
│
├── FRONTEND/
│   └── plant-disease-ui/
│       ├── public/
│       │   ├── index.html
│       │   └── favicon.png
│       │
│       ├── src/
│       │   ├── App.js
│       │   ├── App.css
│       │   └── ...
│       │
│       ├── package.json
│       └── package-lock.json
│
├── .gitignore
└── README.md
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd LeafSense
```

### 2. Create and activate the Python virtual environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
pip install -r Backend\requirments.txt
```

---

## Running the Backend

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir Backend
```

The backend will run at:

```text
http://localhost:8000
```

---

## Running the Frontend

Open a second terminal:

```powershell
cd FRONTEND\plant-disease-ui
npm install
npm start
```

The frontend will run at:

```text
http://localhost:3000
```

Keep both the backend and frontend terminals running while using the application.

---

## Model

The trained CNN model is stored as:

```text
Backend/model.keras
```

The class-name mapping is stored in:

```text
Backend/class_names.json
```

---

## Limitations

* The model is trained specifically for three potato leaf categories.
* The dataset contains class imbalance.
* Model predictions depend on image quality, lighting, background, and similarity to the training data.
* The application should be treated as a machine-learning classification system rather than a replacement for professional agricultural diagnosis.

---

## Future Scope

Possible improvements include:

* Expanding the number of plant species and disease classes
* Increasing the size and diversity of the training dataset
* Using transfer learning with pretrained CNN architectures
* Improving robustness to different lighting and backgrounds
* Deploying the FastAPI backend to a cloud platform
* Adding prediction history
* Adding additional agricultural information for detected diseases

---

## License

This project is intended for educational and research purposes.
