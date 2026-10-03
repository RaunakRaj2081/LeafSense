import React, {
  useEffect,
  useRef,
  useState
} from "react";

import "./App.css";


const API_URL = "http://localhost:8000";


function App() {

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);


  // ==================================================
  // HANDLE FILE
  // ==================================================

  const handleFile = (selectedFile) => {

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);
    setResult(null);

    setPreview((oldPreview) => {

      if (oldPreview) {
        URL.revokeObjectURL(oldPreview);
      }

      return URL.createObjectURL(selectedFile);
    });
  };


  // ==================================================
  // START CAMERA
  // ==================================================

  const startCamera = async () => {

    try {

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "environment"
            }
          },
          audio: false
        });

      streamRef.current = stream;

      setCameraOn(true);

    } catch (error) {

      console.error(
        "Camera error:",
        error
      );

      alert(
        "Unable to access camera. Please allow camera permission."
      );
    }
  };


  // ==================================================
  // CONNECT VIDEO STREAM
  // ==================================================

  useEffect(() => {

    if (
      cameraOn &&
      videoRef.current &&
      streamRef.current
    ) {

      videoRef.current.srcObject =
        streamRef.current;
    }

  }, [cameraOn]);


  // ==================================================
  // STOP CAMERA
  // ==================================================

  const stopCamera = () => {

    if (streamRef.current) {

      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };


  // ==================================================
  // CAPTURE PHOTO
  // ==================================================

  const capturePhoto = () => {

    if (
      !videoRef.current ||
      !canvasRef.current
    ) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context =
      canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      (blob) => {

        if (!blob) {
          return;
        }

        const capturedFile =
          new File(
            [blob],
            "camera.jpg",
            {
              type: "image/jpeg"
            }
          );

        handleFile(capturedFile);

        stopCamera();
      },
      "image/jpeg",
      0.95
    );
  };


  // ==================================================
  // CLEAR
  // ==================================================

  const handleClear = () => {

    stopCamera();

    setFile(null);
    setPreview(null);
    setResult(null);
    setLoading(false);
  };


  // ==================================================
  // PREDICT
  // ==================================================

  const handlePredict = async () => {

    if (!file) {
      alert("Please select or capture a leaf image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {

      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      const response =
        await fetch(
          `${API_URL}/predict`,
          {
            method: "POST",
            body: formData
          }
        );


      if (!response.ok) {

        throw new Error(
          `Server returned ${response.status}`
        );
      }


      const data =
        await response.json();

      setResult(data);

    } catch (error) {

      console.error(
        "Prediction error:",
        error
      );

      alert(
        "Backend not reachable. Please make sure the FastAPI server is running."
      );

    } finally {

      setLoading(false);
    }
  };


  // ==================================================
  // DRAG & DROP
  // ==================================================

  const handleDragOver = (event) => {

    event.preventDefault();

    setDragging(true);
  };


  const handleDragLeave = () => {

    setDragging(false);
  };


  const handleDrop = (event) => {

    event.preventDefault();

    setDragging(false);

    const droppedFile =
      event.dataTransfer.files[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  };


  // ==================================================
  // CLEANUP
  // ==================================================

  useEffect(() => {

    return () => {

      if (streamRef.current) {

        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });
      }

      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };

  }, [preview]);


  // ==================================================
  // UI
  // ==================================================

  return (
    <div className="app">

      <div className="container">

        <h1>
          Plant Disease Detection
        </h1>

        <p className="subtitle">
          Upload or capture a potato leaf image
          to detect its condition.
        </p>


        {/* ==========================================
            UPLOAD AREA
        ========================================== */}

        {!cameraOn && (

          <div
            className={`dropzone ${
              dragging ? "dragging" : ""
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >

            <input
              type="file"
              accept="image/*"
              id="fileInput"
              onChange={(event) => {

                const selectedFile =
                  event.target.files[0];

                if (selectedFile) {
                  handleFile(selectedFile);
                }

              }}
            />

            <label htmlFor="fileInput">

              <div className="upload-icon">
                📷
              </div>

              <p>
                Drag & drop an image here
              </p>

              <span>
                or click to browse
              </span>

            </label>

          </div>
        )}


        {/* ==========================================
            CAMERA
        ========================================== */}

        {cameraOn && (

          <div className="camera-container">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="camera-video"
            />

            <button
              onClick={capturePhoto}
              className="primary-button"
            >
              Capture Photo
            </button>

            <button
              onClick={stopCamera}
              className="secondary-button"
            >
              Stop Camera
            </button>

          </div>
        )}


        <canvas
          ref={canvasRef}
          style={{
            display: "none"
          }}
        />


        {/* ==========================================
            CAMERA BUTTON
        ========================================== */}

        {!cameraOn && (

          <button
            onClick={startCamera}
            className="camera-button"
          >
            Use Camera
          </button>
        )}


        {/* ==========================================
            IMAGE PREVIEW
        ========================================== */}

        {preview && (

          <div className="preview-section">

            <h2>
              Preview
            </h2>

            <img
              src={preview}
              alt="Leaf preview"
              className="preview-image"
            />

          </div>
        )}


        {/* ==========================================
            ACTION BUTTONS
        ========================================== */}

        {file && (

          <div className="actions">

            <button
              onClick={handlePredict}
              disabled={loading}
              className="predict-button"
            >

              {loading
                ? "Predicting..."
                : "Predict Disease"}

            </button>


            <button
              onClick={handleClear}
              className="clear-button"
              disabled={loading}
            >
              Clear
            </button>

          </div>
        )}


        {/* ==========================================
            LOADING
        ========================================== */}

        {loading && (

          <div className="loading">

            <div className="spinner"></div>

            <p>
              Analyzing leaf image...
            </p>

          </div>
        )}


        {/* ==========================================
            RESULT
        ========================================== */}

        {result && !loading && (

          <div className="result">

            <h2>
              Prediction Result
            </h2>

            <div className="result-class">

              {result.class}

            </div>


            <div className="confidence-section">

              <div className="confidence-header">

                <span>
                  Confidence
                </span>

                <span>
                  {(result.confidence * 100).toFixed(2)}%
                </span>

              </div>


              <div className="confidence-bar">

                <div
                  className="confidence-fill"
                  style={{
                    width: `${
                      result.confidence * 100
                    }%`
                  }}
                />

              </div>

            </div>

          </div>
        )}

      </div>

    </div>
  );
}


export default App;