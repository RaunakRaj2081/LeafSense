import { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://10.14.21.188:8000"; // 👈 IMPORTANT

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    setFile(file);
    setResult(null);
    setPreview(URL.createObjectURL(file));
  };

  // 📸 START CAMERA
  const startCamera = async () => {
    setResult(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      alert("Camera not supported. Use HTTPS or localhost.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      alert("Camera permission denied");
    }
  };

  // 📸 CAPTURE PHOTO
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      const imageFile = new File([blob], "camera.jpg", {
        type: "image/jpeg",
      });
      handleFile(imageFile);
      stopCamera();
    });
  };

  // 🛑 STOP CAMERA
  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach((t) => t.stop());
    setCameraOn(false);
  };

  // 🧹 CLEAR
  const handleClear = () => {
    stopCamera();
    setFile(null);
    setPreview(null);
    setResult(null);
    setLoading(false);
  };

  // 📤 DRAG & DROP
  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // 🔮 PREDICT
  const handlePredict = async () => {
    if (!file) return alert("Please upload or capture an image");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Server error");

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Backend not reachable");
    } finally {
      setLoading(false);
    }
  };

  const getBarColor = (c) => {
    if (c > 0.75) return "#2ecc71";
    if (c > 0.5) return "#f39c12";
    return "#e74c3c";
  };

  return (
    <div className="container">
      <h1>🌿 Plant Disease Detection</h1>

      {/* CAMERA */}
      {cameraOn && (
        <div className="camera">
          <video ref={videoRef} autoPlay playsInline />
          <button onClick={capturePhoto}>📸 Capture</button>
        </div>
      )}

      {/* UPLOAD */}
      {!preview && !cameraOn && (
        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <p>Drag & drop image here</p>
          <span>or</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <button className="camera-btn" onClick={startCamera}>
            📱 Use Camera
          </button>
        </div>
      )}

      {/* PREVIEW */}
      {preview && (
        <div className="preview">
          <img src={preview} alt="preview" />
        </div>
      )}

      {/* BUTTONS */}
      <div className="button-group">
        {file && (
          <button className="clear-btn" onClick={handleClear}>
            Clear
          </button>
        )}
        <button onClick={handlePredict} disabled={loading || !file}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </div>

      {/* LOADING */}
      {loading && <div className="spinner"></div>}

      {/* RESULT */}
      {result && (
        <div className="result">
          <h3>{result.class}</h3>
          <div className="bar">
            <div
              className="fill"
              style={{
                width: `${result.confidence * 100}%`,
                backgroundColor: getBarColor(result.confidence),
              }}
            ></div>
          </div>
          <p className="confidence-text">
            {(result.confidence * 100).toFixed(2)}% confidence
          </p>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

export default App;
