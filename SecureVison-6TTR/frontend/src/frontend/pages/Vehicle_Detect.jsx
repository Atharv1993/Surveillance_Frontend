import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Vehicle_Detect.css"; // Import your CSS file for styling

const VehicleIdentificationSystem = () => {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [stream, setStream] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraPopupOpen, setCameraPopupOpen] = useState(false);
  const [registrationPopupOpen, setRegistrationPopupOpen] = useState(false);
  const [authenticationPopupOpen, setAuthenticationPopupOpen] = useState(false);
  const [authResultPopupOpen, setAuthResultPopupOpen] = useState(false);
  const [captureMode, setCaptureMode] = useState(""); // "register" or "authenticate"
  
  const [plateData, setPlateData] = useState({
    plateNumber: "",
    fullImage: "",
    plateImage: ""
  });
  
  const [vehicleData, setVehicleData] = useState({
    owner: "",
    vehicleType: "",
    color: "",
    model: ""
  });
  
  const [authResult, setAuthResult] = useState(null);
  
  const instructionSteps = [
    "Position the vehicle directly in front of the camera",
    "Ensure the license plate is clearly visible",
    "Make sure there's sufficient lighting on the plate",
    "Keep the camera steady during capture",
    "The plate should be within 1-3 meters from the camera"
  ];

  // Start webcam
  const startCamera = async () => {
    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      const userMedia = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(userMedia);
      
      if (videoRef.current) {
        videoRef.current.srcObject = userMedia;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(err => {
            console.error("Error playing video:", err);
          });
        };
      }
      setCameraActive(true);
    } catch (err) {
      console.error("Error accessing camera:", err);
      window.alert("Error accessing camera. Please check permissions.");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setCameraActive(false);
    }
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // Capture image from webcam
  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert to blob for upload
      return new Promise((resolve) => {
        canvas.toBlob(blob => {
          resolve(blob);
        }, "image/jpeg", 0.95);
      });
    }
    return null;
  };

  // Process image for plate detection
  const handleCaptureVehicle = async () => {
    if (!cameraActive) {
      window.alert("Please start the camera first");
      return;
    }
    
    setIsLoading(true);
    try {
      const imageBlob = await captureImage();
      if (!imageBlob) {
        throw new Error("Failed to capture image");
      }
      
      const formData = new FormData();
      formData.append("image", imageBlob, "vehicle.jpg");
      
      const response = await axios.post("http://127.0.0.1:5000/vehicle_plate/process_vehicle_image", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });
      
      if (response.data.success) {
        setPlateData({
          plateNumber: response.data.plate_number,
          fullImage: `data:image/jpeg;base64,${response.data.full_image}`,
          plateImage: `data:image/jpeg;base64,${response.data.plate_image}`
        });
        
        setCameraPopupOpen(false);
        stopCamera(); // Stop camera after successful capture
        
        if (captureMode === "register") {
          setRegistrationPopupOpen(true);
        } else if (captureMode === "authenticate") {
          setAuthenticationPopupOpen(true);
        }
      } else {
        window.alert(response.data.message);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      window.alert("Error processing image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Register vehicle
  const handleRegister = async () => {
    setIsLoading(true);
    try {
      const data = {
        plate_number: plateData.plateNumber,
        owner: vehicleData.owner,
        vehicle_type: vehicleData.vehicleType,
        color: vehicleData.color,
        model: vehicleData.model,
        full_image: plateData.fullImage.split(",")[1],
        plate_image: plateData.plateImage.split(",")[1]
      };
      
      const response = await axios.post("http://127.0.0.1:5000/vehicle_plate/register_vehicle", data);
      
      if (response.data.success) {
        window.alert("Vehicle registered successfully!");
        setRegistrationPopupOpen(false);
        resetForm();
      } else {
        window.alert(response.data.message || "Registration failed");
      }
    } catch (error) {
      console.error("Error registering vehicle:", error);
      window.alert("Error registering vehicle. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Authenticate vehicle
  const handleAuthenticate = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/vehicle_plate/authenticate_vehicle", {
        plate_number: plateData.plateNumber
      });
      
      setAuthResult(response.data);
      setAuthenticationPopupOpen(false);
      setAuthResultPopupOpen(true);
    } catch (error) {
      console.error("Error authenticating vehicle:", error);
      window.alert("Error authenticating vehicle. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Reset form
  const resetForm = () => {
    setPlateData({
      plateNumber: "",
      fullImage: "",
      plateImage: ""
    });
    setVehicleData({
      owner: "",
      vehicleType: "",
      color: "",
      model: ""
    });
  };

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    if (name === "plateNumber") {
      setPlateData({ ...plateData, plateNumber: value.toUpperCase() });
    } else {
      setVehicleData({ ...vehicleData, [name]: value });
    }
  };

  const handleGoBack = () => navigate(-1);
  const handleGoHome = () => navigate("/");
  
  // Handle camera popup open
  const openCameraPopup = (mode) => {
    setCaptureMode(mode);
    setCameraPopupOpen(true);
  };
  
  // Handle camera popup close
  const closeCameraPopup = () => {
    stopCamera();
    setCameraPopupOpen(false);
  };

  return (
    <div className="vis-root">
      <div className="vis-container">
        <div className="vis-navigation-buttons">
          <button onClick={handleGoBack} className="vis-back-btn">◀</button>
          <button onClick={handleGoHome} className="vis-home-btn">🏠︎</button>
        </div>

        <h2 className="vis-title">VEHICLE IDENTIFICATION SYSTEM</h2>

        <div className="vis-instructions-box">
          <h3 className="vis-subtitle">Important Instructions</h3>
          <ol className="vis-instruction-list">
            {instructionSteps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        </div>

        <div className="vis-button-container">
          <button
            className="vis-register-btn"
            onClick={() => openCameraPopup("register")}
            disabled={isLoading}
          >
            Register Vehicle
          </button>
          <button
            className="vis-authenticate-btn"
            onClick={() => openCameraPopup("authenticate")}
            disabled={isLoading}
          >
            {isLoading ? "Processing..." : "Authenticate Vehicle"}
          </button>
        </div>
      </div>

      {/* Camera Popup */}
      {cameraPopupOpen && (
        <div className="vis-popup-overlay">
          <div className="vis-popup-content">
            <h2>{captureMode === "register" ? "Register New Vehicle" : "Authenticate Vehicle"}</h2>
            
            <div className="vis-webcam-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="vis-webcam-feed"
                style={{ display: cameraActive ? "block" : "none" }}
              />
              <canvas ref={canvasRef} style={{ display: "none" }} />
              
              {!cameraActive && (
                <div className="vis-camera-placeholder">
                  <p>Camera is currently off</p>
                </div>
              )}
            </div>

            <div className="vis-camera-controls">
              {!cameraActive ? (
                <button 
                  onClick={startCamera}
                  className="vis-camera-btn vis-start-btn"
                >
                  Start Camera
                </button>
              ) : (
                <button 
                  onClick={stopCamera}
                  className="vis-camera-btn vis-stop-btn"
                >
                  Stop Camera
                </button>
              )}
            </div>

            <div className="vis-popup-actions">
              <button 
                onClick={handleCaptureVehicle}
                disabled={isLoading || !cameraActive}
                className="vis-capture-btn"
              >
                {isLoading ? "Processing..." : "Capture Vehicle"}
              </button>
              <button 
                onClick={closeCameraPopup}
                className="vis-cancel-btn"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Registration Form Popup */}
      {registrationPopupOpen && (
        <div className="vis-popup-overlay">
          <div className="vis-popup-content">
            <h2>Vehicle Registration</h2>
            
            <div className="vis-image-preview-container">
              <div className="vis-preview-section">
                <h3>Vehicle Image</h3>
                <img src={plateData.fullImage} alt="Vehicle" className="vis-preview-image" />
              </div>
              <div className="vis-preview-section">
                <h3>License Plate</h3>
                <img src={plateData.plateImage} alt="License Plate" className="vis-preview-image vis-plate-image" />
              </div>
            </div>
            
            <div className="vis-form-container">
              <div className="vis-form-group">
                <label>License Plate Number:</label>
                <input
                  type="text"
                  name="plateNumber"
                  value={plateData.plateNumber}
                  onChange={handleInputChange}
                />
              </div>
              
              <div className="vis-form-group">
                <label>Owner Name:</label>
                <input
                  type="text"
                  name="owner"
                  value={vehicleData.owner}
                  onChange={handleInputChange}
                />
              </div>
              
              <div className="vis-form-group">
                <label>Vehicle Type:</label>
                <select
                  name="vehicleType"
                  value={vehicleData.vehicleType}
                  onChange={handleInputChange}
                >
                  <option value="">Select Type</option>
                  <option value="Car">Car</option>
                  <option value="Bike">Bike</option>
                  <option value="Truck">Truck</option>
                  <option value="Van">Van</option>
                  <option value="Bus">Bus</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              
              <div className="vis-form-group">
                <label>Color:</label>
                <input
                  type="text"
                  name="color"
                  value={vehicleData.color}
                  onChange={handleInputChange}
                />
              </div>
              
              <div className="vis-form-group">
                <label>Model:</label>
                <input
                  type="text"
                  name="model"
                  value={vehicleData.model}
                  onChange={handleInputChange}
                />
              </div>
            </div>
            
            <div className="vis-popup-actions">
              <button 
                onClick={handleRegister}
                disabled={isLoading || !plateData.plateNumber || !vehicleData.owner || !vehicleData.vehicleType}
                className="vis-register-submit-btn"
              >
                {isLoading ? "Registering..." : "Register Vehicle"}
              </button>
              <button 
                onClick={() => setRegistrationPopupOpen(false)}
                className="vis-cancel-btn"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Authentication Form Popup */}
      {authenticationPopupOpen && (
        <div className="vis-popup-overlay">
          <div className="vis-popup-content">
            <h2>Vehicle Authentication</h2>
            
            <div className="vis-plate-preview-container">
              <h3>Detected License Plate</h3>
              <img src={plateData.plateImage} alt="License Plate" className="vis-preview-image vis-plate-image" />
            </div>
            
            <div className="vis-form-group">
              <label>License Plate Number:</label>
              <input
                type="text"
                name="plateNumber"
                value={plateData.plateNumber}
                onChange={handleInputChange}
              />
              <p className="vis-helper-text">Edit if recognition is not accurate</p>
            </div>
            
            <div className="vis-popup-actions">
              <button 
                onClick={handleAuthenticate}
                disabled={isLoading || !plateData.plateNumber}
                className="vis-verify-btn"
              >
                {isLoading ? "Verifying..." : "Verify Vehicle"}
              </button>
              <button 
                onClick={() => setAuthenticationPopupOpen(false)}
                className="vis-cancel-btn"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Authentication Result Popup */}
      {authResultPopupOpen && (
        <div className="vis-popup-overlay">
          <div className="vis-popup-content">
            <h2 className={authResult?.success ? "vis-success-title" : "vis-error-title"}>
              {authResult?.success ? "Access Granted" : "Access Denied"}
            </h2>
            
            {authResult?.success ? (
              <div className="vis-auth-result-container">
                <div className="vis-result-icon vis-success">✓</div>
                <h3>Vehicle Authenticated Successfully</h3>
                
                <div className="vis-vehicle-details">
                  <div className="vis-vehicle-image">
                    {authResult.vehicle.full_image && (
                      <img 
                        src={`data:image/jpeg;base64,${authResult.vehicle.full_image}`} 
                        alt="Vehicle" 
                        className="vis-preview-image"
                      />
                    )}
                  </div>
                  
                  <div className="vis-vehicle-info">
                    <p><strong>Plate Number:</strong> {authResult.vehicle.plate_number}</p>
                    <p><strong>Owner:</strong> {authResult.vehicle.owner}</p>
                    <p><strong>Vehicle Type:</strong> {authResult.vehicle.vehicle_type}</p>
                    {authResult.vehicle.color && (
                      <p><strong>Color:</strong> {authResult.vehicle.color}</p>
                    )}
                    {authResult.vehicle.model && (
                      <p><strong>Model:</strong> {authResult.vehicle.model}</p>
                    )}
                    <p><strong>Registered:</strong> {new Date(authResult.vehicle.registered_at).toLocaleString()}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="vis-auth-result-container">
                <div className="vis-result-icon vis-error">✗</div>
                <h3>Vehicle Not Recognized</h3>
                <p>The license plate <strong>{authResult?.plate_number}</strong> is not registered in the system.</p>
                <p>Please register this vehicle or verify the plate number.</p>
              </div>
            )}
            
            <button 
              onClick={() => setAuthResultPopupOpen(false)}
              className="vis-close-btn"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VehicleIdentificationSystem;