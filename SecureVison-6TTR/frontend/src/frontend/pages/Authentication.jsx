import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Autentication.css";

const Authentication = () => {
  const navigate = useNavigate();

  const [idType, setIdType] = useState("aadhar");
  const [cameraPopupOpen, setCameraPopupOpen] = useState(false);
  const [capturedIdImage, setCapturedIdImage] = useState(null);
  const [ocrPopupOpen, setOcrPopupOpen] = useState(false);
  const [ocrData, setOcrData] = useState({ name: "", roll: "" });
  const [proceedToFaceReg, setProceedToFaceReg] = useState(false);
  const [popupOpen, setPopupOpen] = useState(false);
  const [attendancePopupOpen, setAttendancePopupOpen] = useState(false);
  const [registrationResult, setRegistrationResult] = useState({
    userName: "",
    userImages: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [attendanceData, setAttendanceData] = useState([]);
  const [authMode, setAuthMode] = useState(false); // false for Register, true for Authenticate

  const Ipwebcam = "http://192.168.105.181:8080/shot.jpg";

  const instructionSteps = [
    "Ensure you are in a well-lit area with minimal background noise",
    "Look directly at the camera",
    "Keep your face centered",
    "Remain still during capture",
    "Avoid wearing hats or sunglasses",
  ];

  useEffect(() => {
    if (proceedToFaceReg && capturedIdImage) {
      if (authMode) {
        handleAuth();
      } else {
        handleRegister();
      }
      setProceedToFaceReg(false);
    }
  }, [proceedToFaceReg, capturedIdImage]);

  const handleCaptureID = async () => {
    try {
      const response = await axios.get(`${Ipwebcam}`, {
        responseType: "arraybuffer",
      });
      const base64Image = btoa(
        new Uint8Array(response.data).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ""
        )
      );

      setCapturedIdImage(base64Image);
      setCameraPopupOpen(false);
      handleOCR(base64Image);
    } catch (error) {
      console.error("Error capturing ID card:", error);
      window.alert("Failed to capture ID card. Please try again.");
    }
  };

  const handleOCR = async (image) => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/extract-id", {
        id_type: idType,
        id_image: image,
      });

      if (response.data.success) {
        setOcrData({ name: response.data.name, roll: response.data.roll });
        setOcrPopupOpen(true);
      } else {
        window.alert("OCR extraction failed");
      }
    } catch (error) {
      console.error("Error extracting OCR data:", error);
      window.alert("OCR extraction failed due to a server error");
    }
  };

  const handleRegister = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/register-face", {
        id_type: idType,
        name: ocrData.name,
        roll: ocrData.roll,
      });

      if (response.data.success) {
        setRegistrationResult({
          userName: response.data.userName,
          userImages: response.data.userImages,
        });
        setPopupOpen(true);
      } else {
        window.alert(response.data.message || "Registration failed");
      }
    } catch (error) {
      console.error("Error sending data:", error);
      window.alert("Registration failed due to a server error");
    }
  };

  const handleAuth = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/Authenticate", {
        id_type: idType,
        name: ocrData.name,
        roll: ocrData.roll,
      });

      if (response.data.success) {
        window.alert(
          `${response.data.status}, It's ${response.data.name}_${response.data.roll}`
        );
      } else {
        window.alert(response.data.message || "Authentication failed");
      }
    } catch (error) {
      console.error("Authentication error:", error);
      window.alert("Authentication failed due to a server error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGroupAuth = () => {
    navigate("/group_auth");
  };

  const fetchTodaysAttendance = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/todayattendance");

      if (response.data.success) {
        setAttendanceData(response.data.attendance);
        setAttendancePopupOpen(true);
      } else {
        window.alert("No attendance records found");
      }
    } catch (error) {
      console.error("Error fetching attendance:", error);
      window.alert("Failed to fetch attendance");
    }
  };

  const handleGoBack = () => navigate(-1);
  const handleGoHome = () => navigate("/");

  return (
    <div className="root">
      <div className="id-card-container">
        <div className="navigation-buttons">
          <button onClick={handleGoBack} className="back-btn">◀</button>
          <button onClick={handleGoHome} className="home-btn">🏠︎</button>
        </div>

        <h2 className="title">AUTHENTICATION PROCESS</h2>

        <div className="instructions-box">
          <h3 className="subtitle">Important Instructions</h3>
          <ol className="instruction-list">
            {instructionSteps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        </div>

        <div className="button-container">
          <button
            className="register-btn"
            onClick={() => {
              setAuthMode(false);
              setCameraPopupOpen(true);
            }}
          >
            Register
          </button>
          <button
            className="authenticate-btn"
            onClick={() => {
              setAuthMode(true);
              setCameraPopupOpen(true);
            }}
            disabled={isLoading}
          >
            {isLoading ? "Authenticating..." : "Authenticate"}
          </button>
          <button className="authenticate-btn" onClick={handleGroupAuth}>
            Group Authentication
          </button>
          <button className="attendance-btn" onClick={fetchTodaysAttendance}>
            Today's Records
          </button>
        </div>
      </div>

      {/* Registration Popup */}
      {popupOpen && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>Face Registration Successful</h2>
            <p>{registrationResult.userName}'s faces have been successfully stored.</p>
            <p>Number of images captured: {registrationResult.userImages}</p>
            <button onClick={() => setPopupOpen(false)}>Close</button>
          </div>
        </div>
      )}

      {/* Attendance Popup */}
      {attendancePopupOpen && (
        <div className="popup-overlay">
          <div className="popup-content attendance-popup">
            <div className="popup-header">
              <h2>Today's Attendance</h2>
              <button onClick={() => setAttendancePopupOpen(false)}>×</button>
            </div>
            <table className="attendance-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Roll Number</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {attendanceData.map((record, index) => (
                  <tr key={index}>
                    <td>{record[0]}</td>
                    <td>{record[1]}</td>
                    <td>{record[2]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Camera Popup */}
      {cameraPopupOpen && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>Select ID Type and Capture ID Card</h2>
            <div className="id-selection-box">
              <label htmlFor="idType" className="id-label">Select ID Type:</label>
              <select
                id="idType"
                value={idType}
                onChange={(e) => setIdType(e.target.value)}
                className="id-select"
              >
                <option value="aadhar">Aadhar Card</option>
                <option value="license">Driving License</option>
                <option value="college">College ID</option>
              </select>
            </div>

            <div className="webcam-container">
              <img src={`${Ipwebcam}`} alt="Webcam Feed" className="webcam-feed" />
            </div>

            <div className="popup-actions">
              <button onClick={handleCaptureID}>Capture ID Card</button>
              <button onClick={() => setCameraPopupOpen(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* OCR Verification Popup */}
      {ocrPopupOpen && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>Verify Extracted Information</h2>
            <label>Name:</label>
            <input
              type="text"
              value={ocrData.name}
              onChange={(e) => setOcrData({ ...ocrData, name: e.target.value })}
            />
            <label>Roll Number:</label>
            <input
              type="text"
              value={ocrData.roll}
              onChange={(e) => setOcrData({ ...ocrData, roll: e.target.value })}
            />
            <button onClick={() => {
              setOcrPopupOpen(false);
              setProceedToFaceReg(true);
            }}>
              Confirm & {authMode ? "Authenticate" : "Register"}
            </button>
            <button onClick={() => setOcrPopupOpen(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Authentication;
