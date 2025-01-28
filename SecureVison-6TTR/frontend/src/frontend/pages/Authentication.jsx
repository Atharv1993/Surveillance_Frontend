import React, { useState } from 'react';
import { data, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Autentication.css';

const Authentication = () => {
  const navigate = useNavigate();
  const [popupOpen, setPopupOpen] = useState(false);
  const [attendancePopupOpen, setAttendancePopupOpen] = useState(false);
  const [registrationResult, setRegistrationResult] = useState({
    userName: '',
    userImages: 0
  });
 
  const [isLoading, setIsLoading] = useState(false);
  const [attendanceData, setAttendanceData] = useState([]);

  const instructionSteps = [
    "Ensure you are in a well-lit area with minimal background noise",
    "Look directly at the camera",
    "Keep your face centered",
    "Remain still during capture",
    "Avoid wearing hats or sunglasses"
  ];

  const handleAuth = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/Authenticate', { data: 1 });
      
      if (response.data.success) {
        window.alert(response.data.status);
      } else {
        window.alert(response.data.status || 'Authentication failed');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      window.alert('Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/register-face', { data: 1 });
      
      if (response.data.success) {
        setRegistrationResult({
          userName: response.data.userName,
          userImages: response.data.userImages
        });
        setPopupOpen(true);
      }
    } catch (error) {
      console.error('Error sending data:', error);
    }
  };

  const handleGroupAuth = () => {
    navigate('/group_auth');
  }

  const fetchTodaysAttendance = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/api/todayattendance');
      
      console.log(response.data.attendance)
      if (response.data.success) {
        setAttendanceData(response.data.attendance);
        setAttendancePopupOpen(true);
      } else {
        window.alert('No attendance records found');
      }
    } catch (error) {
      console.error('Error fetching attendance:', error);
      window.alert('Failed to fetch attendance');
    }
  };
      
  const handleGoBack = () => navigate(-1);
  const handleGoHome = () => navigate('/');

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
          <button className="register-btn" onClick={handleRegister}>Register</button>
          <button 
            className="authenticate-btn" 
            onClick={handleAuth} 
            disabled={isLoading}
          >
            {isLoading ? 'Authenticating...' : 'Authenticate'}
          </button>
          <button 
            className="authenticate-btn" 
            onClick={handleGroupAuth} 
          >
            Group Authentication
          </button>
          <button 
            className="attendance-btn" 
            onClick={fetchTodaysAttendance}
          >
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
    </div>
  );
};

export default Authentication;