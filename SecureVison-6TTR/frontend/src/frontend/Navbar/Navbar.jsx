import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const [showRecordsDropdown, setShowRecordsDropdown] = useState(false);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);

  // Close dropdowns when another is opened
  const toggleRecordsDropdown = () => {
    setShowRecordsDropdown(!showRecordsDropdown);
    setShowServicesDropdown(false);
  };

  const toggleServicesDropdown = () => {
    setShowServicesDropdown(!showServicesDropdown);
    setShowRecordsDropdown(false);
  };

  // Optional: Close dropdowns when clicking anywhere else
  const handleBlur = () => {
    setTimeout(() => {
      setShowRecordsDropdown(false);
      setShowServicesDropdown(false);
    }, 150);
  };

  return (
    <div>
      <nav className="navbar">
        <Link to="/" className="logo">
          6TTR
        </Link>
        <div className="nav-links">
          <Link to="/">Home</Link>

          {/* Records Dropdown */}
          <div className="dropdown" tabIndex={0} onBlur={handleBlur}>
            <button
              className="dropdown-toggle"
              onClick={toggleRecordsDropdown}
              aria-haspopup="true"
              aria-expanded={showRecordsDropdown}
            >
              Records
              <span className="dropdown-arrow">▼</span>
            </button>
            {showRecordsDropdown && (
              <div className="dropdown-menu">
                <Link to="/face-records" onClick={() => setShowRecordsDropdown(false)}>
                  Face Recognition
                </Link>
                <Link to="/vehicle-records" onClick={() => setShowRecordsDropdown(false)}>
                  Vehicle Records
                </Link>
              </div>
            )}
          </div>

          {/* Services Dropdown */}
          <div className="dropdown" tabIndex={0} onBlur={handleBlur}>
            <button
              className="dropdown-toggle"
              onClick={toggleServicesDropdown}
              aria-haspopup="true"
              aria-expanded={showServicesDropdown}
            >
              Services
              <span className="dropdown-arrow">▼</span>
            </button>
            {showServicesDropdown && (
              <div className="dropdown-menu">
                <Link to="/authentication" onClick={() => setShowServicesDropdown(false)}>
                  Authentication
                </Link>
                <Link to="/vehicle" onClick={() => setShowServicesDropdown(false)}>
                  Vehicle Detect
                </Link>
                <Link to="/human-detection" onClick={() => setShowServicesDropdown(false)}>
                  Human Detection
                </Link>
              </div>
            )}
          </div>

          <Link to="/login" id="logout-btn">
            LOGOUT
          </Link>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
