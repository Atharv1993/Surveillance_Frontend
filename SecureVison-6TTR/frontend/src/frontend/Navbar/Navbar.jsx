import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const [showDropdown, setShowDropdown] = useState(false);
  
  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
  };

  return (
    <div>
      <nav className="navbar">
        <Link to="/" className="logo">
          6TTR
        </Link>
        <div className="nav-links">
          <Link to="/">Home</Link>
          <div className="dropdown">
            <button className="dropdown-toggle" onClick={toggleDropdown}>
              Records
              <span className="dropdown-arrow">▼</span>
            </button>
            {showDropdown && (
              <div className="dropdown-menu">
                <Link to="/face-records" onClick={() => setShowDropdown(false)}>
                  Face Recognition
                </Link>
                <Link to="/vehicle-records" onClick={() => setShowDropdown(false)}>
                  Vehicle Records
                </Link>
              </div>
            )}
          </div>
          <Link to="/services">Services</Link>
          <Link to="/login" id="logout-btn">
            LOGOUT
          </Link>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;