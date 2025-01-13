import React, { useState } from 'react';
import './Navbar.css';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <div>
      <nav className="navbar">
        <a href="#" className="logo">
          6TTR
        </a>
        <div className={`nav-links ${isMenuOpen ? 'open' : ''}`}>
          <a href="#home">Home</a>
          <a href="#Records">Records</a>
          <a href="#services">Services</a>
          <a href="#" id="logout-btn">
            LOGOUT
          </a>
        </div>
        <div className="hamburger" onClick={toggleMenu}>
          <div className="bar"></div>
          <div className="bar"></div>
          <div className="bar"></div>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
