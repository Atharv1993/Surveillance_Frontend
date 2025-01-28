import React from 'react'
import './Navbar.css'

const Navbar = () => {
  return (
    <div>
          <nav className="navbar">
        <a href="#" className="logo">
          6TTR
        </a>
        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#Records">Records</a>
          <a href="#services">Services</a>
          <a href="#"  id="logout-btn">
            LOGOUT
          </a>
        </div>
      </nav>
    </div>
  )
}
export default Navbar
