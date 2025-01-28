import React from 'react'
import { useNavigate } from 'react-router-dom'

const ServicesList = () => {
  const navigate = useNavigate();

  const handleAuthenticationClick = () => {
    navigate('/authentication');
  };

  return (
    <div>
      <section className="services" id="services">
        <div className="services-category">MILITARY WEB SOLUTIONS</div>
        <h2 className="services-title">
          Secure, efficient, and tailored for the Indian army
        </h2>
        <div className="services-grid">
          <div className="service-card1">
            <img
              src="/images/Vehicle_Authentication.jpg"
              alt="Vehicle Authorization Dashboard"
            />
            <div className="service-card-content">
              <h3>Vehicle Authorization</h3>
              <p>
                Get a comprehensive view of all critical military operations.
              </p>
            </div>
          </div>
          <div className="service-card2">
            <img
              src="/images/Intrusion_Detection.jpg"
              alt="Intrusion Detection Module"
            />
            <div className="service-card-content">
              <h3>Intrusion Detection</h3>
              <p>Stay alert with advanced intrusion detection capabilities.</p>
            </div>
          </div>
          <div 
            className="service-card3" 
            onClick={handleAuthenticationClick}
          >
            <img
              src="/images/2FA.jpg"
              alt="Two-Factor Authentication"
            />
            <div className="service-card-content">
              <h3>Two-Factor Authentication</h3>
              <p>Enhance security with two-factor authentication methods.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default ServicesList