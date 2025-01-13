import React from 'react'

const ServicesList = () => {
  return (
    <div><section class="services" id="services">
    <div class="services-category">MILITARY WEB SOLUTIONS</div>
    <h2 class="services-title">
      Secure, efficient, and tailored for the Indian army
    </h2>
    <div class="services-grid">
      <div class="service-card">
        <img
          src="/images/Vehicle_Authentication.jpg"
          alt="Vehicle Authorization Dashboard"
        />
        <div class="service-card-content">
          <h3>Vehicle Authorization</h3>
          <p>
            Get a comprehensive view of all critical military operations.
          </p>
        </div>
      </div>
      <div class="service-card">
        <img
          src="/images/Intrusion_Detection.jpg"
          alt="Intrusion Detection Module"
        />
        <div class="service-card-content">
          <h3>Intrusion Detection</h3>
          <p>Stay alert with advanced intrusion detection capabilities.</p>
        </div>
      </div>
      <div class="service-card">
        <img
          src="/images/2FA.jpg"
          alt="Two-Factor Authentication"
        />
        <div class="service-card-content">
          <h3>Two-Factor Authentication</h3>
          <p>Enhance security with two-factor authentication methods.</p>
        </div>
      </div>
    </div>
  </section></div>
  )
}

export default ServicesList