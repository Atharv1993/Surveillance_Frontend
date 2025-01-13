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
          src="https://cdn.b12.io/client_media/QfcKELDP/a47f8f7a-d0a6-11ef-8fc1-0242ac110002-jpg-hero_image.jpeg"
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
          src="https://cdn.b12.io/client_media/QfcKELDP/a4df5842-d0a6-11ef-8fc1-0242ac110002-jpg-hero_image.jpeg"
          alt="Intrusion Detection Module"
        />
        <div class="service-card-content">
          <h3>Intrusion Detection</h3>
          <p>Stay alert with advanced intrusion detection capabilities.</p>
        </div>
      </div>
      <div class="service-card">
        <img
          src="https://cdn.b12.io/client_media/QfcKELDP/a4b1b86a-d0a6-11ef-8fc1-0242ac110002-jpg-hero_image.jpeg"
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