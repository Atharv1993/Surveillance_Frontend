import React from 'react'
import './Homepage.css';
import Navbar from '../Navbar/Navbar';
import Hero from '../Hero/Hero';
import ServicesList from '../Services/ServicesList';

const Homepage = () => {
  return (
    <div><Navbar/>
    <Hero/>
    <ServicesList/>
    </div>
  )
}

export default Homepage