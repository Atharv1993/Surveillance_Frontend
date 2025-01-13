import React from "react";
import "./App.css";
import Navbar from "./components/Navbar/Navbar";
import Hero from "./components/Hero/Hero";
import ServicesList from "./components/Services/Services";


function App() {
  

  return (
    <div><Navbar/>
    <Hero/>
    <ServicesList/>
    </div>
  );
}

export default App;
