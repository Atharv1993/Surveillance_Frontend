import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Authentication from './frontend/pages/Authentication';
import Homepage from './frontend/pages/Homepage';
import Group_Auth from './frontend/pages/Group_Auth';
import VehicleDetect from './frontend/pages/Vehicle_Detect';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/authentication" element={<Authentication />} />
        <Route path="/group_auth" element={<Group_Auth/>}/>
        <Route path="/vehicle" element={<VehicleDetect/>}/>
      </Routes>
    </Router>
  );
};

export default App;