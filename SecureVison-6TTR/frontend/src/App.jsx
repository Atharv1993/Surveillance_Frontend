import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Authentication from './frontend/pages/Authentication';
import Homepage from './frontend/pages/Homepage';
import Group_Auth from './frontend/pages/Group_Auth';
import VehicleDetect from './frontend/pages/Vehicle_Detect';
import FaceRecordsDashboard from './frontend/pages/FaceRecordsDashboard';
import VehicleRecordsDashboard from './frontend/pages/VehicleRecordsDashboard ';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/authentication" element={<Authentication />} />
        <Route path="/group_auth" element={<Group_Auth/>}/>
        <Route path="/vehicle" element={<VehicleDetect/>}/>
        <Route path="/face-records" element={<FaceRecordsDashboard />} />
        <Route path="/vehicle-records" element={<VehicleRecordsDashboard />} />
      
      </Routes>
    </Router>
  );
};

export default App;