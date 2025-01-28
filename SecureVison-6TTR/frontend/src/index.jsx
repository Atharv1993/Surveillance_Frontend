import React from "react";
import ReactDOM from "react-dom/client"; // Note the change here
// import { BrowserRouter } from "react-router-dom";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root")); // New API
root.render(
  <React.StrictMode>

      <App />
 
  </React.StrictMode>
);
