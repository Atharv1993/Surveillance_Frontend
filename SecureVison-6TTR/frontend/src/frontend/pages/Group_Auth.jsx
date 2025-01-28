import React, { useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./Group_Auth.css";

const Group_Auth = () => {
  const navigate = useNavigate();

  // Function to start the video feed
  const startVideoFeed = () => {
    axios
      .post("http://localhost:5000/api/start_video")
      .then((response) => {
        console.log(response.data.message); // Log success message
      })
      .catch((error) => {
        console.error("Error starting video feed:", error);
      });
  };

  const markAttendance = () => {
    axios
      .post("http://localhost:5000/api/group_markattendance")
      .then((response) => {
        alert(response.data.message);
      })
      .catch((error) => {
        console.error("Error marking attendance:", error);
      });
  };

  const stopVideoFeed = () => {
    axios
      .post("http://localhost:5000/api/stop_video")
      .then((response) => {
        alert(response.data.message);
        navigate("/authentication");
      })
      .catch((error) => {
        console.error("Error stopping video feed:", error);
      });
  };

  // Use useEffect to start the video feed when the component mounts
  useEffect(() => {
    startVideoFeed();
  }, []);

  return (
    <div className="Group_div">
      <h1>Group Recognition</h1>
      <img
        id="video-feed"
        src="http://localhost:5000/api/video_feed"
        alt="Live Video Feed"
      />
      <button onClick={markAttendance}>Mark Attendance</button>
      <button onClick={stopVideoFeed}>Stop Video Feed</button>
    </div>
  );
};

export default Group_Auth;
