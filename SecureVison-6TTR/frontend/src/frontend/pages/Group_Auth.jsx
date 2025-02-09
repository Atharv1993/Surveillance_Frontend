import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./Group_Auth.css";

const Group_Auth = () => {
  const navigate = useNavigate();
  const hasRun = useRef(false); // Prevent multiple API calls
  const [videoSrc, setVideoSrc] = useState(""); // Store video feed URL

  // Function to start the video feed
  const startVideoFeed = () => {
    axios
      .post("http://localhost:5000/api/start_video")
      .then(() => {
        // Append timestamp to force reload and prevent caching
        setVideoSrc(`http://localhost:5000/api/video_feed?timestamp=${new Date().getTime()}`);
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
        setVideoSrc(""); // Clear video source
        navigate("/authentication"); // Navigate to another page
      })
      .catch((error) => {
        console.error("Error stopping video feed:", error);
      });
  };

  // Start video feed when the component mounts
  useEffect(() => {
    if (!hasRun.current) {
      startVideoFeed();
      hasRun.current = true;
    }
    return () => setVideoSrc(""); // Cleanup: Reset src when unmounting
  }, []);

  return (
    <div className="Group_div">
      <h1>Group Recognition</h1>
      <img id="video-feed" src={videoSrc} alt="Live Video Feed" />
      <button onClick={markAttendance}>Mark Attendance</button>
      <button onClick={stopVideoFeed}>Stop Video Feed</button>
    </div>
  );
};

export default Group_Auth;
