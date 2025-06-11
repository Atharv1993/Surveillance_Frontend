import React, { useEffect, useState } from "react";
import axios from "axios";

const BACKEND_URL = "http://127.0.0.1:5000";

const Human_Detection = () => {
  const [images, setImages] = useState([]);

  const fetchImages = async () => {
    const res = await axios.get(`${BACKEND_URL}/api/detection_images`);
    setImages(res.data);
  };

  useEffect(() => {
    fetchImages();
    // Optionally, refresh every 10 seconds
    const interval = setInterval(fetchImages, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Human Detection Images</h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "16px" }}>
        {images.map((img) => (
          <div key={img.filename} style={{ border: "1px solid #ccc", padding: 8 }}>
            <img
              src={`${BACKEND_URL}${img.url}`}
              alt={img.filename}
              style={{ maxWidth: 200, maxHeight: 200 }}
            />
            <div>{img.timestamp ? new Date(img.timestamp).toLocaleString() : "Unknown time"}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Human_Detection;
