import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import { useLocation, useNavigate } from "react-router-dom";

const BACKEND_URL = "http://127.0.0.1:5000";
const POLL_INTERVAL = 5000; // 5 seconds

const ToastContext = createContext();

export const useToast = () => useContext(ToastContext);

export const ToastProvider = ({ children }) => {
  const [lastImage, setLastImage] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const polling = useRef(null);

  // Poll for new images
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await axios.get(`${BACKEND_URL}/api/detection_images`);
        if (res.data && res.data.length > 0) {
          const newest = res.data[0];
          if (lastImage && newest.filename !== lastImage.filename) {
            // Only notify if not on Human_Detection page
            if (!location.pathname.startsWith("/human-detection")) {
              toast.info(
                <span>
                  New human detection image received!{" "}
                  <button
                    onClick={() => {
                      navigate("/human-detection");
                      toast.dismiss();
                    }}
                    style={{ marginLeft: 8, color: "blue", textDecoration: "underline", background: "none", border: "none", cursor: "pointer" }}
                  >
                    View
                  </button>
                </span>
              );
            }
          }
          setLastImage(newest);
        }
      } catch (e) {
        // ignore errors
      }
    };

    poll();
    polling.current = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(polling.current);
    // eslint-disable-next-line
  }, [location.pathname, lastImage]);

  return (
    <ToastContext.Provider value={{}}>
      <ToastContainer position="top-right" autoClose={5000} />
      {children}
    </ToastContext.Provider>
  );
};
