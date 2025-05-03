import cv2
import threading
import time
import os

# Set OpenMP environment variables here as well for consistency
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class CameraManager:
    """
    Singleton class to manage camera access across different blueprints
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance.camera = None
                cls._instance.in_use = False
                cls._instance.camera_index = 1  # Default camera index
                cls._instance.last_access = None
                cls._instance.camera_lock = threading.RLock()  # Reentrant lock
                cls._instance.access_timeout = 10  # Increased timeout for operations
            return cls._instance
    
    def set_camera_index(self, index):
        """Configure which camera index to use"""
        with self.camera_lock:
            self.camera_index = index
            # If camera is already open, close and reopen with new index
            if self.camera is not None:
                self.release_camera()
    
    def get_camera(self):
        """Get camera object with exclusive access"""
        with self.camera_lock:
            # If camera is in use by another process, wait for release
            timeout = self.access_timeout
            start_time = time.time()
            
            while self.in_use and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.in_use:
                print("Camera access timeout - camera appears to be in use")
                # Force release if timeout occurs (prevents deadlock)
                self.release_camera()
            
            # If camera is not initialized, initialize it
            if self.camera is None:
                # Try multiple times to open the camera
                for attempt in range(3):
                    try:
                        self.camera = cv2.VideoCapture(self.camera_index)
                        if self.camera.isOpened():
                            print(f"Successfully opened camera at index {self.camera_index}")
                            break
                        else:
                            print(f"Attempt {attempt+1} failed to open camera at index {self.camera_index}")
                            time.sleep(1)
                    except Exception as e:
                        print(f"Error opening camera: {e}")
                        time.sleep(1)
                
                if not self.camera or not self.camera.isOpened():
                    print(f"Failed to open camera at index {self.camera_index} after multiple attempts")
                    self.camera = None
                    return None
                
                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            self.in_use = True
            self.last_access = time.time()
            return self.camera
    
    def release_camera(self):
        """Release the camera for other processes to use"""
        with self.camera_lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    print(f"Error releasing camera: {e}")
                self.camera = None
            self.in_use = False
    
    def capture_frame(self):
        """Capture a single frame and immediately release the camera"""
        with self.camera_lock:
            camera = self.get_camera()
            
            if camera is None:
                return False, None
            
            # Try multiple times to read a frame
            for attempt in range(3):
                try:
                    ret, frame = camera.read()
                    if ret:
                        break
                    print(f"Attempt {attempt+1} failed to read frame")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error reading frame: {e}")
                    time.sleep(0.5)
            
            self.release_camera()
            
            if not ret:
                print("Failed to capture a valid frame after multiple attempts")
                
            return ret, frame
    
    def cleanup(self):
        """Clean up camera resources"""
        with self.camera_lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    print(f"Error during cleanup: {e}")
                self.camera = None
            self.in_use = False

# Create a global instance of the camera manager
camera_manager = CameraManager()