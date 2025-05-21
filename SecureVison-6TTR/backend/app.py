from flask import Flask
from flask_cors import CORS
import atexit
import os

# Set OpenMP environment variables to avoid conflicts BEFORE importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from modules.camera_manager.camera_manager import camera_manager
from modules.face_recognition import face_recognition_bp
from modules.vehicle_identification.vehicle_identification import vehicle_plate_bp

app = Flask(__name__)
CORS(app)

camera_manager.set_camera_index(0)  # Set the camera index to 1 (or your laptop's front camera index)

# Register blueprints
app.register_blueprint(face_recognition_bp, url_prefix='/face_recog')
app.register_blueprint(vehicle_plate_bp, url_prefix='/vehicle_plate')

# Clean up camera resources on application exit
def cleanup_resources():
    print("Cleaning up camera resources...")
    camera_manager.cleanup()

atexit.register(cleanup_resources)

if __name__ == '__main__':
    app.run(debug=True) 