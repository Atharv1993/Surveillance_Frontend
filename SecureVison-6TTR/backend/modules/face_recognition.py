from flask import Blueprint, Response, jsonify, request
import cv2
import faiss
import numpy as np
import os
import csv
import json
import pandas as pd
import requests
from datetime import datetime
import threading
import time

# Set OpenMP environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import dependencies after environment variables are set
from deepface import DeepFace
from modules.ocr_service import extract_ocr_data
from modules.camera_manager.camera_manager import camera_manager

#Mongo Connection
from pymongo import MongoClient
from bson.binary import Binary
import base64
from io import BytesIO
from PIL import Image

face_recognition_bp = Blueprint("face_recognition", __name__)

MONGO_URI = "mongodb://localhost:27017" 
client = MongoClient(MONGO_URI)
db = client["Smart_Surveillance"]
face_collection = db["face_metadata"]

# Directories
attendance_directory = "Attendance"
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

# Initialize FAISS with 512D embeddings
embedding_dim = 512
faiss_index = faiss.IndexFlatIP(embedding_dim)
embeddings_db = {}

embeddings_file = "embeddings.json"

# Load stored embeddings if available
if os.path.exists(embeddings_file):
    with open(embeddings_file, "r") as f:
        embeddings_db = json.load(f)

    if embeddings_db:
        stored_embeddings = np.array(list(embeddings_db.values()), dtype=np.float32)
        faiss_index.add(stored_embeddings)

print(f"Loaded {len(embeddings_db)} embeddings into FAISS.")

# Thread lock for video feed
video_lock = threading.Lock()

def get_date_today():
    return datetime.now().strftime("%Y-%m-%d")

def save_attendance(name, roll):
    filename = f"{attendance_directory}/Attendance-{get_date_today()}.csv"
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=["Name", "Roll", "Time"])
    else:
        df = pd.read_csv(filename)
    
    if roll not in df["Roll"].values:
        now = datetime.now().strftime("%H:%M:%S")
        df = pd.concat([df, pd.DataFrame([[name, roll, now]], columns=["Name", "Roll", "Time"])]).reset_index(drop=True)
        df.to_csv(filename, index=False)

def extract_face_embedding(image):
    try:
        # Added retry mechanism for more reliability
        for attempt in range(3):
            try:
                embedding = DeepFace.represent(image, model_name="ArcFace", enforce_detection=True)[0]["embedding"]
                embedding = np.array(embedding, dtype=np.float32)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding
                return embedding
            except Exception as e:
                if attempt < 2:  # Try again if not the last attempt
                    print(f"Face embedding extraction attempt {attempt+1} failed: {e}")
                    time.sleep(0.5)
                else:
                    raise e
    except Exception as e:
        print(f"Face embedding extraction error after all attempts: {e}")
        return None

@face_recognition_bp.route('/api/extract-id', methods=['POST'])
def extract_id():
    data = request.json
    id_type = data.get("id_type")  # Accept id_type as input
    id_image_base64 = data.get("id_image")  # Captured ID as base64

    # Decode Base64 ID Image
    if id_image_base64:
        img_data = base64.b64decode(id_image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return jsonify({'success': False, 'message': 'No ID image provided'})

    ocr_data = extract_ocr_data(frame, id_type)
    new_username = ocr_data["name"]
    new_userid = str(ocr_data["id"])
    
    return jsonify({"success": True, "name": new_username, "roll": new_userid})

@face_recognition_bp.route('/api/register-face', methods=['POST'])
def register_face():
    data = request.json
    new_username = data.get("name", "").upper()  
    new_userid = data.get("roll", "")  
    id_type = data.get("id_type", "")
    
    if not new_username or not new_userid:
        return jsonify({'success': False, 'message': 'Invalid user during face details'})

    # Capture a single frame using the camera manager
    for attempt in range(3):
        ret, frame = camera_manager.capture_frame()
        if ret and frame is not None:
            break
        print(f"Attempt {attempt+1} to capture frame failed")
        time.sleep(1)
    
    if not ret or frame is None:
        return jsonify({'success': False, 'message': 'Camera error - could not capture frame'})

    # Extract face embedding
    embedding = extract_face_embedding(frame)
    if embedding is None:
        return jsonify({'success': False, 'message': 'No face detected'})

    # Save embedding in FAISS
    user_key = f"{new_username}_{new_userid}"
    if user_key in embeddings_db:
        return jsonify({'success': False, 'message': 'User already registered'})

    faiss_index.add(np.array([embedding], dtype=np.float32))
    embeddings_db[user_key] = embedding.tolist()  # Convert NumPy array to list for JSON

    # Save embeddings to JSON
    with open(embeddings_file, "w") as f:
        json.dump(embeddings_db, f, indent=4)

    # Convert Image to Base64 for MongoDB storage
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Save metadata to MongoDB
    face_collection.insert_one({
        "face_id": new_userid,
        "name": new_username,
        "image": img_base64,
        "id_type": id_type
    })

    print(f"Registered {new_username} successfully.")

    return jsonify({'success': True, 'userName': new_username})


@face_recognition_bp.route("/api/Authenticate", methods=["POST"])
def authenticate():
    data = request.json
    new_username = data.get("name", "")
    new_username = new_username.strip().upper().replace("  ", " ")

    new_userid = data.get("roll", "").strip()
    id_type = data.get("id_type", "")   

    key = f"{new_username}_{new_userid}"
    # Check if the name and roll combination exists in the database
    if key not in embeddings_db:
        return jsonify({"success": False, "message": "User not found"})

    # Capture face for face recognition using camera manager
    for attempt in range(3):
        ret, frame = camera_manager.capture_frame()
        if ret and frame is not None:
            break
        print(f"Authentication: Attempt {attempt+1} to capture frame failed")
        time.sleep(1)
    
    if not ret or frame is None:
        return jsonify({"success": False, "message": "Camera error - could not capture frame"})

    # Extract face embedding
    embedding = extract_face_embedding(frame)
    if embedding is None:
        return jsonify({"success": False, "message": "No face detected"})

    # Check FAISS index size
    print(f"FAISS index size: {faiss_index.ntotal}")
    
    if faiss_index.ntotal == 0:
        return jsonify({"success": False, "message": "No registered faces"})

    # Perform FAISS search
    D, I = faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
    
    if D[0][0] > 0.6:  # Adjusted threshold
        recognized_key = list(embeddings_db.keys())[I[0][0]]
        name, roll = recognized_key.split("_")

        save_attendance(name, roll)

        return jsonify({
                "success": True,
                "status": "Face recognized",
                "name": name,
                "roll": roll
            })

    return jsonify({"success": False, "message": "Face not recognized"})


@face_recognition_bp.route("/api/todayattendance", methods=["GET"])
def get_todays_attendance():
    try:
        attendance_file = f"{attendance_directory}/Attendance-{get_date_today()}.csv"
        if not os.path.exists(attendance_file):
            return jsonify({
                "success": False,
                "message": "No attendance records for today"
            })
        
        attendance_data = []
        with open(attendance_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            attendance_data = [row for row in reader]
        print(attendance_data)
        return jsonify({
            "success": True,
            "attendance": attendance_data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })


# Group Recognition Logic
recognized_faces = set()
# Global flag to control video feed
stop_video_flag = False

def generate_video_feed():
    global recognized_faces, stop_video_flag
    
    # Use lock to prevent concurrent camera access
    with video_lock:
        cap = camera_manager.get_camera()
    
    if cap is None:
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + b"Camera not available" + b"\r\n")
        return

    try:
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce processing load
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
        frame_skip = 2  # Process every 4th frame to reduce lag
        frame_count = 0
    
        while not stop_video_flag:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame from camera")
                    time.sleep(0.5)
                    continue
        
                frame_count += 1
                if frame_count % frame_skip == 0:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
                        for (x, y, w, h) in faces:
                            # Make sure face region is valid
                            if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                                face = frame[y:y+h, x:x+w]
                                
                                # Skip small faces
                                if w < 80 or h < 80:
                                    continue
                                
                                # Extract embedding using existing method
                                embedding = extract_face_embedding(face)
                                if embedding is None:
                                    continue  # Skip if face is not detected properly
        
                                # Perform FAISS search
                                D, I = faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
        
                                if D[0][0] > 0.5:  # Similarity threshold (adjustable)
                                    recognized_key = list(embeddings_db.keys())[I[0][0]]
                                    name, roll = recognized_key.split("_")
        
                                    recognized_faces.add((name, roll))
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
        
                # Compress frame for streaming
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            
            except Exception as e:
                print(f"Error in video feed generation: {e}")
                time.sleep(0.5)
    
    finally:
        # Make sure camera is released even if an exception occurs
        with video_lock:
            camera_manager.release_camera()


@face_recognition_bp.route("/api/video_feed")
def video_feed():
    global stop_video_flag
    stop_video_flag = False
    return Response(generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")


@face_recognition_bp.route("/api/stop_video", methods=["POST"])
def stop_video():
    global stop_video_flag
    stop_video_flag = True
    # Explicitly release the camera
    with video_lock:
        camera_manager.release_camera()
    return jsonify({"message": "Video feed stopped!"})


@face_recognition_bp.route('/api/start_video', methods=['POST'])
def start_video():
    global stop_video_flag
    stop_video_flag = False
    return jsonify({"message": "Video feed started successfully"})


@face_recognition_bp.route('/api/group_markattendance', methods=['POST'])
def group_markattendance():
    global recognized_faces
    if not recognized_faces:
        return jsonify({"success": False, "message": "No faces recognized."})

    count = len(recognized_faces)
    for name, roll in recognized_faces:
        save_attendance(name, roll)
    
    recognized_faces = set()  # Reset after marking attendance
    
    return jsonify({
        "success": True, 
        "message": f"Attendance marked for {count} detected faces."
    })