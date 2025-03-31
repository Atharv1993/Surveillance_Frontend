import os
import cv2
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from flask import Flask,Response, request, jsonify
from flask_cors import CORS
import csv
from deepface import DeepFace

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Directory paths
faces_directory = "static/faces"
attendance_directory = "Attendance"
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

def get_date_today():
    return datetime.now().strftime("%Y-%m-%d")

# Helper function to save attendance
def save_attendance(name, roll):
    filename = f"{attendance_directory}/Attendance-{get_date_today()}.csv"
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=["Name", "Roll", "Time"])
    else:
        df = pd.read_csv(filename)
    if roll not in df["Roll"].values:
        now = datetime.now().strftime("%H:%M:%S")
        df = pd.concat([df, pd.DataFrame([[name, roll, now]], columns=["Name", "Roll", "Time"])])
        df.to_csv(filename, index=False)

def resize_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (160, 160))
    return None

@app.route('/api/ocr', methods=['GET'])
def get_data():
    return jsonify({'Name': 'Atharva', 'id': 51 })

@app.route('/api/register-face', methods=['POST'])
def register_face():
    ocr_data = requests.get('http://localhost:5000/api/ocr').json()
    new_username = ocr_data['Name']
    new_userid = str(ocr_data['id'])
    
    if new_username and new_userid:
        img_count =0
        user_folder = os.path.join(faces_directory, f"{new_username}_{new_userid}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        else:
                # Count the number of .jpg images in the folder
                img_count = sum(
                    1 for file in os.listdir(user_folder)
                    if os.path.isfile(os.path.join(user_folder, file)) and 
                    file.lower().endswith('.jpg')
                )
        
        # Start capturing images
        cap = cv2.VideoCapture(0)
        count = 0
        captured_images = []
        
        while count < 5:  # Capture 5 images
            ret, frame = cap.read()
            if ret:
                face = resize_and_crop_face(frame)
                if face is not None:
                    image_path = os.path.join(user_folder, f"{new_username}_{img_count}.jpg")
                    cv2.imwrite(image_path, face)
                    captured_images.append(image_path)
                    count += 1
                    img_count+=1
                cv2.imshow("Capturing Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Train model on captured images
        if captured_images:
            DeepFace.find(img_path=captured_images[0], db_path=faces_directory)
        
        return jsonify({
            'success': True,
            'userName': new_username,
            'userImages': len(captured_images)
        })
    
    return jsonify({'success': False, 'message': 'Registration failed'})

@app.route("/api/Authenticate", methods=["POST"])
def authenticate():
    try:
        cap = cv2.VideoCapture(0)
        recognized = False
        
        while not recognized:
            ret, frame = cap.read()
            if ret:
                face = resize_and_crop_face(frame)
                if face is not None:
                    temp_face_path = "temp_face.jpg"
                    cv2.imwrite(temp_face_path, face)
                    
                    try:
                        results = DeepFace.find(
                            img_path=temp_face_path, 
                            db_path=faces_directory, 
                            enforce_detection=False
                        )
                        
                        if len(results) > 0 and not results[0].empty:
                            identity = results[0].iloc[0]["identity"]
                            folder_name = os.path.basename(os.path.dirname(identity))
                            name, roll = folder_name.split("_")
                            
                            save_attendance(name, roll)
                            recognized = True
                            
                            return jsonify({
                                "success": True,
                                "status": "Face recognized", 
                                "name": name, 
                                "roll": roll
                            })
                    except Exception as e:
                        print(f"Recognition error: {e}")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return jsonify({
            "success": False,
            "status": "No face recognized"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "status": f"Authentication failed: {str(e)}"
        })

@app.route("/api/todayattendance", methods=["GET"])
def get_todays_attendance():
    try:
        # attendance_dir = "Attendance"
        attendance_file = f"{attendance_directory}/Attendance-{get_date_today()}.csv"
        if not os.path.exists(attendance_file):
            return jsonify({
                "success": False,
                "message": "No attendance records for today"
            })
        
        attendance_data = []
        with open(attendance_file, mode='r') as file:
            reader = csv.reader(file)
            # for row in reader:
            #     print(row['name'],row['roll'],row["time"])
            #     attendance_data.append({
            #         "name": row['name'],
            #         "roll": row['roll'],
            #         "time": row['time']
            #     })
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
            "message": "error"
        })


# Group Auth Logic

recognized_faces = set()
# Global flag to control video feed
stop_video_flag = False

def generate_video_feed():
    global recognized_faces
    global stop_video_flag
    cap = cv2.VideoCapture(0)
    
    # Set a higher FPS if the camera supports it (check your camera specs)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # You can resize the frame to reduce processing load
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    # Set a limit on how often to process frames (e.g., every 2nd frame)
    frame_skip = 1  # Process every 2nd frame
    frame_count = 0
    
    while True:
        if stop_video_flag:
            break  # Stop the feed when the flag is set

        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip == 0:  # Only process certain frames to reduce lag
            # Perform face detection and recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                temp_path = "temp_group.jpg"
                cv2.imwrite(temp_path, face)
                try:
                    results = DeepFace.find(img_path=temp_path, db_path=faces_directory, enforce_detection=False)
                    if len(results) > 0 and not results[0].empty:
                        identity = results[0].iloc[0]["identity"]
                        name, roll = os.path.basename(os.path.dirname(identity)).split("_")
                        recognized_faces.add((name, roll))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except Exception as e:
                    print(f"Error: {e}")

        # Encode frame as JPEG and yield it
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.route("/api/video_feed")
def video_feed():
    return Response(generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stop_video", methods=["POST"])
def stop_video():
    global stop_video_flag
    stop_video_flag = True
    return jsonify({"message": "Video feed stopped!"})

@app.route('/api/start_video', methods=['POST'])
def start_video():
    global stop_video_flag
    stop_video_flag = False
    return jsonify({"message": "Video feed started successfully"})

@app.route('/api/group_markattendance', methods=['POST'])
def group_markattendance():
    global recognized_faces
    for name, roll in recognized_faces:
        save_attendance(name, roll)
    return jsonify({"message": "Attendance marked for detected faces."})

if __name__ == '__main__':
    app.run(debug=True)