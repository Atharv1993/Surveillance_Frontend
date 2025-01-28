from flask import Flask,Response, render_template, request, redirect, url_for, jsonify
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
import shutil

import threading

app = Flask(_name_)

# Directory paths
faces_directory = "static/faces"
attendance_directory = "Attendance"
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

# Helper function to get today's date
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

# Helper function to resize and crop faces
def resize_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (160, 160))
    return None

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        new_username = request.form.get("newusername")
        new_userid = request.form.get("newuserid")
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
            while count < 5:  # Capture 5 images
                ret, frame = cap.read()
                if ret:
                    face = resize_and_crop_face(frame)
                    if face is not None:
                        cv2.imwrite(os.path.join(user_folder, f"{new_username}_{img_count}.jpg"), face)
                        count += 1
                        img_count += 1
                    cv2.imshow("Capturing Faces", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            # Train model on captured images
            DeepFace.find(img_path=os.path.join(user_folder, f"{new_username}_0.jpg"), db_path=faces_directory)
            return redirect(url_for("index"))
    return render_template("index.html")



# Route to start face recognition
@app.route("/startrecognition", methods=["GET"])
def start_recognition():
    cap = cv2.VideoCapture(0)
    recognized = False
    while not recognized:
        ret, frame = cap.read()
        if ret:
            face = resize_and_crop_face(frame)
            if face is not None:
                # Temporarily save the face image to a file
                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, face)
                
                try:
                    # Perform the face recognition
                    results = DeepFace.find(img_path=temp_face_path, db_path=faces_directory, enforce_detection=False)
                    
                    # Check if any match was found in the first DataFrame
                    if len(results) > 0 and not results[0].empty:
                        # Get the identity from the first match
                        identity = results[0].iloc[0]["identity"]
                        
                        # Extract the folder name from the full path
                        folder_name = os.path.basename(os.path.dirname(identity))
                        
                        # Split the folder name to get name and roll
                        name, roll = folder_name.split("_")
                        
                        # Save attendance with the name and roll number
                        save_attendance(name, roll)
                        recognized = True
                        break
                except Exception as e:
                    print(f"Error during face recognition: {e}")
            cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "Face recognized and attendance marked." if recognized else "No face recognized."})

# --------------------------------------------------------------------
# Route to serve the Group Recognition page
@app.route("/grouprecognition")
def group_recognition():
    global stop_video_flag
    stop_video_flag = False
    return render_template("group_recognition.html")


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


# Route to stream video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------------------------------------------------------------------------------------------------------------


# Global variable to store recognized faces in real-time
recognized_faces = set()

# Global flag to control video feed
stop_video_flag = False

# Route to stop the video feed
@app.route("/stop_video", methods=["POST"])
def stop_video():
    global stop_video_flag
    stop_video_flag = True
    return jsonify({"message": "Video feed stopped!"})




# Route to mark attendance
@app.route("/markattendance", methods=["POST"])
def mark_attendance():
    global recognized_faces
    for name, roll in recognized_faces:
        save_attendance(name, roll)
    return jsonify({"message": "Attendance marked for detected faces."})


# ----------------------------------------------------------------------------------------------------



if _name_ == "_main_":
    app.run(debug=True)