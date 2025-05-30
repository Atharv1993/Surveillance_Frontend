from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from PIL import Image
import numpy as np
import cv2
import re
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import base64
from bson.objectid import ObjectId

vehicle_plate_bp = Blueprint('vehicle_plate', __name__)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Vehicle Detection module using device: {device}")

# Load YOLOv8 model with GPU support
model = YOLO('modules/vehicle_identification/license_plate_detector.pt')
# Explicitly set device to CUDA if available
if device.type == 'cuda':
    print("Using CUDA for YOLO license plate detection")
    model.to(device)

# Initialize PaddleOCR with GPU support if available
use_gpu = device.type == 'cuda'
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
if use_gpu:
    print("Using GPU for PaddleOCR")
else:
    print("Using CPU for PaddleOCR")

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["Smart_Surveillance"]
registered_vehicles = db['vehicles']

# Helper: Clean and standardize text
def clean_text(text):
    text = text.upper()
    text = re.sub(r'\bIND\b', '', text)  # Remove 'IND' text that might appear
    text = re.sub(r'[^A-Z0-9 ]', '', text)  # Keep only alphanumeric
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip().replace(" ", "")  # Remove all spaces

# Helper: Crop license plate
def crop_plate(image, results):
    if len(results[0].boxes) == 0:
        return None
    
    # Use first detection
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image[y1:y2, x1:x2]

# Helper: Extract plate number using PaddleOCR
def extract_text_from_image(cropped_image):
    # Preprocess for OCR
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with Paddle
    paddle_result = paddle_ocr.ocr(thresh, cls=True)
    
    # Extract text
    paddle_texts = [line[1][0] for block in paddle_result for line in block]
    paddle_text = ' '.join(paddle_texts)
    
    # Clean the text
    cleaned_text = clean_text(paddle_text)
    
    return cleaned_text


# Helper: Encode image to base64 for sending to frontend
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Route: Process vehicle image
@vehicle_plate_bp.route('/process_vehicle_image', methods=['POST'])
def process_vehicle_image():
    file = request.files.get('image')
    
    if not file:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    # Read image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Detect license plate with GPU if available
    results = model.predict(source=image, imgsz=640, conf=0.5, device=0 if device.type == 'cuda' else 'cpu')
    
    # Crop license plate
    cropped = crop_plate(image, results)
    if cropped is None:
        return jsonify({
            'success': False, 
            'message': 'License plate not detected',
            'full_image': encode_image(image)
        })
    
    # Extract plate number
    plate_number = extract_text_from_image(cropped)
    
    # Return results
    return jsonify({
        'success': True,
        'message': 'License plate detected',
        'plate_number': plate_number,
        'full_image': encode_image(image),
        'plate_image': encode_image(cropped)
    })
