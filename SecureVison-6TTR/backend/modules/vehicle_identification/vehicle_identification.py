from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from PIL import Image
import numpy as np
import cv2
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
import base64
from bson.objectid import ObjectId

vehicle_plate_bp = Blueprint('vehicle_plate', __name__)

# Load YOLOv8 model
model = YOLO('modules/vehicle_identification/license_plate_detector.pt')

# Initialize PaddleOCR
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["Smart_Surveillance"]
registered_vehicles = db['vehicles']
vehicle_logs = db['vehicle_logs']  # New collection for vehicle logs

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
    
    # Detect license plate
    results = model.predict(source=image, imgsz=640, conf=0.5)
    
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

# Route: Vehicle Registration
@vehicle_plate_bp.route('/register_vehicle', methods=['POST'])
def register_vehicle():
    data = request.json
    
    plate_number = data.get('plate_number')
    owner = data.get('owner')
    vehicle_type = data.get('vehicle_type')
    color = data.get('color')
    model = data.get('model')
    full_image = data.get('full_image')
    plate_image = data.get('plate_image')
    
    if not plate_number or not owner or not vehicle_type:
        return jsonify({'success': False, 'message': 'Missing required data'})
    
    # Check if already registered
    existing = registered_vehicles.find_one({'plate_number': plate_number.upper()})
    if existing:
        return jsonify({'success': False, 'message': 'Vehicle already registered'})
    
    # Save to MongoDB
    registered_vehicles.insert_one({
        'plate_number': plate_number.upper(),
        'owner': owner,
        'vehicle_type': vehicle_type,
        'color': color,
        'model': model,
        'full_image': full_image,
        'plate_image': plate_image,
        'registered_at': datetime.utcnow()
    })
    
    return jsonify({
        'success': True,
        'message': 'Vehicle registered successfully',
        'plate_number': plate_number.upper()
    })


# Route: Vehicle Authentication
@vehicle_plate_bp.route('/authenticate_vehicle', methods=['POST'])
def authenticate_vehicle():
    data = request.json
    plate_number = data.get('plate_number')
    
    if not plate_number:
        return jsonify({'success': False, 'message': 'No plate number provided'})
    
    # Find vehicle in database
    vehicle = registered_vehicles.find_one({'plate_number': plate_number.upper()})
    
    # Create authentication log
    log_data = {
        'plate_number': plate_number.upper(),
        'timestamp': datetime.utcnow(),
        'access_granted': vehicle is not None
    }
    
    # Add additional vehicle info if available
    if vehicle:
        log_data.update({
            'owner': vehicle.get('owner', 'Unknown'),
            'vehicle_type': vehicle.get('vehicle_type', 'Unknown'),
            'model': vehicle.get('model', '')
        })
    
    # Save log to MongoDB
    vehicle_logs.insert_one(log_data)
    
    if vehicle:
        # Convert MongoDB ObjectId to string for JSON serialization
        vehicle['_id'] = str(vehicle['_id'])
        return jsonify({
            'success': True, 
            'message': 'Access Granted', 
            'vehicle': {
                'plate_number': vehicle['plate_number'],
                'owner': vehicle['owner'],
                'vehicle_type': vehicle['vehicle_type'],
                'color': vehicle.get('color', ''),
                'model': vehicle.get('model', ''),
                'full_image': vehicle.get('full_image', ''),
                'plate_image': vehicle.get('plate_image', ''),
                'registered_at': vehicle['registered_at'].isoformat()
            }
        })
    else:
        return jsonify({
            'success': False, 
            'message': 'Access Denied', 
            'plate_number': plate_number.upper()
        })

# ----------------------------------------------------------------------------------------------------
# vehicle dashboard endpoints

# Route to get all vehicle records
@vehicle_plate_bp.route("/records", methods=["GET"])
def get_vehicle_records():
    try:
        # Get all vehicle records from MongoDB
        vehicles = list(registered_vehicles.find())
        
        # Convert ObjectId to string and ensure field format consistency
        for vehicle in vehicles:
            vehicle["_id"] = str(vehicle["_id"])
            
            # Ensure registered_at is ISO format
            if "registered_at" in vehicle:
                vehicle["registered_at"] = vehicle["registered_at"].isoformat()
        
        return jsonify(vehicles), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get vehicle detection stats
@vehicle_plate_bp.route("/stats", methods=["GET"])
def get_vehicle_stats():
    try:
        # Get total vehicles
        total_vehicles = registered_vehicles.count_documents({})
        
        # Get registered vehicles (those with owner)
        registered_count = registered_vehicles.count_documents({
            "owner": {"$exists": True, "$ne": None}
        })
        
        # Get detected today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        detected_today = vehicle_logs.count_documents({
            "timestamp": {"$gte": today}
        })
        
        # Get access granted today
        access_granted_today = vehicle_logs.count_documents({
            "timestamp": {"$gte": today},
            "access_granted": True
        })
        
        stats = {
            "totalVehicles": total_vehicles,
            "registeredVehicles": registered_count,
            "detectedToday": detected_today,
            "accessGrantedToday": access_granted_today
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to delete a vehicle record
@vehicle_plate_bp.route("/delete/<id>", methods=["DELETE"])
def delete_vehicle_record(id):
    try:
        # Delete the vehicle record
        result = registered_vehicles.delete_one({"_id": ObjectId(id)})
        
        if result.deleted_count == 0:
            return jsonify({"error": "Vehicle record not found"}), 404
            
        return jsonify({"message": "Vehicle record deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to update a vehicle record
@vehicle_plate_bp.route("/update/<id>", methods=["PUT"])
def update_vehicle_record(id):
    try:
        data = request.json
        
        # Check if record exists
        if not registered_vehicles.find_one({"_id": ObjectId(id)}):
            return jsonify({"error": "Vehicle record not found"}), 404
        
        # Update fields
        update_data = {}
        if "owner_name" in data:
            update_data["owner"] = data["owner_name"]
        if "vehicle_model" in data:
            update_data["model"] = data["vehicle_model"]
        
        # Update in database
        registered_vehicles.update_one(
            {"_id": ObjectId(id)},
            {"$set": update_data}
        )
        
        return jsonify({"message": "Vehicle record updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get vehicle logs
@vehicle_plate_bp.route("/logs", methods=["GET"])
def get_vehicle_logs():
    try:
        # Get filter date from query param
        filter_date = request.args.get('date')
        
        # Prepare query
        query = {}
        
        if filter_date:
            try:
                # Parse the date string to datetime
                filter_date = datetime.strptime(filter_date, '%Y-%m-%d')
                next_day = filter_date + timedelta(days=1)
                
                # Filter logs for the specified date
                query = {
                    "timestamp": {
                        "$gte": filter_date,
                        "$lt": next_day
                    }
                }
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Get logs from database
        logs = list(vehicle_logs.find(query).sort("timestamp", -1))
        
        # Format logs for JSON response
        formatted_logs = []
        for log in logs:
            formatted_log = {
                "_id": str(log["_id"]),
                "plate_number": log["plate_number"],
                "timestamp": log["timestamp"].isoformat(),
                "access_granted": log["access_granted"],
                "owner": log.get("owner", "Unknown"),
                "vehicle_type": log.get("vehicle_type", "Unknown"),
                "model": log.get("model", "")
            }
            formatted_logs.append(formatted_log)
        
        return jsonify(formatted_logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500