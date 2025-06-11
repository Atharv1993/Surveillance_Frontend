import os
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from datetime import datetime
from werkzeug.utils import secure_filename

human_detection_bp = Blueprint("human_detection", __name__)

DETECTION_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'detection_images')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@human_detection_bp.route('/api/upload_detection', methods=['POST'])
def upload_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = secure_filename(f"detection_{ts}.jpg")
        os.makedirs(DETECTION_FOLDER, exist_ok=True)
        file.save(os.path.join(DETECTION_FOLDER, filename))
        return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 201
    return jsonify({'error': 'Invalid file type'}), 400

@human_detection_bp.route('/api/detection_images', methods=['GET'])
def list_detection_images():
    os.makedirs(DETECTION_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(DETECTION_FOLDER) if allowed_file(f)]
    files.sort(reverse=True)  # newest first
    images = []
    for fname in files:
        # Extract timestamp from filename
        try:
            ts_str = fname.replace('detection_', '').replace('.jpg', '')
            ts = datetime.strptime(ts_str, '%Y%m%d_%H%M%S_%f')
            timestamp = ts.isoformat()
        except Exception:
            timestamp = None
        images.append({
            'filename': fname,
            'timestamp': timestamp,
            'url': f"/detection_images/{fname}"
        })
    return jsonify(images)

@human_detection_bp.route('/detection_images/<filename>')
def serve_detection_image(filename):
    return send_from_directory(DETECTION_FOLDER, filename)
