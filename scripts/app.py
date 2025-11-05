"""
PANGIL Flask Backend - Oral Lesion Detection System
Integrates with trained YOLOv11 model for accurate detection
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Update this path to point to your trained model
MODEL_PATH = "runs/GI_disease_yolo11n/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Please update MODEL_PATH to point to your trained model")
    model = None

# Class name mapping from model output to display names
CLASS_NAME_MAPPING = {
    "Aphthous_Ulcers": "Aphthous Ulcer",
    "Gingivitis": "Gingivitis",
    "Mucosal_Tags": "Mucosal Tags",
    "Oral_Candidiasis": "Oral Candidiasis",
    "Xerostomia": "Xerostomia",
    # Enamel_Hypoplasia is excluded as per user request
}

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return send_from_directory('static', path)

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect oral lesions in uploaded image using YOLO model
    Returns: JSON with detections array containing type, confidence, and bbox
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check MODEL_PATH configuration.'
        }), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Run YOLO inference
        results = model(img, conf=0.25)  # confidence threshold
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Map class name to display name
                display_name = CLASS_NAME_MAPPING.get(class_name, class_name)
                
                # Skip Enamel_Hypoplasia if detected
                if class_name == "Enamel_Hypoplasia":
                    continue
                
                # Convert to percentage and format bbox
                detection = {
                    'type': display_name,
                    'confidence': round(confidence * 100, 1),
                    'bbox': {
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                }
                
                detections.append(detection)
        
        # Convert image to base64 for returning to frontend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'detections': detections,
            'imageUrl': img_data_url,
            'message': f'Detected {len(detections)} oral lesion(s)'
        })
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    print("=" * 60)
    print("PANGIL - Oral Lesion Detection System")
    print("=" * 60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print("=" * 60)
    print("Starting Flask server on http://0.0.0.0:5000")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
