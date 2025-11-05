"""
PANGIL - Oral Imaging Detection System
Flask Application for Raspberry Pi

Installation:
1. pip install flask flask-cors opencv-python pillow
2. Install your YOLO model dependencies (ultralytics, ncnn, etc.)
3. Update MODEL_PATH with your model location
4. Run: python flask-app.py

The app will be available at http://your-pi-ip:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
MODEL_PATH = "path/to/your/model"  # Update this with your model path
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease to Lesion Mapping
LESION_TO_DISEASES = {
    "Aphthous Ulcer": ["Crohn's Disease", "Ulcerative Colitis", "Celiac Disease"],
    "Xerostomia": ["Chronic Liver Disease", "Celiac Disease"],
    "Dental Caries": ["Celiac Disease"],
    "Mucosal Tags": ["Crohn's Disease", "Celiac Disease"],
    "Gingivitis": ["Crohn's Disease", "Chronic Liver Disease", "Ulcerative Colitis"],
    "Oral Candidiasis": ["Chronic Liver Disease", "Celiac Disease"]
}

ALL_DISEASES = ["Crohn's Disease", "Ulcerative Colitis", "Celiac Disease", "Chronic Liver Disease"]

def load_model():
    """Load your YOLO model here"""
    # TODO: Replace with your actual model loading code
    # Example:
    # from ultralytics import YOLO
    # model = YOLO(MODEL_PATH)
    # return model
    return None

# Load model at startup
model = load_model()

def calculate_disease_probabilities(detections):
    """
    Calculate disease probabilities from detected lesions
    Based on the algorithm from your prototype
    """
    disease_scores = {disease: 0.0 for disease in ALL_DISEASES}
    
    for detection in detections:
        lesion_type = detection['type']
        confidence = detection['confidence']
        
        if lesion_type in LESION_TO_DISEASES:
            diseases = LESION_TO_DISEASES[lesion_type]
            contribution = confidence / len(diseases)
            
            for disease in diseases:
                disease_scores[disease] += contribution
    
    # Normalize to 100%
    total = sum(disease_scores.values())
    if total > 0:
        disease_probabilities = {
            disease: round((score / total) * 100, 2)
            for disease, score in disease_scores.items()
        }
    else:
        disease_probabilities = {disease: 0.0 for disease in ALL_DISEASES}
    
    return disease_probabilities

def run_inference(image):
    """
    Run YOLO inference on the image
    Returns: list of detections with bounding boxes
    """
    # TODO: Replace with your actual inference code
    # This is a placeholder that returns demo data
    
    # Example of what your inference should return:
    detections = [
        {
            "type": "Aphthous Ulcer",
            "confidence": 85,
            "bbox": {"x": 120, "y": 80, "width": 60, "height": 50}
        },
        {
            "type": "Gingivitis",
            "confidence": 76,
            "bbox": {"x": 200, "y": 150, "width": 70, "height": 60}
        }
    ]
    
    # Your actual inference code would look something like:
    # results = model(image)
    # detections = []
    # for result in results:
    #     for box in result.boxes:
    #         detections.append({
    #             "type": result.names[int(box.cls)],
    #             "confidence": float(box.conf) * 100,
    #             "bbox": {
    #                 "x": int(box.xyxy[0][0]),
    #                 "y": int(box.xyxy[0][1]),
    #                 "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
    #                 "height": int(box.xyxy[0][3] - box.xyxy[0][1])
    #             }
    #         })
    
    return detections

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image or video frame"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        detections = run_inference(image)
        
        # Calculate disease probabilities
        disease_probabilities = calculate_disease_probabilities(detections)
        
        # Save image for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, image)
        
        return jsonify({
            'detections': detections,
            'disease_probabilities': disease_probabilities,
            'image_path': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print("Starting PANGIL - Oral Imaging Detection System")
    print("Access the application at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
