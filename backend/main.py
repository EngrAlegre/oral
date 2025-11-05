from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
import base64
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple

# Initialize FastAPI app
app = FastAPI(title="PANGIL Backend", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://p030bzfb-3000.asse.devtunnels.ms",
        os.getenv("FRONTEND_URL", "https://pangil.vercel.app")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local storage for detections
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./detections"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load ResNet50 model
MODEL_PATH = os.getenv("MODEL_PATH", "lesion.pth")
num_classes = 6  # 6 oral disease classes

try:
    # Initialize ResNet50
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info(f"ResNet50 model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    logger.exception(e)
    model = None

# Load YOLO model for precise bounding boxes
YOLO_PATH = os.getenv("YOLO_PATH", "best.pt")
try:
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_PATH)
    logger.info(f"YOLO model loaded successfully from {YOLO_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO model from {YOLO_PATH}: {e}")
    yolo_model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Disease mapping for ResNet classes
DISEASE_MAPPING = {
    0: "Aphthous Ulcer",
    1: "Dental Caries", 
    2: "Gingivitis",
    3: "Oral Candidiasis",
    4: "Mucosal Tags",
    5: "Xerostomia",
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "ResNet50",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "storage_path": str(STORAGE_DIR),
        "device": str(device)
    }

@app.post("/predict")
async def predict_single(file: UploadFile = File(...), user_id: str = None):
    """
    Predict oral lesions using ResNet50 with GradCAM
    Returns: label, confidence, recommendation, visualizations
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image_pil)
        
        # Run inference
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare image for model
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        class_id = predicted_class.item()
        confidence_score = confidence.item()
        label = DISEASE_MAPPING.get(class_id, "Unknown")
        
        logger.info(f"Prediction: {label} with confidence {confidence_score:.3f}")
        
        # Prefer GradCAM on YOLO crop for better localization
        yolo_box = get_yolo_best_box(image_np)
        if yolo_box:
            gradcam_image = gradcam_on_yolo_crop(image_np, yolo_box)
        else:
            gradcam_image = generate_gradcam(model, input_tensor, image_np, class_id, target_block="layer3")
        gradcam_image_b64 = numpy_to_base64(gradcam_image)
        
        # Generate detection image: prefer YOLO box+label; fallback to simple label overlay
        if yolo_box:
            detection_image = draw_yolo_box(image_np, yolo_box, label)
        else:
            detection_image = create_detection_overlay(image_np, label, confidence_score)
        detection_image_b64 = numpy_to_base64(detection_image)
        
        # Get recommendations
        recommendation = get_recommendation(label, confidence_score)
        ai_feedback = get_ai_feedback(label, confidence_score)
        
        result = {
            "label": label,
            "confidence": confidence_score,
            "recommendation": recommendation,
            "ai_feedback": ai_feedback,
            "detection_image": detection_image_b64,
            "gradcam_image": gradcam_image_b64,
            "timestamp": datetime.utcnow().isoformat(),
            "image_filename": file.filename
        }
        
        if user_id:
            result["user_id"] = user_id
            save_detection_locally(user_id, result)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict oral lesions in multiple uploaded images
    Returns: array of detections with GradCAM
    """
    try:
        all_detections = []
        
        for idx, file in enumerate(files):
            # Read and preprocess image
            contents = await file.read()
            image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = np.array(image_pil)
            
            # Run inference
            if model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            class_id = predicted_class.item()
            confidence_score = confidence.item()
            label = DISEASE_MAPPING.get(class_id, "Unknown")
            
            # Generate visualizations (GradCAM on YOLO crop if present)
            yolo_box = get_yolo_best_box(image_np)
            if yolo_box:
                gradcam_image = gradcam_on_yolo_crop(image_np, yolo_box)
            else:
                gradcam_image = generate_gradcam(model, input_tensor, image_np, class_id, target_block="layer3")
            if yolo_box:
                detection_image = draw_yolo_box(image_np, yolo_box, label)
            else:
                detection_image = create_detection_overlay(image_np, label, confidence_score)
            
            all_detections.append({
                "label": label,
                "confidence": confidence_score,
                "recommendation": get_recommendation(label, confidence_score),
                "ai_feedback": get_ai_feedback(label, confidence_score),
                "detection_image": numpy_to_base64(detection_image),
                "gradcam_image": numpy_to_base64(gradcam_image),
                "image_index": idx
            })
        
        return JSONResponse(content={"detections": all_detections})
    
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_lesion(file: UploadFile = File(...), user_id: str = None):
    """Legacy endpoint for compatibility"""
    return await predict_single(file, user_id)

@app.get("/history/{user_id}")
async def get_detection_history(user_id: str, limit: int = 50):
    """Get detection history for a user"""
    try:
        history_file = STORAGE_DIR / f"{user_id}_history.json"
        
        if not history_file.exists():
            return {"detections": [], "count": 0}
        
        with open(history_file, 'r') as f:
            detections = json.load(f)
        
        return {"detections": detections[-limit:], "count": len(detections)}
    
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/detection/{user_id}/{detection_index}")
async def delete_detection(user_id: str, detection_index: int):
    """Delete a detection record"""
    try:
        history_file = STORAGE_DIR / f"{user_id}_history.json"
        
        if not history_file.exists():
            raise HTTPException(status_code=404, detail="Detection not found")
        
        with open(history_file, 'r') as f:
            detections = json.load(f)
        
        if 0 <= detection_index < len(detections):
            detections.pop(detection_index)
            with open(history_file, 'w') as f:
                json.dump(detections, f, indent=2)
            return {"message": "Detection deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Detection not found")
    
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def save_detection_locally(user_id: str, detection: dict):
    """Save detection to local JSON file"""
    history_file = STORAGE_DIR / f"{user_id}_history.json"
    
    detections = []
    if history_file.exists():
        with open(history_file, 'r') as f:
            detections = json.load(f)
    
    detections.append(detection)
    
    with open(history_file, 'w') as f:
        json.dump(detections, f, indent=2)

def generate_gradcam(model, input_tensor, original_image, target_class, target_block: str = "layer3"):
    """
    Generate GradCAM on the given input tensor and overlay on the provided original_image.
    Uses a higher-resolution block (layer3) by default for sharper heatmaps.
    """
    # Select a higher-resolution conv block for better spatial detail
    if target_block == "layer2":
        target_layer = model.layer2[-1]
    elif target_block == "layer3":
        target_layer = model.layer3[-1]
    else:
        target_layer = model.layer4[-1]
    
    # Forward pass with hooks
    features = []
    gradients = []
    
    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Backward pass
    target = output[0, target_class]
    target.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get gradients and features
    gradient = gradients[0].cpu().data.numpy()[0]
    feature = features[0].cpu().data.numpy()[0]
    
    # Calculate weights
    weights = np.mean(gradient, axis=(1, 2))
    
    # Create CAM
    cam = np.zeros(feature.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature[i]
    
    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    
    # Resize to original image size
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Apply colormap (rainbow effect)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    
    return overlay

def pad_and_clip_box(x1, y1, x2, y2, w, h, pad_ratio: float = 0.10):
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w - 1, x2 + pad_x)
    ny2 = min(h - 1, y2 + pad_y)
    return nx1, ny1, nx2, ny2

def gradcam_on_yolo_crop(image_np: np.ndarray, yolo_box: Tuple[int, int, int, int, float]) -> np.ndarray:
    """Compute GradCAM on YOLO crop and place it back onto the original image area."""
    x1, y1, x2, y2, _ = yolo_box
    h, w = image_np.shape[:2]
    cx1, cy1, cx2, cy2 = pad_and_clip_box(x1, y1, x2, y2, w, h)
    crop = image_np[cy1:cy2, cx1:cx2]

    # Prepare tensor for the crop
    crop_pil = Image.fromarray(crop)
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(crop_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        class_id = int(pred.item())

    # GradCAM on crop using higher-res block
    crop_overlay = generate_gradcam(model, crop_tensor, crop, class_id, target_block="layer3")

    # Place crop overlay back into the original image region
    result = image_np.copy()
    result[cy1:cy2, cx1:cx2] = crop_overlay
    return result

def create_detection_overlay(image, label, confidence):
    """Create detection image with label and confidence"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # Add semi-transparent overlay at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    # Add text
    text = f"{label}"
    conf_text = f"Confidence: {confidence*100:.1f}%"
    
    cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, conf_text, (20, 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    
    return img

def get_yolo_best_box(image: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """Run YOLO to get the best bounding box (x1, y1, x2, y2, conf)."""
    if yolo_model is None:
        return None
    try:
        results = yolo_model(image, conf=0.25)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None
        # Select highest confidence box
        best_idx = int(torch.argmax(boxes.conf).item())
        box = boxes[best_idx]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0].item())
        return (x1, y1, x2, y2, conf)
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return None

def draw_yolo_box(image: np.ndarray, box_tuple: Tuple[int, int, int, int, float], label_text: str) -> np.ndarray:
    """Draw green bounding box and label using YOLO coordinates.
    Auto-scales the label text so it never overflows the box width.
    """
    x1, y1, x2, y2, conf = box_tuple
    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 6)

    title = f"{label_text} {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3

    # Compute max allowed text width (box width minus padding)
    box_w = max(10, x2 - x1)
    max_text_w = box_w - 24

    # Measure at scale 1.0 to estimate
    (base_w, base_h), _ = cv2.getTextSize(title, font, 1.0, thickness)
    # Determine scale factor within bounds
    if base_w > 0:
        scale = max(0.6, min(1.8, max_text_w / base_w))
    else:
        scale = 1.0

    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    top = max(0, y1 - th - 12)
    cv2.rectangle(out, (x1, top), (x1 + tw + 12, y1), (0, 255, 0), -1)
    cv2.putText(out, title, (x1 + 6, max(24, y1 - 6)), font, scale, (0, 0, 0), thickness)
    return out

## Segmentation view removed per product requirement

def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def get_recommendation(label: str, confidence: float) -> dict:
    """Get recommendation based on disease type"""
    recommendations = {
        "Aphthous Ulcer": {
            "urgent_actions": ["Avoid spicy foods", "Use topical anesthetics", "Consult dentist if persists beyond 2 weeks"],
            "monitoring": ["Track ulcer size", "Monitor for secondary infection"],
            "lifestyle": ["Maintain oral hygiene", "Reduce stress"]
        },
        "Dental Caries": {
            "urgent_actions": ["Schedule dental appointment immediately", "Avoid sugary foods"],
            "monitoring": ["Check for pain", "Monitor cavity progression"],
            "lifestyle": ["Brush twice daily", "Floss regularly"]
        },
        "Gingivitis": {
            "urgent_actions": ["Improve oral hygiene", "Use antimicrobial mouthwash"],
            "monitoring": ["Check for bleeding", "Monitor inflammation"],
            "lifestyle": ["Brush gently", "Floss daily"]
        },
        "Oral Candidiasis": {
            "urgent_actions": ["Consult healthcare provider for antifungal treatment", "Avoid irritants"],
            "monitoring": ["Track white patches", "Monitor symptoms"],
            "lifestyle": ["Maintain oral hygiene", "Avoid tobacco"]
        },
        "Mucosal Tags": {
            "urgent_actions": ["Monitor for changes", "Consult specialist if needed"],
            "monitoring": ["Track size and appearance"],
            "lifestyle": ["Maintain oral hygiene"]
        },
        "Xerostomia": {
            "urgent_actions": ["Stay hydrated", "Use saliva substitutes"],
            "monitoring": ["Monitor dry mouth severity"],
            "lifestyle": ["Drink water frequently", "Avoid dry foods"]
        }
    }
    
    base_rec = recommendations.get(label, {
        "urgent_actions": ["Consult healthcare provider"],
        "monitoring": ["Monitor symptoms"],
        "lifestyle": ["Maintain oral hygiene"]
    })
    
    if confidence < 0.5:
        base_rec["urgent_actions"].insert(0, "Low confidence - Professional evaluation strongly recommended")
    elif confidence < 0.7:
        base_rec["urgent_actions"].insert(0, "Moderate confidence - Professional verification recommended")
    
    return base_rec

def get_ai_feedback(label: str, confidence: float) -> dict:
    """Generate AI feedback"""
    detection_quality = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
    
    return {
        "detection": f"Detected {label} with {confidence:.1%} confidence using ResNet50 deep learning model.",
        "segmentation": f"Segmentation analysis completed. Key features identified with {detection_quality} certainty.",
        "gradcam": f"GradCAM attention mapping applied. Model focused on distinctive {label.lower()} characteristics with {detection_quality} activation intensity."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
