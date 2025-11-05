# PANGIL - Oral Lesion Detection System

Flask backend with YOLOv11 integration for accurate oral lesion detection.

## Setup Instructions

### 1. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Configure Model Path

Edit `app.py` and update the `MODEL_PATH` variable to point to your trained model:

\`\`\`python
MODEL_PATH = "path/to/your/runs/GI_disease_yolo11n/weights/best.pt"
\`\`\`

### 3. Run the Server

\`\`\`bash
python app.py
\`\`\`

The server will start on `http://0.0.0.0:5000`

### 4. Access the Application

- **Local access**: http://localhost:5000
- **Network access**: http://YOUR_PI_IP:5000

## API Endpoints

### POST /api/detect
Upload an image for oral lesion detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
\`\`\`json
{
  "detections": [
    {
      "type": "Aphthous Ulcer",
      "confidence": 85.3,
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 140,
        "height": 120
      }
    }
  ],
  "imageUrl": "data:image/jpeg;base64,...",
  "message": "Detected 1 oral lesion(s)"
}
\`\`\`

### GET /api/health
Check server and model status.

## Model Classes

The system detects the following oral lesions:
- Aphthous Ulcer
- Gingivitis
- Mucosal Tags
- Oral Candidiasis
- Xerostomia

## Deployment on Raspberry Pi

1. Copy all files to your Raspberry Pi
2. Install dependencies: `pip install -r requirements.txt`
3. Update MODEL_PATH in app.py
4. Run: `python app.py`
5. Access from any device on your network at `http://PI_IP:5000`

## Troubleshooting

**Model not loading:**
- Check that MODEL_PATH points to your trained model
- Ensure the model file exists and is accessible
- Check file permissions

**Import errors:**
- Make sure all dependencies are installed
- Try: `pip install --upgrade ultralytics`

**Camera not working:**
- This backend uses uploaded images, not live camera
- The frontend handles camera capture and sends images to this backend
