# PANGIL - Oral Imaging Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![React](https://img.shields.io/badge/React-19-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C)

**PANGIL** is an AI-powered oral disease detection system that uses deep learning to analyze images of the oral cavity and identify common oral health conditions. The system combines a modern web interface with powerful machine learning models to provide real-time detection and recommendations.

## Features

- **Real-time Camera Detection** - Capture and analyze oral images directly from your device camera
- **Image Upload** - Upload existing images for analysis
- **AI-Powered Detection** - Uses ResNet50 for classification and YOLO for precise bounding box detection
- **GradCAM Visualization** - Highlights the regions the AI focuses on for its predictions
- **Detection History** - View and manage past detection results
- **Disease Information** - Educational content about oral lesions and related conditions
- **User Authentication** - Simple sign-in/sign-up flow with local storage

## Detectable Conditions

PANGIL can detect 6 common oral health conditions:

| Condition | Description |
|-----------|-------------|
| **Aphthous Ulcer** | Canker sores - small, painful ulcers in the mouth |
| **Dental Caries** | Tooth decay and cavities |
| **Gingivitis** | Inflammation of the gums |
| **Oral Candidiasis** | Thrush - fungal infection causing white patches |
| **Mucosal Tags** | Small tissue growths in the oral cavity |
| **Xerostomia** | Dry mouth condition |

## Tech Stack

### Frontend
- **Framework**: Next.js 15 with App Router
- **UI Library**: React 19
- **Styling**: Tailwind CSS 4
- **Components**: Radix UI primitives
- **Animations**: Framer Motion
- **Forms**: React Hook Form + Zod validation

### Backend
- **API Framework**: FastAPI
- **ML Framework**: PyTorch
- **Object Detection**: YOLO (ultralytics)
- **Image Processing**: OpenCV, Pillow
- **Classification Model**: ResNet50

## Project Structure

```
PANGIL/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── backend/               # Python backend
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration
│   ├── requirements.txt   # Python dependencies
│   ├── lesion.pth         # ResNet50 model weights
│   └── best.pt            # YOLO model weights
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   ├── detection-interface.tsx
│   ├── camera-view.tsx
│   ├── image-upload.tsx
│   ├── results-panel.tsx
│   └── ...
├── lib/                   # Utility functions
├── hooks/                 # Custom React hooks
└── public/               # Static assets
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm/pnpm
- Python 3.10+
- CUDA-compatible GPU (optional, for faster inference)

### Frontend Setup

```bash
# Install dependencies
npm install
# or
pnpm install

# Run development server
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH="lesion.pth"
export YOLO_PATH="best.pt"
export FRONTEND_URL="http://localhost:3000"

# Run the backend
python main.py
```

The backend API will be available at `http://localhost:8000`

### Environment Variables

#### Frontend (.env.local)
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

#### Backend
```env
MODEL_PATH=lesion.pth
YOLO_PATH=best.pt
FRONTEND_URL=http://localhost:3000
ENVIRONMENT=development
STORAGE_DIR=./detections
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict-batch` | Batch image prediction |
| `POST` | `/detect` | Legacy detection endpoint |
| `GET` | `/history/{user_id}` | Get detection history |
| `DELETE` | `/detection/{user_id}/{index}` | Delete a detection record |

## Deployment

### Frontend (Vercel)
The frontend can be deployed to Vercel with automatic deployments on push to main branch.

### Backend Options
- **Raspberry Pi** - See `RASPBERRY_PI_SETUP_GUIDE.md`
- **Docker** - Use the provided `Dockerfile` in the backend directory
- **Railway** - Use `railway-deployment.yaml` configuration

For detailed deployment instructions, see `DEPLOYMENT_GUIDE.md`.

## How It Works

1. **Image Capture/Upload** - User provides an oral image via camera or file upload
2. **Preprocessing** - Image is resized and normalized for the models
3. **YOLO Detection** - Identifies regions of interest with bounding boxes
4. **ResNet50 Classification** - Classifies the detected condition
5. **GradCAM Visualization** - Generates attention heatmap showing model focus
6. **Results** - Returns prediction with confidence score, recommendations, and visualizations

## Screenshots

The application includes:
- Splash screen with branding
- User authentication (Sign In / Sign Up)
- Home selection screen with feature cards
- Camera view for real-time capture
- Image upload interface
- Results panel with detection overlay and GradCAM
- Detection history viewer
- Disease information pages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

**PANGIL is intended for educational and screening purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for proper evaluation and treatment of oral health conditions.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- UI components from [Radix UI](https://www.radix-ui.com/)
- Deep learning powered by [PyTorch](https://pytorch.org/)
- Object detection using [Ultralytics YOLO](https://ultralytics.com/)
