# PANGIL Deployment Guide

## Architecture Overview

PANGIL is a full-stack application with:
- **Frontend**: Next.js + React on Vercel
- **Backend**: FastAPI + TensorFlow on Raspberry Pi with ngrok tunnel
- **Storage**: Local JSON file storage on Raspberry Pi
- **Model**: YOLO model (.pt format)

## Prerequisites

1. **Vercel Account** - For frontend hosting
2. **Raspberry Pi 5** - For backend hosting with ngrok tunnel
3. **GitHub Repository** - For version control
4. **ngrok Account** - For exposing Raspberry Pi to internet

## Step 1: Prepare Your Backend

### 1.1 Set Up Backend on Raspberry Pi

\`\`\`bash
# Create backend directory
mkdir -p /home/pi/PANGIL
cd /home/pi/PANGIL

# Copy backend files
cp backend/main.py .
cp backend/requirements.txt .

# Create models directory
mkdir -p models
cp path/to/your/model.pt models/lesion.pt
\`\`\`

### 1.2 Test Backend Locally

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH="/home/pi/PANGIL/models/lesion.pt"
export FRONTEND_URL="https://pangil.vercel.app"
export ENVIRONMENT="production"

# Run backend
python main.py

# Test health endpoint
curl http://localhost:8000/health
\`\`\`

## Step 2: Expose Raspberry Pi with ngrok

### 2.1 Install ngrok

\`\`\`bash
# Download ngrok for ARM64
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.zip
unzip ngrok-v3-stable-linux-arm64.zip
sudo mv ngrok /usr/local/bin/
\`\`\`

### 2.2 Authenticate ngrok

\`\`\`bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
\`\`\`

### 2.3 Start ngrok Tunnel

\`\`\`bash
ngrok http 8000
\`\`\`

**Copy the HTTPS URL provided by ngrok**

## Step 3: Deploy Frontend to Vercel

### 3.1 Connect to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "Add New..." → "Project"
3. Import your GitHub repository

### 3.2 Set Environment Variables

In Vercel dashboard, add:

\`\`\`
NEXT_PUBLIC_BACKEND_URL=https://your-ngrok-url.ngrok.io
MODEL_PATH=/home/kerljann/yolo/lesion.pt
FRONTEND_URL=https://pangil.vercel.app
ENVIRONMENT=production
\`\`\`

### 3.3 Deploy

Vercel will automatically deploy on push to main branch.

## Step 4: Test Deployment

### 4.1 Test Backend

\`\`\`bash
curl https://your-ngrok-url.ngrok.io/health
\`\`\`

Expected response:
\`\`\`json
{
  "status": "healthy",
  "model_loaded": true,
  "storage_path": "/home/pi/PANGIL/detections"
}
\`\`\`

### 4.2 Test Frontend

1. Visit `https://pangil.vercel.app`
2. Upload an image
3. Verify detection results appear

## Troubleshooting

### Backend Not Loading Model

- Check `MODEL_PATH` environment variable
- Verify model file exists at the specified path
- Check logs for error messages

### CORS Errors

- Verify `FRONTEND_URL` in backend environment variables
- Check CORS middleware in `main.py`

### ngrok Tunnel Disconnects

- Use systemd service to auto-restart
- Check internet connection on Raspberry Pi
- Consider upgrading to ngrok paid plan for static domains

## Security Best Practices

1. **Environment Variables**: Never commit secrets
2. **CORS**: Restrict to your domain
3. **HTTPS**: Always use HTTPS (ngrok provides this)
4. **Rate Limiting**: Implement API rate limiting
5. **File Storage**: Secure access to detection history files

## Monitoring

### Monitor ngrok Tunnel

\`\`\`bash
# View ngrok dashboard
open http://127.0.0.1:4040
\`\`\`

### Monitor FastAPI Backend

\`\`\`bash
# View logs
sudo journalctl -u pangil-backend.service -f
\`\`\`

## Production Checklist

- [ ] ngrok tunnel running with authentication
- [ ] FastAPI backend running on Raspberry Pi
- [ ] NEXT_PUBLIC_BACKEND_URL set in Vercel
- [ ] CORS configured correctly
- [ ] Health check endpoint responding
- [ ] Test detection working end-to-end
- [ ] Systemd services configured for auto-start
- [ ] Monitoring and logging set up
