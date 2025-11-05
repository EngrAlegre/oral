# PANGIL: Vercel + Raspberry Pi Integration Guide

## Overview
This guide explains how to securely connect your Vercel frontend (https://pangil.vercel.app) to your Raspberry Pi 5 backend running FastAPI through ngrok tunnel.

## Architecture
\`\`\`
Vercel Frontend (https://pangil.vercel.app)
         ↓
    HTTPS Request
         ↓
ngrok Tunnel (https://your-ngrok-url.ngrok.io)
         ↓
Raspberry Pi 5 (FastAPI Backend on port 8000)
         ↓
Local JSON File Storage (/home/pi/PANGIL/detections)
\`\`\`

## Prerequisites
- Raspberry Pi 5 with Raspberry Pi OS
- Python 3.9+ installed
- FastAPI backend running on port 8000
- ngrok account
- Vercel project deployed

## Step 1: Set Up Raspberry Pi Backend

### 1.1 Install Dependencies
\`\`\`bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
sudo apt-get install libatlas-base-dev libjasper-dev libtiff5 libjasper1 libharfbuzz0b libwebp6
\`\`\`

### 1.2 Create Virtual Environment
\`\`\`bash
cd /home/pi/PANGIL
python3 -m venv venv
source venv/bin/activate
\`\`\`

### 1.3 Install Python Dependencies
\`\`\`bash
pip install fastapi uvicorn tensorflow opencv-python pillow python-multipart
\`\`\`

### 1.4 Create .env File on Raspberry Pi
\`\`\`bash
# /home/pi/PANGIL/.env
MODEL_PATH=/home/kerljann/yolo/lesion.pt
FRONTEND_URL=https://pangil.vercel.app
ENVIRONMENT=production
STORAGE_DIR=/home/pi/PANGIL/detections
\`\`\`

### 1.5 Run FastAPI Backend
\`\`\`bash
cd /home/pi/PANGIL
source venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
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

You'll see output like:
\`\`\`
Forwarding                    https://your-ngrok-url.ngrok.io -> http://localhost:8000
\`\`\`

**Copy the HTTPS URL**

## Step 3: Configure Vercel Environment Variables

### 3.1 Go to Vercel Dashboard
1. Navigate to your PANGIL project
2. Go to **Settings → Environment Variables**

### 3.2 Add Environment Variables
\`\`\`
NEXT_PUBLIC_BACKEND_URL=https://your-ngrok-url.ngrok.io
MODEL_PATH=/home/kerljann/yolo/lesion.pt
FRONTEND_URL=https://pangil.vercel.app
ENVIRONMENT=production
\`\`\`

### 3.3 Redeploy Frontend
\`\`\`bash
git push  # Vercel will automatically redeploy
\`\`\`

## Step 4: Configure CORS on Raspberry Pi Backend

The FastAPI backend already has CORS configured for:
- `http://localhost:3000` (local development)
- `https://pangil.vercel.app` (production)
- Custom `FRONTEND_URL` from environment

## Step 5: Test the Connection

### 5.1 Test Health Check
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

### 5.2 Test from Vercel Frontend
1. Open https://pangil.vercel.app
2. Upload an image
3. Check browser console for API responses
4. Verify detection results appear

## Step 6: Set Up Auto-Start on Raspberry Pi

### 6.1 Create Systemd Service for FastAPI
\`\`\`bash
sudo nano /etc/systemd/system/pangil-backend.service
\`\`\`

Add:
\`\`\`ini
[Unit]
Description=PANGIL FastAPI Backend
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PANGIL
Environment="PATH=/home/pi/PANGIL/venv/bin"
ExecStart=/home/pi/PANGIL/venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`\`\`

### 6.2 Create Systemd Service for ngrok
\`\`\`bash
sudo nano /etc/systemd/system/pangil-ngrok.service
\`\`\`

Add:
\`\`\`ini
[Unit]
Description=PANGIL ngrok Tunnel
After=network.target pangil-backend.service

[Service]
Type=simple
User=pi
ExecStart=/usr/local/bin/ngrok http 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`\`\`

### 6.3 Enable Services
\`\`\`bash
sudo systemctl daemon-reload
sudo systemctl enable pangil-backend.service
sudo systemctl enable pangil-ngrok.service
sudo systemctl start pangil-backend.service
sudo systemctl start pangil-ngrok.service
\`\`\`

### 6.4 Check Status
\`\`\`bash
sudo systemctl status pangil-backend.service
sudo systemctl status pangil-ngrok.service
\`\`\`

## Step 7: Security Best Practices

### 7.1 Secure ngrok Tunnel
- Use ngrok authentication token
- Consider ngrok's paid plan for static domains
- Monitor ngrok dashboard for suspicious activity

### 7.2 HTTPS Only
- Always use HTTPS URLs (ngrok provides this automatically)
- Never expose HTTP endpoints to the internet

### 7.3 File Storage Security
- Restrict access to detection history files
- Use proper file permissions on Raspberry Pi

## Troubleshooting

### Issue: "Connection refused" from Vercel
**Solution**: 
- Verify ngrok tunnel is running: `ngrok status`
- Check FastAPI backend is running: `curl http://localhost:8000/health`
- Verify NEXT_PUBLIC_BACKEND_URL is correct in Vercel

### Issue: CORS errors in browser console
**Solution**:
- Verify FRONTEND_URL is set correctly on Raspberry Pi
- Check CORS middleware configuration in FastAPI

### Issue: ngrok tunnel keeps disconnecting
**Solution**:
- Use systemd service to auto-restart
- Check internet connection on Raspberry Pi

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
