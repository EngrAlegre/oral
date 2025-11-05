#!/bin/bash
# Setup script for Raspberry Pi PANGIL backend

set -e

echo "🚀 PANGIL Raspberry Pi Setup Script"
echo "===================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libatlas-base-dev libjasper-dev libtiff5 libjasper1 libharfbuzz0b libwebp6

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
cd /home/pi/PANGIL
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install fastapi uvicorn tensorflow opencv-python pillow pymongo python-multipart slowapi

# Download ngrok
echo "🌐 Installing ngrok..."
wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.zip
unzip -q ngrok-v3-stable-linux-arm64.zip
sudo mv ngrok /usr/local/bin/
rm ngrok-v3-stable-linux-arm64.zip

# Create systemd services
echo "⚙️  Creating systemd services..."
sudo tee /etc/systemd/system/pangil-backend.service > /dev/null <<EOF
[Unit]
Description=PANGIL FastAPI Backend
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PANGIL
Environment="PATH=/home/pi/PANGIL/venv/bin"
ExecStart=/home/pi/PANGIL/venv/bin/python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/pangil-ngrok.service > /dev/null <<EOF
[Unit]
Description=PANGIL ngrok Tunnel
After=network.target pangil-backend.service

[Service]
Type=simple
User=pi
ExecStart=/usr/local/bin/ngrok start pangil
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable services
echo "✅ Enabling systemd services..."
sudo systemctl daemon-reload
sudo systemctl enable pangil-backend.service
sudo systemctl enable pangil-ngrok.service

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure .env file: nano /home/pi/PANGIL/.env"
echo "2. Configure ngrok: ngrok config add-authtoken YOUR_TOKEN"
echo "3. Start services: sudo systemctl start pangil-backend.service pangil-ngrok.service"
echo "4. Check status: sudo systemctl status pangil-backend.service"
echo "5. View ngrok URL: curl http://127.0.0.1:4040/api/tunnels"
