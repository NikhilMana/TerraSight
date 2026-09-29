# TerraSight // Fovea-LiDAR Deployment Guide

**Problem Statement**: SIH26053 — *Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception* (DRDO)  
**System**: Fovea-LiDAR Perception Suite  
**Version**: 1.0 (Production Candidate)  

---

## 1. System Requirements

### Hardware Targets
| Requirement | Minimum (Evaluation) | Recommended (Defense Production) | Edge Embedded (Robotics) |
| :--- | :--- | :--- | :--- |
| **Processor** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7/i9 or AMD Ryzen 7 | NVIDIA Jetson AGX Orin / Xavier |
| **GPU** | CPU Only (SIMD Vectorized) | NVIDIA RTX 3060 / 4050 / A100 | Integrated Ampere/Orin GPU |
| **RAM** | 8 GB | 16 GB | 16 - 32 GB LPDDR5 |
| **Storage** | 2 GB free SSD space | 5 GB SSD | 16 GB NVMe / eMMC |
| **Operating System** | Windows 10/11, Ubuntu 20.04/22.04 | Ubuntu 22.04 LTS Server | Jetson Linux (L4T / JetPack 5.x/6.x) |

---

## 2. Deployment Option A: Local Unified Server (Single-Command)

TerraSight is architected so that the FastAPI Python backend serves both the REST/WebSocket API **and** the built WebGL React SPA from port `8000`.

### On Windows
```cmd
# Double click or run:
start.bat
```

### On Linux / macOS
```bash
chmod +x start.sh
./start.sh
```

### Manual CLI Execution
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build React production bundle
cd frontend
npm install
npm run build
cd ..

# 3. Launch unified production server
python run_production.py --host 0.0.0.0 --port 8000
```
- Open `http://localhost:8000` in any web browser.
- Interactive API documentation available at `http://localhost:8000/docs`.

---

## 3. Deployment Option B: Docker Container Deployment

### Standard CPU Deployment
```bash
# Build and run with Docker Compose
docker compose up --build -d

# Check service logs
docker compose logs -f

# Check health status
curl http://localhost:8000/api/health
```

### NVIDIA GPU Passthrough Deployment
Ensure the **NVIDIA Container Toolkit** is installed on the host machine:
```bash
# Run with GPU profile
docker compose --profile gpu up --build -d
```

### Standalone Docker Run
```bash
# Build container image
docker build -t terrasight:latest .

# Run container
docker run -d --name terrasight -p 8000:8000 terrasight:latest
```

---

## 4. Deployment Option C: Cloud Deployment (AWS / GCP / Azure)

### AWS EC2 (Ubuntu 22.04 LTS, `g4dn.xlarge` or `t3.xlarge`)
```bash
# 1. Update system and install Docker
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker $USER

# 2. Clone repository
git clone <repo-url> TerraSight
cd TerraSight

# 3. Run production container
docker compose up --build -d

# 4. Configure Security Group:
# Allow Inbound TCP on Port 8000 (or configure Nginx Reverse Proxy with SSL on Port 443)
```

### Nginx Reverse Proxy Configuration (Production HTTPS)
```nginx
server {
    listen 80;
    server_name perception.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name perception.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/perception.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/perception.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 5. Deployment Option D: Edge Robotics (NVIDIA Jetson AGX Orin)

For deployment directly on autonomous unmanned ground vehicles (UGVs) or defense robotic platforms:

1. Flash JetPack 5.1+ / 6.0 with CUDA and PyTorch for Jetson.
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run with local Jetson TensorRT or PyTorch CUDA runtime:
   ```bash
   python3 run_production.py --host 0.0.0.0 --port 8000
   ```
4. Downstream ROS 2 navigation stacks can subscribe directly to the `/api/scenarios/run` or `/api/comparator` endpoints for real-time 2.5D active cell occupancy.

---

## 6. Verification & Health Monitoring

Verify deployment integrity with the built-in diagnostic test suite:
```bash
# Execute full system verification
python -m unittest discover -s tests
```

Expected output:
```
......................
----------------------------------------------------------------------
Ran 22 tests in ~1.2s

OK
```
