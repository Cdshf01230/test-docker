# Dockerfile 
FROM python:3.11-slim

# System deps for OpenCV (headless) and common runtime needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code + model weights
COPY main.py /app/main.py
COPY yolov8n-pose.pt /app/yolov8n-pose.pt

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
