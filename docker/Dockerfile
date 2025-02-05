FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' api-user

# Set up cache directory for the user
ENV TRANSFORMERS_CACHE=/home/api-user/.cache/huggingface
RUN mkdir -p /home/api-user/.cache/huggingface && \
    chown -R api-user:api-user /home/api-user/.cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Create directory for model weights
RUN mkdir -p app/models/prelabeling/weights

# Download YOLOv8 weights
RUN cd app/models/prelabeling/weights && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-seg.pt -o yolov8x-seg.pt

# Copy project and set permissions
COPY . .
RUN chown -R api-user:api-user /app && \
    pip install -e .

USER api-user

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 