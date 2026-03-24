# Dockerfile for Vertex AI LSTM training job
# This container will be used for training LSTM models on Vertex AI

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_CLOUD_PROJECT=stock-ml-trading-487
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/config/keys/ml-training-sa-key.json

# Create directories for outputs
RUN mkdir -p /app/outputs/models
RUN mkdir -p /app/outputs/logs

# Set the entrypoint
ENTRYPOINT ["python", "gcp/training/vertex_training_job.py"]
