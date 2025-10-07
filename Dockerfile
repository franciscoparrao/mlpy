# MLPY Production Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    redis \
    prometheus-client \
    python-json-logger

# Copy the MLPY framework
COPY mlpy/ ./mlpy/
COPY setup.py .
COPY README.md .

# Install MLPY in development mode
RUN pip install -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs /app/experiments

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MLPY_ENV=production
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data
ENV LOG_PATH=/app/logs

# Expose ports
EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "mlpy.mlops.serving:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]