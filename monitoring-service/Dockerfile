# monitoring-service/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && apt-get clean

# Copy source code
COPY monitor.py .

# Create directories
RUN mkdir -p /app/logs /app/mlflow

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Run the monitoring server
CMD ["python", "monitor.py"]