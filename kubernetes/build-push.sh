#!/bin/bash
# kubernetes/build-push.sh

# Set your container registry here
REGISTRY="doublerandomexp25"  # I see you're using this registry from the error message

# Get the parent directory path (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="latest"

echo "Project root: $PROJECT_ROOT"

# Build images
echo "Building Docker images..."
docker build -t ${REGISTRY}/api-service:${TAG} "${PROJECT_ROOT}/api-service"
docker build -t ${REGISTRY}/monitoring-service:${TAG} "${PROJECT_ROOT}/monitoring-service"
docker build -t ${REGISTRY}/training-service:${TAG} "${PROJECT_ROOT}/training-service"
docker build -t ${REGISTRY}/visualization-service:${TAG} "${PROJECT_ROOT}/visualization-service"

# Push images
echo "Pushing images to registry..."
docker push ${REGISTRY}/api-service:${TAG}
docker push ${REGISTRY}/monitoring-service:${TAG}
docker push ${REGISTRY}/training-service:${TAG}
docker push ${REGISTRY}/visualization-service:${TAG}

echo "Done!"