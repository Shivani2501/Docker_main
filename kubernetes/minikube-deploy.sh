#!/bin/bash

# Point to Minikube's Docker daemon
eval $(minikube docker-env)

# Set registry to blank for local images
export REGISTRY="localhost"

# Build images directly in Minikube's Docker environment
echo "Building images in Minikube's Docker environment..."
docker build -t ${REGISTRY}/api-service:latest ./api-service/
docker build -t ${REGISTRY}/monitoring-service:latest ./monitoring-service/
docker build -t ${REGISTRY}/training-service:latest ./training-service/
docker build -t ${REGISTRY}/visualization-service:latest ./visualization-service/

# Create temporary directory for processed manifests
TMP_DIR=$(mktemp -d)
echo "Creating temporary directory for processed manifests: $TMP_DIR"

# Process each YAML file with envsubst
for file in kubernetes/*.yaml; do
  echo "Processing $file..."
  envsubst < "$file" > "$TMP_DIR/$(basename $file)"
done

# Modify imagePullPolicy in the processed files to ensure local images are used
echo "Setting imagePullPolicy to Never for local images..."
sed -i 's/imagePullPolicy: IfNotPresent/imagePullPolicy: Never/g' $TMP_DIR/*.yaml

# Apply all processed manifests
echo "Applying Kubernetes manifests with local images..."
kubectl apply -f "$TMP_DIR"

# Optional: clean up
rm -rf "$TMP_DIR"

echo "Deployment complete!"