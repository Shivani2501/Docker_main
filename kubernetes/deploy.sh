#!/bin/bash
export REGISTRY="doublerandomexp25"

# Create temporary directory for processed manifests
TMP_DIR=$(mktemp -d)
echo "Creating temporary directory for processed manifests: $TMP_DIR"

# Process each YAML file with envsubst
for file in *.yaml; do
  if [ "$file" != "kustomization.yaml" ]; then
    envsubst < "$file" > "$TMP_DIR/$file"
  fi
done

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests with remote images..."
kubectl apply -f "$TMP_DIR"

echo "Deployment complete!"