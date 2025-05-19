#!/bin/bash

# Create directories in shared volume to ensure they exist
echo "Creating a script to initialize the shared volume..."

cat <<EOF > init-shared-volume.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: init-shared-volume
spec:
  template:
    spec:
      containers:
      - name: init-volume
        image: busybox
        command: ['sh', '-c', 'mkdir -p /data/logs /data/models /data/mlflow && echo "Directories created" && ls -la /data']
        volumeMounts:
        - name: shared-volume
          mountPath: /data
      restartPolicy: Never
      volumes:
      - name: shared-volume
        persistentVolumeClaim:
          claimName: shared-volume-claim
  backoffLimit: 0
EOF

echo "Applying init job..."
kubectl apply -f init-shared-volume.yaml

echo "Waiting for job to complete..."
kubectl wait --for=condition=complete job/init-shared-volume --timeout=30s

echo "Checking job logs..."
kubectl logs -l job-name=init-shared-volume

echo "Patching api-service deployment to add logs directory..."
kubectl patch deployment api-service --type=json -p='[
  {
    "op": "add", 
    "path": "/spec/template/spec/containers/0/volumeMounts/1", 
    "value": {
      "name": "shared-volume", 
      "mountPath": "/app/logs"
    }
  }
]'

echo "Restarting the deployments to pick up volume changes..."
kubectl rollout restart deployment api-service
kubectl rollout restart deployment visualization-service
kubectl rollout restart deployment monitoring-service

echo "Waiting for deployments to stabilize..."
kubectl rollout status deployment api-service
kubectl rollout status deployment visualization-service
kubectl rollout status deployment monitoring-service

echo "Check volume mounts in api-service:"
POD_NAME=$(kubectl get pod -l app=api-service -o jsonpath="{.items[0].metadata.name}")
kubectl exec -it $POD_NAME -- ls -la /app

echo "Setup complete! You should now be able to access logs."