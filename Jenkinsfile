pipeline {
    agent any
    
    environment {
        REGISTRY = "localhost"  // Use 'localhost' for local Minikube
        VERSION = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Connect to Minikube Docker') {
            steps {
                // Evaluate the Minikube docker-env
                sh '''
                    # Get Minikube Docker environment
                    eval $(minikube docker-env)
                    
                    # Export these for other stages
                    echo "export DOCKER_TLS_VERIFY=\\"$(minikube docker-env | grep DOCKER_TLS_VERIFY | cut -d '=' -f2 | tr -d '\\"\\')\\"" > docker-env.sh
                    echo "export DOCKER_HOST=\\"$(minikube docker-env | grep DOCKER_HOST | cut -d '=' -f2 | tr -d '\\"\\')\\"" >> docker-env.sh
                    echo "export DOCKER_CERT_PATH=\\"$(minikube docker-env | grep DOCKER_CERT_PATH | cut -d '=' -f2 | tr -d '\\"\\')\\"" >> docker-env.sh
                    
                    cat docker-env.sh
                '''
            }
        }
        
        stage('Build Images') {
            steps {
                sh '''
                    # Load Minikube Docker environment
                    source docker-env.sh
                    
                    # Build images directly in Minikube
                    docker build -t ${REGISTRY}/api-service:${VERSION} ./api-service
                    docker build -t ${REGISTRY}/monitoring-service:${VERSION} ./monitoring-service
                    docker build -t ${REGISTRY}/training-service:${VERSION} ./training-service
                    docker build -t ${REGISTRY}/visualization-service:${VERSION} ./visualization-service
                    
                    # Tag as latest too
                    docker tag ${REGISTRY}/api-service:${VERSION} ${REGISTRY}/api-service:latest
                    docker tag ${REGISTRY}/monitoring-service:${VERSION} ${REGISTRY}/monitoring-service:latest
                    docker tag ${REGISTRY}/training-service:${VERSION} ${REGISTRY}/training-service:latest
                    docker tag ${REGISTRY}/visualization-service:${VERSION} ${REGISTRY}/visualization-service:latest
                '''
            }
        }
        
        stage('Update Kubernetes Manifests') {
            steps {
                sh '''
                    mkdir -p k8s-processed
                    
                    # Update registry and add imagePullPolicy: Never for local images
                    for file in kubernetes/*.yaml; do
                        sed 's|${REGISTRY}|localhost|g; s|:latest|:${VERSION}|g; s|imagePullPolicy: IfNotPresent|imagePullPolicy: Never|g' "$file" > "k8s-processed/$(basename $file)"
                    done
                    
                    ls -la k8s-processed/
                '''
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                sh '''
                    # Use Minikube kubectl directly
                    minikube kubectl -- apply -f k8s-processed/
                    
                    # Show deployment status
                    minikube kubectl -- get pods
                    minikube kubectl -- get deployments
                '''
            }
        }
    }
    
    post {
        always {
            sh 'rm -rf k8s-processed docker-env.sh'
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            echo 'Pipeline failed!'
        }
    }
}