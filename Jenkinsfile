pipeline {
    agent any

    environment {
        REGISTRY = "doublerandomexp25"
        REGISTRY_CREDENTIAL = 'docker-hub-credential'
        VERSION = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'ls -la'
            }
        }

        stage('Build & Tag Images') {
            steps {
                sh "docker build -t ${REGISTRY}/api-service:${VERSION} ./api-service"
                sh "docker build -t ${REGISTRY}/monitoring-service:${VERSION} ./monitoring-service"
                sh "docker build -t ${REGISTRY}/training-service:${VERSION} ./training-service"
                sh "docker build -t ${REGISTRY}/visualization-service:${VERSION} ./visualization-service"
                
                sh "docker tag ${REGISTRY}/api-service:${VERSION} ${REGISTRY}/api-service:latest"
                sh "docker tag ${REGISTRY}/monitoring-service:${VERSION} ${REGISTRY}/monitoring-service:latest"
                sh "docker tag ${REGISTRY}/training-service:${VERSION} ${REGISTRY}/training-service:latest"
                sh "docker tag ${REGISTRY}/visualization-service:${VERSION} ${REGISTRY}/visualization-service:latest"
            }
        }

        stage('Push Images') {
            steps {
                withCredentials([usernamePassword(credentialsId: REGISTRY_CREDENTIAL, 
                                                passwordVariable: 'DOCKER_PASSWORD', 
                                                usernameVariable: 'DOCKER_USERNAME')]) {
                    sh '''
                        mkdir -p $HOME/.docker
                        echo '{"auths":{"https://index.docker.io/v1/":{"auth":"'$(echo -n $DOCKER_USERNAME:$DOCKER_PASSWORD | base64)'"}}}' > $HOME/.docker/config.json
                        docker push ${REGISTRY}/api-service:${VERSION}
                        docker push ${REGISTRY}/monitoring-service:${VERSION}
                        docker push ${REGISTRY}/training-service:${VERSION}
                        docker push ${REGISTRY}/visualization-service:${VERSION}
                        docker push ${REGISTRY}/api-service:latest
                        docker push ${REGISTRY}/monitoring-service:latest
                        docker push ${REGISTRY}/training-service:latest
                        docker push ${REGISTRY}/visualization-service:latest
                        rm -f $HOME/.docker/config.json
                    '''
                }
            }
        }

        stage('Process Kubernetes Manifests') {
            steps {
                sh '''
                rm -rf k8s-processed
                mkdir -p k8s-processed
                for file in kubernetes/*.yaml; do
                    base=$(basename $file)
                    if [ "$base" = "kustomization.yaml" ]; then
                        continue
                    fi
                    sed 's|\\${REGISTRY}|${REGISTRY}|g; s|:latest|:${VERSION}|g' "$file" > "k8s-processed/$base"
                done
                ls -la k8s-processed/
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        echo "Deploying to Kubernetes cluster..."
                        kubectl apply -f k8s-processed/ --validate=true
                        echo "Current deployment status:"
                        kubectl get pods --all-namespaces
                        kubectl get deployments --all-namespaces
                        echo "Waiting for deployments to be ready..."
                        kubectl wait --for=condition=Available --timeout=300s deployment/api-service --all-namespaces || true
                        kubectl wait --for=condition=Available --timeout=300s deployment/monitoring-service --all-namespaces || true
                        kubectl wait --for=condition=Available --timeout=300s deployment/training-service --all-namespaces || true
                        kubectl wait --for=condition=Available --timeout=300s deployment/visualization-service --all-namespaces || true
                        echo "Final deployment status:"
                        kubectl get pods --all-namespaces
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'rm -rf k8s-processed'
            sh '''
            docker rmi ${REGISTRY}/api-service:${VERSION} || true
            docker rmi ${REGISTRY}/monitoring-service:${VERSION} || true
            docker rmi ${REGISTRY}/training-service:${VERSION} || true
            docker rmi ${REGISTRY}/visualization-service:${VERSION} || true
            docker rmi ${REGISTRY}/api-service:latest || true
            docker rmi ${REGISTRY}/monitoring-service:latest || true
            docker rmi ${REGISTRY}/training-service:latest || true
            docker rmi ${REGISTRY}/visualization-service:latest || true
            '''
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed – dumping Kubernetes debug info…'
            withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                sh '''
                    echo "=== Pods ==="
                    kubectl get pods --all-namespaces
                    echo "=== Describe Pods ==="
                    kubectl describe pods --all-namespaces
                    echo "=== Events ==="
                    kubectl get events --all-namespaces
                '''
            }
        }
    }
}
