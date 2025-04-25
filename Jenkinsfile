pipeline {
    agent any
    
    environment {
        REGISTRY = "doublerandomexp25"
        REGISTRY_CREDENTIAL = 'docker-hub-credentials'
        VERSION = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'ls -la'
            }
        }
        
        stage('Build Images') {
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
                withCredentials([usernamePassword(credentialsId: REGISTRY_CREDENTIAL, passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh "echo \$DOCKER_PASSWORD | docker login -u \$DOCKER_USERNAME --password-stdin"
                    
                    sh "docker push ${REGISTRY}/api-service:${VERSION}"
                    sh "docker push ${REGISTRY}/monitoring-service:${VERSION}"
                    sh "docker push ${REGISTRY}/training-service:${VERSION}"
                    sh "docker push ${REGISTRY}/visualization-service:${VERSION}"
                    
                    sh "docker push ${REGISTRY}/api-service:latest"
                    sh "docker push ${REGISTRY}/monitoring-service:latest"
                    sh "docker push ${REGISTRY}/training-service:latest"
                    sh "docker push ${REGISTRY}/visualization-service:latest"
                    
                    sh "docker logout"
                }
            }
        }
        
        stage('Update Kubernetes Manifests') {
            steps {
                sh "mkdir -p k8s-processed"
                
                sh """
                for file in kubernetes/*.yaml; do
                    sed 's|\\\${REGISTRY}|${REGISTRY}|g; s|:latest|:${VERSION}|g' "\$file" > "k8s-processed/\$(basename \$file)"
                done
                """
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                withKubeConfig([credentialsId: 'kubernetes-config']) {
                    sh 'kubectl apply -f k8s-processed/'
                    
                    sh '''
                    echo "Waiting for deployments to be ready..."
                    kubectl wait --for=condition=Available --timeout=300s deployment/api-service
                    kubectl wait --for=condition=Available --timeout=300s deployment/monitoring-service
                    kubectl wait --for=condition=Available --timeout=300s deployment/training-service
                    kubectl wait --for=condition=Available --timeout=300s deployment/visualization-service
                    '''
                }
            }
        }
    }
    
    post {
        always {
            sh 'rm -rf k8s-processed'
            
            sh """
            docker rmi ${REGISTRY}/api-service:${VERSION} || true
            docker rmi ${REGISTRY}/monitoring-service:${VERSION} || true
            docker rmi ${REGISTRY}/training-service:${VERSION} || true
            docker rmi ${REGISTRY}/visualization-service:${VERSION} || true
            docker rmi ${REGISTRY}/api-service:latest || true
            docker rmi ${REGISTRY}/monitoring-service:latest || true
            docker rmi ${REGISTRY}/training-service:latest || true
            docker rmi ${REGISTRY}/visualization-service:latest || true
            """
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            echo 'Pipeline failed!'
        }
    }
}