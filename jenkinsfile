pipeline {
    agent any
    
    environment {
        // Define environment variables used in the pipeline
        REGISTRY = credentials('docker-registry-url') // e.g., 'doublerandomexp25' or 'docker.io/doublerandomexp25'
        REGISTRY_CREDENTIAL = 'docker-hub-credentials' // Jenkins credential ID
        VERSION = "${env.BUILD_NUMBER}"
        EMAIL_RECIPIENTS = 'your-email@example.com' // Set your email address here
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout code from the repository
                checkout scm
                
                // Print environment for debugging
                sh 'ls -la'
            }
        }
        
        stage('Build Images') {
            steps {
                script {
                    // Build the Docker images
                    sh "docker build -t ${REGISTRY}/api-service:${VERSION} ./api-service"
                    sh "docker build -t ${REGISTRY}/monitoring-service:${VERSION} ./monitoring-service"
                    sh "docker build -t ${REGISTRY}/training-service:${VERSION} ./training-service"
                    sh "docker build -t ${REGISTRY}/visualization-service:${VERSION} ./visualization-service"
                    
                    // Also tag as latest
                    sh "docker tag ${REGISTRY}/api-service:${VERSION} ${REGISTRY}/api-service:latest"
                    sh "docker tag ${REGISTRY}/monitoring-service:${VERSION} ${REGISTRY}/monitoring-service:latest"
                    sh "docker tag ${REGISTRY}/training-service:${VERSION} ${REGISTRY}/training-service:latest"
                    sh "docker tag ${REGISTRY}/visualization-service:${VERSION} ${REGISTRY}/visualization-service:latest"
                }
            }
        }
        
        stage('Push Images') {
            steps {
                script {
                    try {
                        // Log in to Docker registry
                        docker.withRegistry('', REGISTRY_CREDENTIAL) {
                            // Push the images
                            sh "docker push ${REGISTRY}/api-service:${VERSION}"
                            sh "docker push ${REGISTRY}/monitoring-service:${VERSION}"
                            sh "docker push ${REGISTRY}/training-service:${VERSION}"
                            sh "docker push ${REGISTRY}/visualization-service:${VERSION}"
                            
                            // Push latest tags
                            sh "docker push ${REGISTRY}/api-service:latest"
                            sh "docker push ${REGISTRY}/monitoring-service:latest"
                            sh "docker push ${REGISTRY}/training-service:latest"
                            sh "docker push ${REGISTRY}/visualization-service:latest"
                        }
                    } catch (Exception e) {
                        // Send notification email if pushing fails
                        emailext (
                            subject: "ERROR: Docker Push Failed - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                            body: """
                            <p>Docker push failed in the Jenkins pipeline.</p>
                            <p><b>Error details:</b><br>${e.getMessage()}</p>
                            <p>Check the console output at: ${env.BUILD_URL}</p>
                            """,
                            to: "${EMAIL_RECIPIENTS}",
                            mimeType: 'text/html'
                        )
                        throw e  // Re-throw the exception to mark the stage as failed
                    }
                }
            }
        }
        
        stage('Update Kubernetes Manifests') {
            steps {
                script {
                    // Create a temporary directory for processed manifests
                    sh "mkdir -p k8s-processed"
                    
                    // Update image tags in Kubernetes manifests
                    sh """
                    for file in kubernetes/*.yaml; do
                        sed 's|\${REGISTRY}|${REGISTRY}|g; s|:latest|:${VERSION}|g' "\$file" > "k8s-processed/\$(basename \$file)"
                    done
                    """
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    try {
                        // Use kubectl to apply the manifests
                        withKubeConfig([credentialsId: 'kubernetes-config']) {
                            sh 'kubectl apply -f k8s-processed/'
                            
                            // Check deployment status
                            sh '''
                            echo "Waiting for deployments to be ready..."
                            kubectl wait --for=condition=Available --timeout=300s deployment/api-service
                            kubectl wait --for=condition=Available --timeout=300s deployment/monitoring-service
                            kubectl wait --for=condition=Available --timeout=300s deployment/training-service
                            kubectl wait --for=condition=Available --timeout=300s deployment/visualization-service
                            '''
                        }
                    } catch (Exception e) {
                        // Send notification email if deployment fails
                        emailext (
                            subject: "ERROR: Kubernetes Deployment Failed - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                            body: """
                            <p>Kubernetes deployment failed in the Jenkins pipeline.</p>
                            <p><b>Error details:</b><br>${e.getMessage()}</p>
                            <p>Check the console output at: ${env.BUILD_URL}</p>
                            """,
                            to: "${EMAIL_RECIPIENTS}",
                            mimeType: 'text/html'
                        )
                        throw e  // Re-throw the exception to mark the stage as failed
                    }
                }
            }
        }
    }
    
    post {
        always {
            // Clean up processed manifests
            sh 'rm -rf k8s-processed'
            
            // Clean up local Docker images to save space
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
            // General failure notification
            emailext (
                subject: "FAILED: Pipeline - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: """
                <p>The Jenkins pipeline has failed.</p>
                <p>Check the console output at: ${env.BUILD_URL}</p>
                """,
                to: "${EMAIL_RECIPIENTS}",
                mimeType: 'text/html'
            )
        }
    }
}
