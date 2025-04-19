# monitoring-service/monitor.py

import os
import mlflow
import mlflow.pytorch
import threading
import subprocess
from flask import Flask, jsonify, request
from waitress import serve
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("monitoring-service")

# Create directories
os.makedirs("/app/mlflow", exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)

# Function to start MLflow server in a separate thread
def start_mlflow_server():
    logger.info("Starting MLflow server process")
    mlflow_cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "/app/mlflow/artifacts",
        "--host", "0.0.0.0",
        "--port", "5001"  # Use different port than Flask
    ]
    subprocess.run(mlflow_cmd)

# Start MLflow server in a separate thread
logger.info("Starting MLflow tracking server")
mlflow_thread = threading.Thread(target=start_mlflow_server)
mlflow_thread.daemon = True  # Thread will exit when main program exits
mlflow_thread.start()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Wait a bit for MLflow server to start up
time.sleep(5)

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "service": "DQN Monitoring Service",
        "status": "running",
        "endpoints": {
            "/": "This help message",
            "/mlflow": "MLflow UI (redirects to MLflow server)",
            "/health": "Health check endpoint",
            "/experiments": "List all experiments",
            "/runs/{experiment_id}": "List runs for an experiment",
            "/metrics/{run_id}": "Get metrics for a run"
        }
    })

@app.route('/mlflow', methods=['GET'])
def mlflow_redirect():
    """Redirect to MLflow UI."""
    return jsonify({
        "message": "MLflow server running on port 5001",
        "url": "http://localhost:5001"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/experiments', methods=['GET'])
def list_experiments():
    """List all experiments."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        result = []
        for exp in experiments:
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
                "last_update_time": exp.last_update_time
            })
        
        return jsonify({"experiments": result}), 200
    
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/runs/<experiment_id>', methods=['GET'])
def list_runs(experiment_id):
    """List runs for an experiment."""
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_id)
        
        result = []
        for run in runs:
            run_info = run.info
            run_data = run.data
            
            result.append({
                "run_id": run_info.run_id,
                "experiment_id": run_info.experiment_id,
                "status": run_info.status,
                "start_time": run_info.start_time,
                "end_time": run_info.end_time,
                "artifact_uri": run_info.artifact_uri,
                "metrics": run_data.metrics,
                "params": run_data.params,
                "tags": run_data.tags
            })
        
        return jsonify({"runs": result}), 200
    
    except Exception as e:
        logger.error(f"Error listing runs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics/<run_id>', methods=['GET'])
def get_metrics(run_id):
    """Get metrics for a run."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        metrics_history = {}
        for key in run.data.metrics.keys():
            metrics = client.get_metric_history(run_id, key)
            metrics_history[key] = [
                {"step": m.step, "timestamp": m.timestamp, "value": m.value}
                for m in metrics
            ]
        
        return jsonify({
            "run_id": run_id,
            "metrics": run.data.metrics,
            "metrics_history": metrics_history
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare_runs', methods=['POST'])
def compare_runs():
    """Compare multiple runs."""
    try:
        data = request.get_json()
        run_ids = data.get('run_ids', [])
        
        if not run_ids:
            return jsonify({"error": "No run IDs provided"}), 400
        
        client = mlflow.tracking.MlflowClient()
        comparison = {}
        
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                comparison[run_id] = {
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "status": run.info.status
                }
            except Exception as e:
                comparison[run_id] = {"error": str(e)}
        
        return jsonify({"comparison": comparison}), 200
    
    except Exception as e:
        logger.error(f"Error comparing runs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/best_model', methods=['GET'])
def get_best_model():
    """Get the best model based on a metric."""
    try:
        experiment_name = request.args.get('experiment_name', 'DQN-CartPole')
        metric_name = request.args.get('metric_name', 'average_score')
        max_results = int(request.args.get('max_results', 5))
        ascending = request.args.get('ascending', 'false').lower() == 'true'
        
        # Get experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return jsonify({"error": f"Experiment '{experiment_name}' not found"}), 404
        
        # Search for runs
        order_by = [f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=order_by,
            max_results=max_results
        )
        
        if runs.empty:
            return jsonify({"error": "No runs found for the experiment"}), 404
        
        # Convert to JSON-serializable format
        result = []
        for _, run in runs.iterrows():
            result.append({
                "run_id": run['run_id'],
                "metrics": {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.")},
                "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
                "start_time": run['start_time'],
                "end_time": run['end_time'],
                "status": run['status'],
                "artifact_uri": run['artifact_uri']
            })
        
        return jsonify({"best_models": result}), 200
    
    except Exception as e:
        logger.error(f"Error getting best model: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start Flask app
    logger.info("Starting Flask app")
    serve(app, host='0.0.0.0', port=5000)