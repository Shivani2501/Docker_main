# api-service/app.py

import os
import json
import numpy as np
import torch
from flask import Flask, request, jsonify
from inference import load_model, predict_action

app = Flask(__name__)

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pth")

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Predict action for a given state."""
    try:
        data = request.get_json()
        state = data.get('state')
        
        if state is None:
            return jsonify({"error": "No state provided in request"}), 400
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Get action prediction
        action, q_values = predict_action(model, state_tensor)
        
        response = {
            "action": int(action),
            "q_values": q_values.tolist(),
            "state": state
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    try:
        models_dir = "/app/models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        models_info = []
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_info = {
                "name": model_file,
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
                "created": os.path.getctime(model_path)
            }
            models_info.append(model_info)
        
        return jsonify({"models": models_info}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_model', methods=['POST'])
def set_model():
    """Set the active model."""
    global model
    
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if model_path is None:
            return jsonify({"error": "No model path provided"}), 400
        
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model not found at {model_path}"}), 404
        
        # Load new model
        new_model = load_model(model_path, device)
        model = new_model
        
        return jsonify({"status": "success", "model_path": model_path}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/simulate_episode', methods=['GET'])
def simulate_episode():
    """Simulate an episode using the current model."""
    try:
        import gym
        
        # First check if model is loaded
        if model is None:
            return jsonify({"error": "No model loaded"}), 400
            
        # Create environment
        env = gym.make('CartPole-v1')
        # Handle new reset API which might return (state, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Extract state from tuple
        else:
            state = reset_result

        done = False
        total_reward = 0
        steps = 0
        trajectory = []
        
        # Run episode
        while not done and steps < 500:  # Cap at 500 steps
            state_tensor = torch.FloatTensor(state).to(device)
            action, q_values = predict_action(model, state_tensor)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            trajectory.append({
                "state": state.tolist(),
                "action": int(action),
                "reward": float(reward),
                "next_state": next_state.tolist(),
                "q_values": q_values.tolist()
            })
            
            state = next_state
        
        result = {
            "total_reward": total_reward,
            "steps": steps,
            "trajectory": trajectory
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in simulate_episode: {error_msg}")
        return jsonify({"error": str(e), "traceback": error_msg}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    """Evaluate model performance over multiple episodes."""
    try:
        episodes = int(request.args.get('episodes', 100))
        
        import gym
        
        # Create environment
        env = gym.make('CartPole-v1')
        
        scores = []
        for episode in range(episodes):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # Extract state from tuple
            else:
                state = reset_result
            state = np.array(state) if isinstance(state, tuple) else state
            
            done = False
            episode_score = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(device)
                action, _ = predict_action(model, state_tensor)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_score += reward
                
                state = next_state
            
            scores.append(episode_score)
        
        results = {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std_dev": float(np.std(scores)),
            "episodes": episodes,
            "scores": scores
        }
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load the model at startup
    model = load_model(MODEL_PATH, device)
    app.run(host='0.0.0.0', port=8000)