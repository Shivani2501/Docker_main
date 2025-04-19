# training-service/train.py

import os
import time
import json
import numpy as np
import gym
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime
from dqn_agent import DQNAgent

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("training-service")

# Environment and training parameters
ENV_NAME = "CartPole-v1"
EPISODES = int(os.getenv("EPISODES", 1000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
DISCOUNT_FACTOR = float(os.getenv("DISCOUNT_FACTOR", 0.99))
EPSILON_START = float(os.getenv("EPSILON_START", 1.0))
EPSILON_END = float(os.getenv("EPSILON_END", 0.01))
EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", 0.995))
MODEL_DIR = "/app/models"
LOG_DIR = "/app/logs"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def connect_to_mlflow(max_retries=5, retry_delay=5):
    """Connect to MLflow with retry logic."""
    for attempt in range(max_retries):
        try:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://monitoring-service:5001")
            logger.info(f"Connecting to MLflow at {mlflow_uri}, attempt {attempt+1}/{max_retries}")
            mlflow.set_tracking_uri(mlflow_uri)
            # Test connection by creating experiment
            mlflow.set_experiment("DQN-CartPole")
            logger.info("Successfully connected to MLflow")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached, could not connect to MLflow")
                return False
            
def main():
    # Connect to MLflow with retry logic
    if not connect_to_mlflow():
        logger.error("Could not connect to MLflow, exiting")
        return
    
    # Create environment
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE
    )
    
    # Training stats
    scores = []
    average_scores = []
    losses = []
    epsilons = []
    best_score = 0
    training_start_time = time.time()
    
    with mlflow.start_run(run_name=f"DQN-CartPole-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "env_name": ENV_NAME,
            "episodes": EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "discount_factor": DISCOUNT_FACTOR,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
        })
        
        # Training loop
        logger.info(f"Starting training for {EPISODES} episodes")
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Get just the observation from reset() return
            state = np.array(state, dtype=np.float32)
            
            episode_score = 0
            episode_loss = 0
            done = False
            
            while not done:
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # Combine both signals to determine if episode is over
                
                episode_score += reward
                
                # Store transition in replay buffer
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Perform optimization step
                loss = agent.learn()
                if loss > 0:
                    episode_loss += loss
            
            # Save model if better
            if episode_score > best_score:
                best_score = episode_score
                agent.save(f"{MODEL_DIR}/best_model.pth")
                logger.info(f"New best model saved with score: {best_score}")
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                agent.save(f"{MODEL_DIR}/checkpoint_episode_{episode}.pth")
            
            # Track stats
            scores.append(episode_score)
            episode_avg_loss = episode_loss / episode_score if episode_score > 0 else 0
            losses.append(episode_avg_loss)
            epsilons.append(agent.epsilon)
            
            # Calculate average score over last 100 episodes
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            average_scores.append(avg_score)
            
            # Log metrics
            mlflow.log_metrics({
                "episode_score": episode_score,
                "average_score": avg_score,
                "loss": episode_avg_loss,
                "epsilon": agent.epsilon
            }, step=episode)
            
            # Save metrics to JSON for visualization service
            metrics = {
                "scores": scores,
                "average_scores": average_scores,
                "losses": losses,
                "epsilons": epsilons,
                "episodes": list(range(1, episode + 1))
            }
            with open(f"{LOG_DIR}/training_metrics.json", "w") as f:
                json.dump(metrics, f)
            
            # Print progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{EPISODES} | Score: {episode_score} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f}")
            
            # If solved (avg score > 475 for last 100 episodes), break
            if avg_score >= 475 and len(scores) >= 100:
                logger.info(f"Environment solved in {episode} episodes! Average score: {avg_score:.2f}")
                break
        
        # Save final model
        agent.save(f"{MODEL_DIR}/final_model.pth")
        
        # Log training time
        training_time = time.time() - training_start_time
        mlflow.log_metric("training_time", training_time)
        
        # Log model
        mlflow.pytorch.log_model(agent.q_network, "model")
        
        logger.info(f"Training completed in {training_time:.2f} seconds")

if __name__ == "__main__":
    main()
