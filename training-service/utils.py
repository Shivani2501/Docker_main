# training-service/utils.py

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def plot_training_results(episodes, scores, avg_scores, save_path=None):
    """Plot training results."""
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, label='Score')
    plt.plot(episodes, avg_scores, label='Average Score (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Training Progress')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()