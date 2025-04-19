# api-service/inference.py

import torch
import torch.nn as nn
import numpy as np


class DQNNetwork(nn.Module):
    """Deep Q-Network for CartPole environment."""
    
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def load_model(model_path, device):
    """Load trained model from path."""
    try:
        # For CartPole
        state_size = 4
        action_size = 2
        
        # Create model architecture
        model = DQNNetwork(state_size, action_size)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['q_network_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        
        print(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def predict_action(model, state_tensor):
    """Predict action for a given state."""
    if len(state_tensor.shape) == 1:
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
    
    return action, q_values.squeeze().cpu().numpy()