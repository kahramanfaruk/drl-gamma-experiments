import sys
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agents.dqn_agent import DQNAgent
from src.utils.utils import one_hot_state

# Base study directory
STUDY_DIR = os.path.join(ROOT_DIR, "results", "gamma_study_analysis")

ACTION_ARROWS = {
    0: "←", 1: "↓", 2: "→", 3: "↑"
}

def load_agent(model_path):
    agent = DQNAgent(state_size=16, action_size=4, gamma=0.99, seed=0)
    try:
        agent.policy_net.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
        agent.policy_net.eval()
        return agent
    except FileNotFoundError:
        return None

def plot_policy_map(model_filename, env_type, gamma):
    # Construct path based on env_type
    type_dir = os.path.join(STUDY_DIR, env_type)
    models_dir = os.path.join(type_dir, "models")
    plots_dir = os.path.join(type_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_filename)
    agent = load_agent(model_path)
    
    if agent is None:
        print(f" Model not found: {model_path}")
        return

    rows, cols = 4, 4
    value_grid = np.zeros((rows, cols))
    policy_grid = np.empty((rows, cols), dtype=object)
    
    for s in range(16):
        row, col = divmod(s, cols)
        state_vec = one_hot_state(s, 16)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).numpy()[0]
        
        best_action = np.argmax(q_values)
        max_q = np.max(q_values)
        
        value_grid[row, col] = max_q
        policy_grid[row, col] = ACTION_ARROWS[best_action]

    plt.figure(figsize=(8, 6))
    sns.heatmap(value_grid, annot=policy_grid, fmt="", cmap="RdYlGn", 
                cbar_kws={'label': 'Q-Value (Expected Return)'}, 
                linewidths=1, linecolor='gray')
    
    plt.title(f"Policy Map | {env_type.capitalize()} | Gamma={gamma}", fontsize=14)
    
    save_path = os.path.join(plots_dir, f"map_gamma_{gamma}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   > Saved map: {save_path}")

def generate_all_maps():
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    types = ["deterministic", "stochastic"]
    
    print("\n Generating Policy Maps (Heatmaps)...")
    for t in types:
        for g in gammas:
            plot_policy_map(f"dqn_gamma_{g}.pth", t, g)
    print(f"Done! Check plots folders inside {STUDY_DIR}")

if __name__ == "__main__":
    generate_all_maps()