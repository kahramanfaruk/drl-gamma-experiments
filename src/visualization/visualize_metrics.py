import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Base study directory
STUDY_DIR = os.path.join(ROOT_DIR, "results", "gamma_study_analysis")

# Define Gammas used in the study
GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
WINDOW = 50

def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_metric_comparison(env_type, metric_key, title, ylabel, agent_type="dqn"):
    plt.figure(figsize=(10, 6))
    
    type_dir = os.path.join(STUDY_DIR, env_type)
    data_dir = os.path.join(type_dir, "data")
    plots_dir = os.path.join(type_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    has_data = False
    for gamma in GAMMAS:
        file_path = os.path.join(data_dir, f"metrics_{agent_type}_gamma_{gamma}.npy")
        
        if not os.path.exists(file_path):
            continue
            
        has_data = True
        history = np.load(file_path, allow_pickle=True).item()
        
        if metric_key not in history:
            continue

        data = history[metric_key]
        smoothed = moving_average(data, WINDOW)
        
        plt.plot(smoothed, label=f"Gamma = {gamma}", linewidth=2, alpha=0.8)

    if not has_data:
        plt.close()
        return

    plt.title(f"{title} | {agent_type.upper()} | {env_type.capitalize()}", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Discount Factor")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(plots_dir, f"compare_{agent_type}_{metric_key}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   > Saved plot: {save_path}")

def run_visualizations(agent_type="dqn"):
    types = ["deterministic", "stochastic"]
    
    for t in types:
        print(f"\n Generating plots for {agent_type.upper()} | {t}...")
        plot_metric_comparison(t, "rewards", "Learning Curve (Avg Reward)", "Reward", agent_type)
        plot_metric_comparison(t, "steps", "Behavioral Analysis (Avg Steps)", "Steps", agent_type)
        plot_metric_comparison(t, "avg_q", "Confidence (Avg Max Q-Value)", "Q-Value", agent_type)
        plot_metric_comparison(t, "loss", "Training Stability (Avg Loss)", "Loss", agent_type)
        plot_metric_comparison(t, "success", "Reliability (Success Rate)", "Success %", agent_type)

if __name__ == "__main__":
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    run_visualizations(agent_type)