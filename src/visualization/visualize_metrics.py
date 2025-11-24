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

def plot_metric_comparison(env_type, metric_key, title, ylabel):
    plt.figure(figsize=(10, 6))
    
    # Define paths specifically for this environment type
    type_dir = os.path.join(STUDY_DIR, env_type)
    data_dir = os.path.join(type_dir, "data")
    plots_dir = os.path.join(type_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    has_data = False
    for gamma in GAMMAS:
        file_path = os.path.join(data_dir, f"metrics_gamma_{gamma}.npy")
        
        if not os.path.exists(file_path):
            print(f" Missing data for {gamma} in {env_type} (Path: {file_path})")
            continue
            
        has_data = True
        history = np.load(file_path, allow_pickle=True).item()
        
        if metric_key not in history:
            print(f"Key {metric_key} not found.")
            continue

        data = history[metric_key]
        smoothed = moving_average(data, WINDOW)
        
        plt.plot(smoothed, label=f"Gamma = {gamma}", linewidth=2, alpha=0.8)

    if not has_data:
        plt.close()
        return

    plt.title(f"{title} ({env_type.capitalize()})", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Discount Factor")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(plots_dir, f"compare_{metric_key}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   > Saved plot: {save_path}")

def run_visualizations():
    types = ["deterministic", "stochastic"]
    
    for t in types:
        print(f"\n Generating plots for {t}...")
        plot_metric_comparison(t, "rewards", "Learning Curve (Avg Reward)", "Reward")
        plot_metric_comparison(t, "steps", "Behavioral Analysis (Avg Steps)", "Steps")
        plot_metric_comparison(t, "avg_q", "Confidence (Avg Max Q-Value)", "Q-Value")
        plot_metric_comparison(t, "loss", "Training Stability (Avg Loss)", "Loss")
        plot_metric_comparison(t, "success", "Reliability (Success Rate)", "Success %")

if __name__ == "__main__":
    run_visualizations()