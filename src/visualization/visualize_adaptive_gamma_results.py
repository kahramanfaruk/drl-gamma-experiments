import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", "data")
PLOTS_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def moving_average(data, window=50):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_dual_axis(data_key, ylabel, filename, title):
    """
    Generic function to plot a metric (left axis) against the Gamma schedule (right axis).
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, alpha=0.3)

    strategies = ["fixed", "adaptive"]
    colors = {"fixed": "tab:red", "adaptive": "tab:green"}
    
    lines = []
    labels = []

    for strat in strategies:
        path = os.path.join(DATA_DIR, f"results_{strat}.npy")
        if not os.path.exists(path): 
            print(f"Missing data: {path}")
            continue
            
        data = np.load(path, allow_pickle=True).item()
        metric_data = moving_average(data[data_key], 50)
        gammas = data["gammas"]
        
        # Main Metric Plot
        l, = ax1.plot(metric_data, label=f"{strat.capitalize()} Strategy", color=colors[strat], linewidth=2)
        lines.append(l)
        labels.append(f"{strat.capitalize()}")
        
        # Plot Gamma Schedule (Only for adaptive)
        if strat == "adaptive":
            ax2 = ax1.twinx()
            ax2.set_ylabel("Discount Factor (γ)", color="tab:blue", fontsize=12, fontweight='bold')
            plot_gammas = gammas[:len(metric_data)]
            l2, = ax2.plot(plot_gammas, color="tab:blue", linestyle="--", alpha=0.6, label="Gamma Schedule")
            lines.append(l2)
            labels.append("Gamma (Adaptive)")
            ax2.tick_params(axis='y', labelcolor="tab:blue")
            ax2.set_ylim(0.75, 1.01)

    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f" Saved plot: {filename}")
    plt.close()

if __name__ == "__main__":
    # Plot 1: Performance
    plot_dual_axis("rewards", "Avg Reward (Win 50)", "adaptive_rewards.png", 
                   "Impact of Adaptive Gamma on Learning Speed")
    
    # Plot 2: Theoretical Validation (Q-Values)
    # This proves the network's value estimates grow as Gamma grows
    plot_dual_axis("q_values", "Avg Max Q-Value", "adaptive_q_values.png", 
                   "Growth of Value Estimates (Q) with Gamma")