import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", "data")
PLOTS_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_adaptive_comparison():
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # --- AXIS 1: PERFORMANCE (REWARD) ---
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("Average Reward (Window 50)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    strategies = ["fixed", "adaptive"]
    colors = {"fixed": "tab:red", "adaptive": "tab:green"}
    
    # Store handles for the legend
    lines = []
    labels = []

    for strat in strategies:
        path = os.path.join(DATA_DIR, f"results_{strat}.npy")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        data = np.load(path, allow_pickle=True).item()
        rewards = moving_average(data["rewards"], 50)
        gammas = data["gammas"]
        
        # Plot Reward Curve
        l, = ax1.plot(rewards, label=f"{strat.capitalize()} Strategy", color=colors[strat], linewidth=2)
        lines.append(l)
        labels.append(f"{strat.capitalize()} Reward")
        
        # --- AXIS 2: THE SCIENTIFIC VARIABLE (GAMMA) ---
        # We only plot the gamma curve for the Adaptive strategy to show the schedule
        if strat == "adaptive":
            ax2 = ax1.twinx()
            ax2.set_ylabel("Discount Factor (γ)", color="tab:blue", fontsize=12, fontweight='bold')
            
            # Match length of smoothing
            plot_gammas = gammas[:len(rewards)] 
            
            l2, = ax2.plot(plot_gammas, color="tab:blue", linestyle="--", alpha=0.6, label="Gamma Schedule")
            lines.append(l2)
            labels.append("Gamma Value (Adaptive)")
            
            ax2.tick_params(axis='y', labelcolor="tab:blue")
            ax2.set_ylim(0.75, 1.01) # Set limits to make gamma change clearly visible

    # Combined Legend
    ax1.legend(lines, labels, loc="upper left", fontsize=10, frameon=True)
    
    plt.title("Experimental Result: Adaptive Gamma vs Fixed Gamma", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "adaptive_vs_fixed_comparison.png")
    plt.savefig(save_path)
    print(f" Plot saved to {save_path}")

if __name__ == "__main__":
    plot_adaptive_comparison()