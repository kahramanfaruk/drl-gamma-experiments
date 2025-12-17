import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# =============================================================================
# Configuration
# =============================================================================
MAP_NAME = "8x8" 
DATA_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", MAP_NAME, "data")
PLOTS_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", MAP_NAME, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Use a clean style for professional plots
plt.style.use('bmh') # 'bmh' is a clean, scientific built-in style

def moving_average(data, window=50):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_dual_axis(data_key, ylabel, output_filename, title, agent_type="dqn"):
    """
    Generates a professional dual-axis plot comparing strategies.
    Includes raw data traces for scientific transparency.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Configure Left Axis (Metric)
    ax1.set_xlabel("Episodes", fontsize=14, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    strategies = ["fixed", "adaptive"]
    
    # Professional Color Palette
    colors = {
        "fixed": "#D32F2F",    # Deep Red
        "adaptive": "#2E7D32"  # Forest Green
    }
    
    lines = []
    labels = []
    data_found = False

    for strat in strategies:
        path_a = os.path.join(DATA_DIR, f"results_{agent_type}_{strat}_{MAP_NAME}.npy")
        path_b = os.path.join(DATA_DIR, f"results_{strat}_{MAP_NAME}.npy")
        path_c = os.path.join(DATA_DIR, f"results_{agent_type}_{strat}.npy")
        path_d = os.path.join(DATA_DIR, f"results_{strat}.npy")
        
        if os.path.exists(path_a):
            path = path_a
        elif os.path.exists(path_b):
            path = path_b
        elif os.path.exists(path_c):
            path = path_c
        elif os.path.exists(path_d):
            path = path_d
        else:
            print(f"Missing data for {agent_type.upper()} | {strat}")
            continue
            
        data_found = True
        print(f"Loading: {path}")
        data = np.load(path, allow_pickle=True).item()
        
        if data_key not in data:
            print(f"Key '{data_key}' not found.")
            continue

        raw_data = data[data_key]
        gammas = data["gammas"]
        
        # 1. Plot Raw Data (Faint) - Shows variance/noise
        ax1.plot(raw_data, color=colors[strat], alpha=0.15, linewidth=1)

        # 2. Plot Smoothed Data (Solid) - Shows trend
        smooth_data = moving_average(raw_data, 50)
        # Pad smoothed data to align with raw x-axis
        padding = [np.nan] * (len(raw_data) - len(smooth_data))
        aligned_smooth = np.concatenate([padding, smooth_data])
        
        l, = ax1.plot(aligned_smooth, label=f"{strat.capitalize()} Strategy", 
                      color=colors[strat], linewidth=2.5)
        lines.append(l)
        labels.append(f"{strat.capitalize()} Trend")
        
        # 3. Add Final Value Annotation
        if len(smooth_data) > 0:
            final_val = smooth_data[-1]
            ax1.text(len(raw_data), final_val, f"{final_val:.2f}", 
                     color=colors[strat], fontweight='bold', va='center')

        # --- RIGHT AXIS: GAMMA SCHEDULE ---
        if strat == "adaptive":
            ax2 = ax1.twinx()
            ax2.set_ylabel("Discount Factor (γ)", color="#1565C0", fontsize=14, fontweight='bold')
            
            # Align Gamma plot
            plot_gammas = gammas[:len(raw_data)]
            
            l2, = ax2.plot(plot_gammas, color="#1565C0", linestyle="--", 
                           linewidth=2, label="Gamma Schedule", alpha=0.7)
            
            lines.append(l2)
            labels.append("Gamma (Adaptive)")
            
            ax2.tick_params(axis='y', labelcolor="#1565C0")
            ax2.set_ylim(0.75, 1.01)
            ax2.grid(False) # Disable grid for secondary axis to avoid clutter

    if not data_found:
        print(f"\nCRITICAL: No data found for {agent_type.upper()}. Please run the training script first.")
        plt.close()
        return

    # Combined Legend with shadow
    ax1.legend(lines, labels, loc="upper left", fontsize=12, frameon=True, shadow=True)
    
    plt.title(f"{title} | {agent_type.upper()} | {MAP_NAME} Map", fontsize=16, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    agent_type = "dqn"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agent_type = config.get("agent", {}).get("type", agent_type)
    
    agent_type = agent_type.lower()
    
    if agent_type not in ["dqn", "qlearning"]:
        print(f"Unknown agent type: {agent_type}. Use 'dqn' or 'qlearning'")
        sys.exit(1)
    
    print(f"Generating plots for {agent_type.upper()} | {MAP_NAME} environment...")
    
    plot_dual_axis("rewards", "Avg Reward (Window 50)", 
                   f"{agent_type}_adaptive_rewards_{MAP_NAME}_pro.png", 
                   "Impact of Adaptive Gamma on Learning Speed", agent_type)
    
    plot_dual_axis("q_values", "Avg Max Q-Value", 
                   f"{agent_type}_adaptive_q_values_{MAP_NAME}_pro.png", 
                   "Growth of Value Estimates (Q) with Gamma", agent_type)