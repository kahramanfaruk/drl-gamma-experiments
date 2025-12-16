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

def plot_dual_axis(data_key, ylabel, filename, title, agent_type="dqn"):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, alpha=0.3)

    strategies = ["fixed", "adaptive"]
    colors = {"fixed": "tab:red", "adaptive": "tab:green"}
    
    lines = []
    labels = []
    has_data = False

    for strat in strategies:
        path_new = os.path.join(DATA_DIR, f"results_{agent_type}_{strat}.npy")
        path_old = os.path.join(DATA_DIR, f"results_{strat}.npy")
        
        path = None
        if os.path.exists(path_new):
            path = path_new
        elif os.path.exists(path_old) and agent_type.lower() == "dqn":
            path = path_old
            print(f" Warning: Using old filename format: {path_old}")
        
        if path is None or not os.path.exists(path):
            print(f" Missing data for {agent_type.upper()} | {strat}: {path_new}")
            continue
        
        has_data = True
        data = np.load(path, allow_pickle=True).item()
        
        if data_key not in data:
            print(f" Key '{data_key}' not found in data for {strat}")
            continue
            
        metric_data = moving_average(data[data_key], 50)
        gammas = data.get("gammas", [])
        
        l, = ax1.plot(metric_data, label=f"{strat.capitalize()} Strategy", color=colors[strat], linewidth=2)
        lines.append(l)
        labels.append(f"{strat.capitalize()}")
        
        if strat == "adaptive" and len(gammas) > 0:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Discount Factor (γ)", color="tab:blue", fontsize=12, fontweight='bold')
            plot_gammas = gammas[:len(metric_data)]
            l2, = ax2.plot(plot_gammas, color="tab:blue", linestyle="--", alpha=0.6, label="Gamma Schedule")
            lines.append(l2)
            labels.append("Gamma (Adaptive)")
            ax2.tick_params(axis='y', labelcolor="tab:blue")
            ax2.set_ylim(0.75, 1.01)

    if not has_data:
        plt.close()
        print(f" No data found for {agent_type.upper()}. Skipping plot.")
        return

    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    plt.title(f"{title} | {agent_type.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{agent_type}_{filename}"))
    print(f" Saved plot: {agent_type}_{filename}")
    plt.close()

if __name__ == "__main__":
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    
    plot_dual_axis("rewards", "Avg Reward (Win 50)", "adaptive_rewards.png", 
                   "Impact of Adaptive Gamma on Learning Speed", agent_type)
    
    plot_dual_axis("q_values", "Avg Max Q-Value", "adaptive_q_values.png", 
                   "Growth of Value Estimates (Q) with Gamma", agent_type)