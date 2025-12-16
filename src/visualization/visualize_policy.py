import sys
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)


from src.agents import create_agent
from src.utils.utils import one_hot_state

STUDY_DIR = os.path.join(ROOT_DIR, "results", "gamma_study_analysis")

ACTION_ARROWS = {
    0: "←", 1: "↓", 2: "→", 3: "↑"
}


def load_agent(model_path, agent_type):
    if agent_type.lower() == "dqn":
        agent = create_agent("dqn", state_size=16, action_size=4, gamma=0.99, seed=0)
        try:
            agent.policy_net.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu'))
            )
            agent.policy_net.eval()
            return agent
        except FileNotFoundError:
            return None
    else:
        agent = create_agent("qlearning", state_size=16, action_size=4, gamma=0.99, seed=0)
        try:
            agent.q_table = np.load(model_path, allow_pickle=True)
            return agent
        except FileNotFoundError:
            return None


def plot_policy_map(model_filename, env_type, gamma, agent_type="dqn"):
    type_dir = os.path.join(STUDY_DIR, env_type)
    models_dir = os.path.join(type_dir, "models")
    plots_dir = os.path.join(type_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_filename)
    agent = load_agent(model_path, agent_type)

    if agent is None:
        print(f" Model not found: {model_path}")
        return

    rows, cols = 4, 4
    value_grid = np.zeros((rows, cols))
    policy_grid = np.empty((rows, cols), dtype=object)

    for s in range(16):
        row, col = divmod(s, cols)
        
        if agent_type.lower() == "dqn":
            state_vec = one_hot_state(s, 16)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor).numpy()[0]
        else:
            q_values = agent.q_table[s]

        best_action = np.argmax(q_values)
        max_q = np.max(q_values)

        value_grid[row, col] = max_q
        policy_grid[row, col] = ACTION_ARROWS[best_action]

    plt.figure(figsize=(8, 6))
    norm = SymLogNorm(linthresh=0.05, linscale=0.5, vmin=-1.0, vmax=1.0, base=10)

    sns.heatmap(value_grid, annot=policy_grid, fmt="", cmap="RdYlGn",
                cbar_kws={'label': 'Q-Value (Log-Scale around 0)'},
                linewidths=1, linecolor='gray',
                norm=norm)

    plt.title(f"Policy Map | {agent_type.upper()} | {env_type.capitalize()} | Gamma={gamma}", fontsize=14)

    save_path = os.path.join(plots_dir, f"map_{agent_type}_gamma_{gamma}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   > Saved map: {save_path}")


def generate_all_maps(agent_type="dqn"):
    GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    types = ["deterministic", "stochastic"]

    print(f"\n Generating Policy Maps (Heatmaps) for {agent_type.upper()}...")
    for t in types:
        for g in GAMMAS:
            if agent_type.lower() == "dqn":
                model_filename = f"{agent_type}_gamma_{g}.pth"
            else:
                model_filename = f"{agent_type}_gamma_{g}.npy"
            plot_policy_map(model_filename, t, g, agent_type)
    print(f"Done! Check plots folders inside {STUDY_DIR}")


if __name__ == "__main__":
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    generate_all_maps(agent_type)
