import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import yaml
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from src.environments.frozenlake_env import make_frozenlake_env
from src.agents.dqn_agent import DQNAgent
from src.visualization.plot_results import plot_rewards, plot_gamma_comparison, plot_losses
from collections import deque
import torch
from src.utils import one_hot_state

# =============================================================================
# Directory Setup
# =============================================================================
RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


# =============================================================================
# Configuration Loader
# =============================================================================
def load_config():
    """
    Load experiment configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Utility: One-hot state encoding
# =============================================================================
def one_hot_state(state, state_size=16):
    """
    Convert a scalar state index to a one-hot encoded vector.

    Parameters
    ----------
    state : int
        Scalar representation of the environment state.
    state_size : int, optional
        Number of total states (default is 16 for FrozenLake 4x4).

    Returns
    -------
    np.ndarray
        One-hot encoded vector representation of the state.
    """
    state_vec = np.zeros(state_size, dtype=np.float32)
    state_vec[state] = 1.0
    return state_vec


# =============================================================================
# Training Function
# =============================================================================
def train_agent(gamma: float, episodes: int, is_slippery: bool, writer: SummaryWriter):
    """
    Train a DQN agent on the FrozenLake environment for a specific discount factor.

    Parameters
    ----------
    gamma : float
        Discount factor for future rewards.
    episodes : int
        Number of training episodes.
    is_slippery : bool
        Whether the environment should be slippery.
    writer : SummaryWriter
        TensorBoard writer for real-time logging.

    Returns
    -------
    tuple[list, list]
        (episode_rewards, episode_losses)
    """
    env = make_frozenlake_env(is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, gamma)
    episode_rewards, episode_avg_losses = [], []
    reward_window = deque(maxlen=100)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = one_hot_state(state, state_size)
        total_reward = 0
        done = False
        episode_losses = []
        steps = 0

        while not done:
            # === Agent action ===
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = one_hot_state(next_state, state_size)

            # === Memorize experience and learn ===
            agent.memorize(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)

            # === For Q-value statistics ===
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(state).to(agent.device))
                mean_q = q_values.mean().item()
                max_q = q_values.max().item()

            # === Update state and metrics ===
            state = next_state
            total_reward += reward
            steps += 1

        # === Episode-level metrics ===
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        reward_window.append(total_reward)
        moving_avg_reward = np.mean(reward_window)
        success = 1 if total_reward > 0 else 0  # Reached goal = success

        # === Log metrics to TensorBoard ===
        writer.add_scalar(f"Gamma_{gamma}/Reward", total_reward, episode)
        writer.add_scalar(f"Gamma_{gamma}/Moving_Avg_Reward", moving_avg_reward, episode)
        writer.add_scalar(f"Gamma_{gamma}/Avg_Loss", avg_loss, episode)
        writer.add_scalar(f"Gamma_{gamma}/Epsilon", agent.epsilon, episode)
        writer.add_scalar(f"Gamma_{gamma}/Episode_Length", steps, episode)
        writer.add_scalar(f"Gamma_{gamma}/Mean_Q_Value", mean_q, episode)
        writer.add_scalar(f"Gamma_{gamma}/Max_Q_Value", max_q, episode)
        writer.add_scalar(f"Gamma_{gamma}/Success", success, episode)

        episode_rewards.append(total_reward)
        episode_avg_losses.append(avg_loss)

        # === Progress output ===
        if episode % 50 == 0:
            print(f"[Gamma: {gamma:.2f}] Episode {episode}/{episodes} | "
                  f"Reward: {total_reward:.2f} | Moving Avg: {moving_avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Mean Q: {mean_q:.3f} | Success: {success}")

    env.close()
    return episode_rewards, episode_avg_losses


# =============================================================================
# Main Experiment Runner
# =============================================================================
def main():
    """
    Run DQN training experiments for multiple gamma values and log all metrics.
    """
    config = load_config()
    print("Loaded config:", config)

    env_config = config.get("environment", {})
    is_slippery = env_config.get("is_slippery", True)
    map_name = env_config.get("map_name", "4x4")
    reward_schedule = tuple(env_config.get("reward_schedule", [1, 0, 0]))

    training_config = config.get("training", {})
    gammas = training_config.get("gammas", [0.1, 0.99])
    episodes = training_config.get("episodes", 2000)

    all_rewards, all_losses = {}, {}

    for gamma in gammas:
        print(f"\n🚀 Starting training for gamma = {gamma}")
        writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, f"gamma_{gamma:.2f}"))

        # Pass reward schedule to environment creation
        env = make_frozenlake_env(
            is_slippery=is_slippery,
            map_name=map_name,
            reward_schedule=reward_schedule,
        )

        rewards, losses = train_agent(gamma, episodes, is_slippery, writer)
        all_rewards[gamma], all_losses[gamma] = rewards, losses

        writer.close()

        # Save metrics and plots...
        np.save(os.path.join(LOGS_DIR, f"rewards_gamma_{gamma:.2f}.npy"), rewards)
        np.save(os.path.join(LOGS_DIR, f"losses_gamma_{gamma:.2f}.npy"), losses)

        plot_rewards(rewards, gamma, os.path.join(PLOTS_DIR, f"reward_gamma_{gamma:.2f}.png"))
        plot_losses(losses, gamma, os.path.join(PLOTS_DIR, f"loss_gamma_{gamma:.2f}.png"))

    plot_gamma_comparison(all_rewards, os.path.join(PLOTS_DIR, "gamma_comparison.png"))

    print("\n✅ Training complete!")
    print("💡 View TensorBoard results with:")
    print(f"   tensorboard --logdir {TENSORBOARD_DIR}\n")



if __name__ == "__main__":
    main()
