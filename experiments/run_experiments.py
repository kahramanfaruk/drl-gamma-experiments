import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import yaml
import numpy as np
import gymnasium as gym
from src.environments.frozenlake_env import make_frozenlake_env
from src.agents.dqn_agent import DQNAgent
from src.visualization.plot_results import plot_rewards, plot_gamma_comparison


RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_config(config_path="/home/faruk/Desktop/drl-gamma-experiments/config.yaml"):
    """
    Load experiment configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the config.yaml file, by default "/home/faruk/Desktop/drl-gamma-experiments/config.yaml"

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def one_hot_state(state, state_size=16):
    """
    Convert scalar state to one-hot encoded vector.

    Parameters
    ----------
    state : int
        Scalar state.
    state_size : int, optional
        Total number of states, by default 16

    Returns
    -------
    np.ndarray
        One-hot encoded state vector.
    """
    state_vec = np.zeros(state_size, dtype=np.float32)
    state_vec[state] = 1.0
    return state_vec


def train_agent(gamma: float, episodes: int, is_slippery: bool):
    """
    Train the DQN agent with a specific discount factor.

    Parameters
    ----------
    gamma : float
        Discount factor.
    episodes : int
        Number of training episodes.
    is_slippery : bool
        Whether environment is slippery.

    Returns
    -------
    list
        Rewards collected per episode.
    """
    env = make_frozenlake_env(is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, gamma)

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = one_hot_state(state, state_size)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = one_hot_state(next_state, state_size)
            agent.memorize(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        # Log progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Gamma: {gamma:.2f} | Episode: {episode + 1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    return episode_rewards


def main():
    config = load_config()
    
    # Extract parameters from config file
    gammas = config.get("training", {}).get("gammas", [0.1, 0.99])
    episodes = config.get("training", {}).get("episodes", 20)
    is_slippery = config.get("environment", {}).get("is_slippery", True)

    all_rewards = {}

    for gamma in gammas:
        print(f"Starting training for gamma = {gamma}")
        rewards = train_agent(gamma, episodes, is_slippery)
        all_rewards[gamma] = rewards

        # Save rewards to file
        np.save(os.path.join(LOGS_DIR, f"rewards_gamma_{gamma:.2f}.npy"), rewards)

        # Plot rewards per gamma
        plot_rewards(rewards, gamma, os.path.join(PLOTS_DIR, f"reward_gamma_{gamma:.2f}.png"))

    # Plot comparison of all gammas
    plot_gamma_comparison(all_rewards, os.path.join(PLOTS_DIR, "gamma_comparison.png"))


if __name__ == "__main__":
    main()
