# grid_search.py
import os
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch
from src.agents.dqn_agent import DQNAgent
from src.environments.frozenlake_env import make_frozenlake_env
from src.visualization.plot_results import plot_rewards
from datetime import datetime
from experiments.run_experiments import one_hot_state


RESULTS_DIR = "results/grid_search"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


def train_dqn(
    gamma: float,
    learning_rate: float,
    batch_size: int,
    epsilon_decay: float,
    is_slippery: bool,
    episodes: int = 300,
    device: str = "cpu"
):
    """
    Train a DQN agent with given hyperparameters and return average reward.
    """

    env = make_frozenlake_env(is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=gamma,
        lr=learning_rate,
        batch_size=batch_size,
        epsilon_decay=epsilon_decay,
        device=device
    )

    writer_name = f"gamma_{gamma}_lr_{learning_rate}_bs_{batch_size}_ed_{epsilon_decay}_slip_{is_slippery}"
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, writer_name))

    episode_rewards = []
    reward_window = deque(maxlen=100)

    for episode in range(1, episodes + 1):
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
        reward_window.append(total_reward)
        moving_avg_reward = np.mean(reward_window)

        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Moving_Avg_Reward", moving_avg_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        if episode % 50 == 0:
            print(
                f"[γ={gamma}, lr={learning_rate}, bs={batch_size}, decay={epsilon_decay}, slip={is_slippery}] "
                f"Ep {episode}/{episodes} | Reward={total_reward:.2f} | Avg100={moving_avg_reward:.3f}"
            )

    env.close()
    writer.close()

    # Compute average reward over last 100 episodes
    avg_final_reward = np.mean(episode_rewards[-100:])
    return episode_rewards, avg_final_reward


def run_grid_search():
    """
    Run grid search over multiple hyperparameter combinations and log results.
    """

    # === Define search space ===
    param_grid = {
        "gamma": [0.8, 0.9, 0.99],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [64, 128],
        "epsilon_decay": [0.995, 0.99],
        "is_slippery": [False],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n🔍 Running grid search over {len(all_combinations)} configurations...\n")

    results_summary = []

    for params in all_combinations:
        print(f"\n🚀 Training with config: {params}")
        rewards, avg_reward = train_dqn(**params)
        config_name = "_".join([f"{k}_{v}" for k, v in params.items()])

        np.save(os.path.join(LOGS_DIR, f"rewards_{config_name}.npy"), rewards)
        plot_rewards(rewards, params["gamma"], os.path.join(PLOTS_DIR, f"reward_{config_name}.png"))

        results_summary.append({
            **params,
            "avg_final_reward": avg_reward
        })

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"grid_search_summary_{timestamp}.txt")

    with open(summary_path, "w") as f:
        f.write("DQN Grid Search Results\n")
        f.write("========================\n\n")
        for r in results_summary:
            f.write(str(r) + "\n")

    print(f"\n✅ Grid search complete! Results saved to {summary_path}\n")
    best_config = max(results_summary, key=lambda x: x["avg_final_reward"])
    print(f"🏆 Best Configuration: {best_config}\n")


if __name__ == "__main__":
    run_grid_search()
