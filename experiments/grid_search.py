import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


import os
import itertools
import numpy as np
import pandas as pd
import json
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch
from src.agents.dqn_agent import DQNAgent
from src.environments.frozenlake_env import make_frozenlake_env
from src.visualization.plot_results import plot_rewards
from datetime import datetime
from src.utils.utils import one_hot_state


print(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results", "grid_search_discrete_deterministic")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")

REWARD_SCHEDULE = (1, -1, -0.01)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


def train_dqn(
        gamma: float,
        learning_rate: float,
        batch_size: int,
        epsilon_decay: float,
        is_slippery: bool,
        seed: int, 
        reward_schedule: tuple = (1, -1, -0.01),
        episodes: int = 1000,
        device: str = "cpu"
):
    """
    Train a DQN agent with given hyperparameters and return average reward.
    """

    # === Seeding ===
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # === End Seeding ===

    env = make_frozenlake_env(is_slippery=is_slippery,
                              reward_schedule=reward_schedule)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=gamma,
        lr=learning_rate,
        batch_size=batch_size,
        epsilon_decay=epsilon_decay,
        device=device,
        seed=seed
    )

    writer_name = f"gamma_{gamma}_lr_{learning_rate}_bs_{batch_size}_ed_{epsilon_decay}_slip_{is_slippery}_seed_{seed}"
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, writer_name))

    episode_rewards = []
    reward_window = deque(maxlen=100)

    # Seed the environment once before the loop
    state, _ = env.reset(seed=seed)

    for episode in range(1, episodes + 1):
        if episode > 1:  # On subsequent episodes, just reset
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

        # Decay epsilon once per episode for stochastic env
        # Without this change, your agent will stop exploring almost immediately and will not be able to 
        # solve the stochastic (is_slippery: True) environment.
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        episode_rewards.append(total_reward)
        reward_window.append(total_reward)
        moving_avg_reward = np.mean(reward_window)

        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Moving_Avg_Reward", moving_avg_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        if episode % 50 == 0:
            print(
                f"[PID {os.getpid()} | Seed {seed}] "
                f"Ep {episode}/{episodes} | Avg100={moving_avg_reward:.3f}"
            )

    env.close()
    writer.close()

    # Compute average reward over last 100 episodes
    avg_final_reward = np.mean(episode_rewards[-100:])
    return episode_rewards, avg_final_reward


def train_dqn_wrapper(params: dict):
    """
    Wrapper function for multiprocessing.
    Runs train_dqn with a dict of params, handles metadata, and saves results.
    """
    # === Metadata collection ===
    start_time = datetime.now()
    pid = os.getpid()
    config_name = "_".join([f"{k}_{v}" for k, v in params.items()])
    # === End Metadata collection ===

    print(f"\n🚀 [PID: {pid} | Start: {start_time.strftime('%H:%M:%S')}] Starting config: {params}")

    # Run the main training function
    rewards, avg_reward = train_dqn(**params)

    # === More Metadata ===
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    # === End Metadata ===

    # Save individual run artifacts
    try:
        np.save(os.path.join(LOGS_DIR, f"rewards_{config_name}.npy"), rewards)
        plot_rewards(rewards, params["gamma"], os.path.join(PLOTS_DIR, f"reward_{config_name}.png"))
    except Exception as e:
        print(f"Error saving artifacts for {config_name}: {e}")

    print(
        f"✅ [PID: {pid}] Finished config (Seed {params.get('seed')}) | Avg Reward: {avg_reward:.3f} | Duration: {duration:.2f}s")

    # Return data for the final summary
    return {
        "run_id": config_name,
        "pid": pid,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        **params,
        "avg_final_reward": avg_reward
    }


def run_grid_search():
    """
    Run grid search over multiple hyperparameter combinations in parallel
    and log results to structured formats (CSV, JSON).
    """

    # === Define search space ===
    param_grid = {
        "gamma": [0.8, 0.85, 0.9, 0.95, 0.99],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [64, 128],
        "epsilon_decay": [0.95, 0.99],
        "is_slippery": [False],
        "reward_schedule": [REWARD_SCHEDULE]
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # === Add unique seed to each combination ===
    all_combinations = [
        {**params, "seed": i} for i, params in enumerate(param_combinations)
    ]
    # === End seed addition ===

    print(f"\n🔍 Running grid search over {len(all_combinations)} configurations in parallel...\n")

    # === Run experiments in parallel ===
    with multiprocessing.Pool() as pool:
        results_summary = pool.map(train_dqn_wrapper, all_combinations)

    print("\n\n" + "=" * 30)
    print("✅ All training runs complete. Compiling summary...")
    print("=" * 30 + "\n")

    # === Save summary ===
    results_df = pd.DataFrame(results_summary)

    # Sort by reward to see the best configurations at the top
    results_df = results_df.sort_values(by="avg_final_reward", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d-%H_%M_%S")

    # --- Save as CSV ---
    csv_path = os.path.join(RESULTS_DIR, f"grid_search_summary_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"📊 Results summary saved to: {csv_path}")

    # --- Save as JSON ---
    json_path = os.path.join(RESULTS_DIR, f"grid_search_summary_{timestamp}.json")
    results_df.to_json(json_path, orient="records", indent=2)
    print(f"📄 JSON summary saved to: {json_path}")

    # --- Print best config ---
    best_config = results_df.iloc[0].to_dict()
    print("\n🏆 Best Configuration:")
    # Pretty print the best config dict
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    run_grid_search()