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
import yaml
from src.agents import create_agent
from src.environments.frozenlake_env import make_frozenlake_env
from src.visualization.plot_results import plot_rewards
from datetime import datetime
from src.utils.utils import one_hot_state


REWARD_SCHEDULE = (1, -1, -0.01)

def get_results_dirs(is_slippery: bool):
    env_type = "stochastic" if is_slippery else "deterministic"
    results_dir = os.path.join(ROOT_DIR, "results", f"grid_search_discrete_{env_type}")
    logs_dir = os.path.join(results_dir, "logs")
    plots_dir = os.path.join(results_dir, "plots")
    tensorboard_dir = os.path.join(results_dir, "tensorboard")
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    return results_dir, logs_dir, plots_dir, tensorboard_dir


def train_agent(
        agent_type: str,
        gamma: float,
        learning_rate: float,
        epsilon_decay: float,
        is_slippery: bool,
        seed: int, 
        reward_schedule: tuple = (1, -1, -0.01),
        episodes: int = 1000,
        device: str = "cpu",
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        batch_size: int = None,
        results_dir: str = None,
        logs_dir: str = None,
        plots_dir: str = None,
        tensorboard_dir: str = None
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = make_frozenlake_env(is_slippery=is_slippery,
                              reward_schedule=reward_schedule)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent_kwargs = {
        "lr": learning_rate,
        "epsilon_decay": epsilon_decay,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
    }
    
    if agent_type.lower() == "dqn":
        if batch_size is None:
            raise ValueError("batch_size is required for DQN agent")
        agent_kwargs.update({
            "batch_size": batch_size,
            "device": device
        })
    
    agent = create_agent(
        agent_type=agent_type,
        state_size=state_size,
        action_size=action_size,
        gamma=gamma,
        seed=seed,
        **agent_kwargs
    )

    if results_dir is None:
        results_dir, logs_dir, plots_dir, tensorboard_dir = get_results_dirs(is_slippery)

    batch_str = f"_bs_{batch_size}" if batch_size is not None else ""
    writer_name = f"{agent_type}_gamma_{gamma}_lr_{learning_rate}{batch_str}_ed_{epsilon_decay}_slip_{is_slippery}_seed_{seed}"
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, writer_name))

    episode_rewards = []
    reward_window = deque(maxlen=100)

    # Seed the environment once before the loop
    state, _ = env.reset(seed=seed)

    for episode in range(1, episodes + 1):
        if episode > 1:  # On subsequent episodes, just reset
            state, _ = env.reset()

        if agent_type.lower() == "dqn":
            state = one_hot_state(state, state_size)
        else:
            state_idx = state
        total_reward = 0
        done = False

        while not done:
            if agent_type.lower() == "dqn":
                action = agent.act(state)
            else:
                action = agent.act(state_idx)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if agent_type.lower() == "dqn":
                next_state = one_hot_state(next_state, state_size)
                agent.memorize(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
            else:
                agent.memorize(state_idx, action, reward, next_state, done)
                state_idx = next_state

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
    config_items = {k: v for k, v in sorted(params.items()) if k != "reward_schedule"}
    config_name = "_".join([f"{k}_{v}" for k, v in config_items.items()])
    config_name = config_name.replace(" ", "_").replace(".", "_").replace("(", "").replace(")", "").replace(",", "")
    if len(config_name) > 200:
        config_name = config_name[:200]
    # === End Metadata collection ===

    print(f"\n[PID: {pid} | Start: {start_time.strftime('%H:%M:%S')}] Starting config: {params}")

    rewards, avg_reward = train_agent(**params)

    # === More Metadata ===
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    # === End Metadata ===

    # Save individual run artifacts
    try:
        _, logs_dir, plots_dir, _ = get_results_dirs(params["is_slippery"])
        
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir, exist_ok=True)
        
        log_path = os.path.join(logs_dir, f"rewards_{config_name}.npy")
        plot_path = os.path.join(plots_dir, f"reward_{config_name}.png")
        
        np.save(log_path, rewards)
        plot_rewards(rewards, params["gamma"], plot_path)
        print(f"   Saved: {os.path.basename(log_path)} and {os.path.basename(plot_path)}")
    except Exception as e:
        import traceback
        print(f"Error saving artifacts for {config_name}: {e}")
        traceback.print_exc()

    print(
        f"[PID: {pid}] Finished config (Seed {params.get('seed')}) | Avg Reward: {avg_reward:.3f} | Duration: {duration:.2f}s")

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


def run_grid_search(agent_type: str = "dqn", is_slippery: bool = None):
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agent_type = config.get("agent", {}).get("type", agent_type)
            epsilon_start = config.get("agent", {}).get("epsilon_start", 1.0)
            epsilon_end = config.get("agent", {}).get("epsilon_end", 0.01)
            if is_slippery is None:
                is_slippery = config.get("environment", {}).get("is_slippery", False)
    else:
        epsilon_start = 1.0
        epsilon_end = 0.01
        if is_slippery is None:
            is_slippery = False

    param_grid = {
        "agent_type": [agent_type],
        "gamma": [0.8, 0.85, 0.9, 0.95, 0.99],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [64, 128],
        "epsilon_decay": [0.95, 0.99],
        "is_slippery": [is_slippery],
        "reward_schedule": [REWARD_SCHEDULE],
        "epsilon_start": [epsilon_start],
        "epsilon_end": [epsilon_end]
    }
    
    if agent_type.lower() == "qlearning":
        param_grid.pop("batch_size")

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # === Add unique seed to each combination ===
    all_combinations = [
        {**params, "seed": i} for i, params in enumerate(param_combinations)
    ]
    # === End seed addition ===

    print(f"\nRunning grid search over {len(all_combinations)} configurations in parallel...\n")

    # === Run experiments in parallel ===
    with multiprocessing.Pool() as pool:
        results_summary = pool.map(train_dqn_wrapper, all_combinations)

    print("\n\n" + "=" * 30)
    print("All training runs complete. Compiling summary...")
    print("=" * 30 + "\n")

    # === Save summary ===
    results_df = pd.DataFrame(results_summary)

    # Sort by reward to see the best configurations at the top
    results_df = results_df.sort_values(by="avg_final_reward", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    
    results_dir, _, _, _ = get_results_dirs(is_slippery)

    # --- Save as CSV ---
    csv_path = os.path.join(results_dir, f"grid_search_summary_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results summary saved to: {csv_path}")

    # --- Save as JSON ---
    json_path = os.path.join(results_dir, f"grid_search_summary_{timestamp}.json")
    results_df.to_json(json_path, orient="records", indent=2)
    print(f"JSON summary saved to: {json_path}")

    # --- Print best config ---
    best_config = results_df.iloc[0].to_dict()
    print("\nBest Configuration:")
    # Pretty print the best config dict
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    run_grid_search()