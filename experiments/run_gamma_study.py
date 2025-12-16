import sys
import os
import torch
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agents import create_agent
from src.environments.frozenlake_env import make_frozenlake_env
from src.utils.utils import one_hot_state
from src.visualization.visualize_metrics import run_visualizations
from src.visualization.visualize_policy import generate_all_maps

# =============================================================================
# Configuration & Constants
# =============================================================================
# Main Study Directory
STUDY_DIR = os.path.join(ROOT_DIR, "results", "gamma_study_analysis")

# Reward Schedule: (Goal, Hole, Step)
REWARD_SCHEDULE = (1, -1, -0.01)

BEST_PARAMS_DETERMINISTIC = {
    "learning_rate": 0.0005,
    "batch_size": 64,
    "epsilon_decay": 0.95,
    "is_slippery": False,
    "episodes": 1000,
    "seed": 12
}

BEST_PARAMS_STOCHASTIC = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epsilon_decay": 0.995,
    "is_slippery": True,
    "episodes": 2000,
    "seed": 25
}

GAMMAS_TO_TEST = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def train_single_gamma(gamma: float, base_params: dict, env_type: str, agent_type: str = "dqn"):
    print(f"\n Testing Gamma = {gamma} [{env_type}] | Agent: {agent_type.upper()}...")

    type_dir = os.path.join(STUDY_DIR, env_type)
    models_dir = os.path.join(type_dir, "models")
    data_dir = os.path.join(type_dir, "data")
    tb_dir = os.path.join(type_dir, "tensorboard")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    run_name = f"{agent_type}_gamma_{gamma}"
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))

    env = make_frozenlake_env(
        is_slippery=base_params["is_slippery"],
        reward_schedule=REWARD_SCHEDULE
    )

    seed = base_params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)

    agent_kwargs = {
        "lr": base_params["learning_rate"],
        "epsilon_decay": base_params["epsilon_decay"],
        "epsilon_start": base_params.get("epsilon_start", 1.0),
        "epsilon_end": base_params.get("epsilon_end", 0.01),
    }
    
    if agent_type.lower() == "dqn":
        agent_kwargs.update({
            "batch_size": base_params["batch_size"],
            "device": base_params.get("device", "cpu")
        })

    agent = create_agent(
        agent_type=agent_type,
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        gamma=gamma,
        seed=seed,
        **agent_kwargs
    )

    history = {
        "rewards": [],
        "steps": [],
        "avg_q": [],
        "loss": [],
        "success": []
    }

    for episode in range(1, base_params["episodes"] + 1):
        state, _ = env.reset(seed=seed if episode == 1 else None)
        
        if agent_type.lower() == "dqn":
            state = one_hot_state(state, 16)
        else:
            state_idx = state
        
        total_reward = 0
        steps = 0
        episode_q_vals = []
        episode_losses = []
        done = False
        terminated = False

        while not done:
            if agent_type.lower() == "dqn":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_vals = agent.policy_net(state_tensor)
                    episode_q_vals.append(q_vals.max().item())
                action = agent.act(state)
            else:
                q_vals = agent.q_table[state_idx]
                episode_q_vals.append(np.max(q_vals))
                action = agent.act(state_idx)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if agent_type.lower() == "dqn":
                next_state_vec = one_hot_state(next_state, 16)
                agent.memorize(state, action, reward, next_state_vec, done)
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
                state = next_state_vec
            else:
                agent.memorize(state_idx, action, reward, next_state, done)
                state_idx = next_state

            total_reward += reward
            steps += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        avg_q_episode = np.mean(episode_q_vals) if episode_q_vals else 0
        avg_loss_episode = np.mean(episode_losses) if episode_losses else 0
        is_success = 1 if (terminated and reward > 0) else 0

        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["avg_q"].append(avg_q_episode)
        history["loss"].append(avg_loss_episode)
        history["success"].append(is_success)

        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Steps", steps, episode)
        writer.add_scalar("Avg_Max_Q", avg_q_episode, episode)
        writer.add_scalar("Loss", avg_loss_episode, episode)
        writer.add_scalar("Success_Rate", is_success, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        if episode % 200 == 0:
            avg_rew = np.mean(history["rewards"][-50:])
            print(f"   > Ep {episode}: Avg Reward={avg_rew:.3f} | Steps={steps} | Loss={avg_loss_episode:.4f}")

    writer.close()

    if agent_type.lower() == "dqn":
        model_path = os.path.join(models_dir, f"{agent_type}_gamma_{gamma}.pth")
        torch.save(agent.policy_net.state_dict(), model_path)
    else:
        model_path = os.path.join(models_dir, f"{agent_type}_gamma_{gamma}.npy")
        np.save(model_path, agent.q_table)

    data_path = os.path.join(data_dir, f"metrics_{agent_type}_gamma_{gamma}.npy")
    np.save(data_path, history)

    print(f" Data saved for {agent_type.upper()} | Gamma={gamma} in {env_type}/")


def run_study(agent_type: str = "dqn"):
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agent_type = config.get("agent", {}).get("type", agent_type)
    
    print("=" * 40 + f"\nSTARTING DETERMINISTIC STUDY | {agent_type.upper()}\n" + "=" * 40)
    for g in GAMMAS_TO_TEST:
        train_single_gamma(g, BEST_PARAMS_DETERMINISTIC, "deterministic", agent_type)

    print("\n" + "=" * 40 + f"\nSTARTING STOCHASTIC STUDY | {agent_type.upper()}\n" + "=" * 40)
    for g in GAMMAS_TO_TEST:
        train_single_gamma(g, BEST_PARAMS_STOCHASTIC, "stochastic", agent_type)

    print(f"\n Study Complete. Results saved to: {STUDY_DIR}")
    print(f"   > Run 'tensorboard --logdir={STUDY_DIR}' to view live results.")
    
    print("\n" + "=" * 40 + f"\nGENERATING VISUALIZATIONS | {agent_type.upper()}\n" + "=" * 40)
    print("\n Generating metric comparisons...")
    run_visualizations(agent_type)
    
    print("\n Generating policy maps...")
    generate_all_maps(agent_type)
    
    print(f"\n All visualizations complete for {agent_type.upper()}!")


if __name__ == "__main__":
    run_study()
