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
from src.visualization.visualize_adaptive_gamma_results_4x4 import plot_dual_axis

# =============================================================================
# Configuration
# =============================================================================
MAP_NAME = "4x4"

STUDY_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study", MAP_NAME)
MODELS_DIR = os.path.join(STUDY_DIR, "models")
DATA_DIR = os.path.join(STUDY_DIR, "data") 
TB_DIR = os.path.join(STUDY_DIR, "tensorboard") 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

REWARD_SCHEDULE = (1, -1, -0.01)

PARAMS = {
    "learning_rate": 0.001, 
    "batch_size": 64, 
    "epsilon_decay": 0.995,    
    "is_slippery": True, 
    "episodes": 3000 if MAP_NAME == "8x8" else 2000, 
    "seed": 25
}

def get_gamma(episode, total_episodes, method="fixed"):
    """
    Calculates gamma based on the current episode.
    """
    if method == "fixed":
        return 0.99
    
    elif method == "adaptive":
        # Linear schedule: Start at 0.8, reach 0.99 at 50% of training
        gamma_start = 0.8
        gamma_end = 0.99
        fraction = 0.5 # Reach max gamma by episode 1000
        
        schedule_duration = total_episodes * fraction
        
        if episode >= schedule_duration:
            return gamma_end
        
        # Linear interpolation formula
        progress = episode / schedule_duration
        return gamma_start + (gamma_end - gamma_start) * progress
    
    return 0.99

def train_agent(strategy_name: str, agent_type: str = "dqn"):
    print(f"\n Starting Experiment: {strategy_name} on {MAP_NAME} | Agent: {agent_type.upper()}")

    run_name = f"{agent_type}_strategy_{strategy_name}_{MAP_NAME}"
    writer = SummaryWriter(log_dir=os.path.join(TB_DIR, run_name))

    env = make_frozenlake_env(
        is_slippery=PARAMS["is_slippery"],
        map_name=MAP_NAME,
        reward_schedule=REWARD_SCHEDULE
    )
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    seed = PARAMS["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    
    agent_kwargs = {
        "lr": PARAMS["learning_rate"],
        "epsilon_decay": PARAMS["epsilon_decay"],
        "epsilon_start": PARAMS.get("epsilon_start", 1.0),
        "epsilon_end": PARAMS.get("epsilon_end", 0.01),
    }
    
    if agent_type.lower() == "dqn":
        agent_kwargs.update({
            "batch_size": PARAMS["batch_size"],
            "device": PARAMS.get("device", "cpu")
        })

    agent = create_agent(
        agent_type=agent_type,
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        seed=seed,
        **agent_kwargs
    )

    # Logging Containers
    history = {
        "rewards": [], 
        "gammas": [], 
        "loss": [], 
        "q_values": [], 
        "steps": []
    }
    
    # 2. Training Loop
    for episode in range(1, PARAMS["episodes"] + 1):
        
        # --- SCIENTIFIC CORE: DYNAMIC GAMMA UPDATE ---
        # We calculate the gamma for this specific moment in time
        current_gamma = get_gamma(episode, PARAMS["episodes"], strategy_name)
        
        # We inject it directly into the agent. 
        # The agent's 'replay' function uses self.gamma, so this update takes effect immediately.
        agent.gamma = current_gamma 
        # ---------------------------------------------

        state, _ = env.reset(seed=seed if episode == 1 else None)
        
        if agent_type.lower() == "dqn":
            state = one_hot_state(state, state_size)
        else:
            state_idx = state
        
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_qs = []
        done = False

        while not done:
            if agent_type.lower() == "dqn":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_vals = agent.policy_net(state_tensor)
                    episode_qs.append(q_vals.max().item())
                action = agent.act(state)
            else:
                q_vals = agent.q_table[state_idx]
                episode_qs.append(np.max(q_vals))
                action = agent.act(state_idx)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if agent_type.lower() == "dqn":
                next_state_vec = one_hot_state(next_state, state_size)
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
            
        # Calculate Metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_qs) if episode_qs else 0
        
        # Logging
        history["rewards"].append(total_reward)
        history["gammas"].append(current_gamma)
        history["loss"].append(avg_loss)
        history["q_values"].append(avg_q)
        history["steps"].append(steps)
        
        # TensorBoard
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Gamma", current_gamma, episode)
        writer.add_scalar("Loss", avg_loss, episode)
        writer.add_scalar("Avg_Max_Q", avg_q, episode)
        writer.add_scalar("Steps", steps, episode)
        
        if episode % 100 == 0:
            avg_rew = np.mean(history["rewards"][-50:])
            print(f"   > Ep {episode}: Reward={avg_rew:.3f} | Gamma={current_gamma:.3f} | Q={avg_q:.3f}")

    writer.close()
    
    np.save(os.path.join(DATA_DIR, f"results_{agent_type}_{strategy_name}_{MAP_NAME}.npy"), history)
    
    if agent_type.lower() == "dqn":
        model_path = os.path.join(MODELS_DIR, f"{agent_type}_{strategy_name}_{MAP_NAME}.pth")
        torch.save(agent.policy_net.state_dict(), model_path)
    else:
        model_path = os.path.join(MODELS_DIR, f"{agent_type}_{strategy_name}_{MAP_NAME}.npy")
        np.save(model_path, agent.q_table)
    
    print(f" Completed {agent_type.upper()} | {strategy_name} on {MAP_NAME}")
    print(f" Model saved to: {model_path}")

if __name__ == "__main__":
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    agent_type = "dqn"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agent_type = config.get("agent", {}).get("type", agent_type)
    
    train_agent("fixed", agent_type)
    train_agent("adaptive", agent_type)
    
    print("\n" + "=" * 40 + f"\nGENERATING VISUALIZATIONS | {agent_type.upper()} | {MAP_NAME}\n" + "=" * 40)
    plot_dual_axis("rewards", "Avg Reward (Win 50)", f"{agent_type}_adaptive_rewards_{MAP_NAME}_pro.png", 
                   "Impact of Adaptive Gamma on Learning Speed", agent_type)
    
    plot_dual_axis("q_values", "Avg Max Q-Value", f"{agent_type}_adaptive_q_values_{MAP_NAME}_pro.png", 
                   "Growth of Value Estimates (Q) with Gamma", agent_type)
    
    print(f"\n All visualizations complete for {agent_type.upper()} | {MAP_NAME}!")