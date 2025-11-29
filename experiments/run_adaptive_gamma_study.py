import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add project root to path to allow imports from src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agents.dqn_agent import DQNAgent
from src.environments.frozenlake_env import make_frozenlake_env
from src.utils.utils import one_hot_state

# =============================================================================
# Configuration
# =============================================================================
STUDY_DIR = os.path.join(ROOT_DIR, "results", "adaptive_gamma_study")
MODELS_DIR = os.path.join(STUDY_DIR, "models")
DATA_DIR = os.path.join(STUDY_DIR, "data") 
TB_DIR = os.path.join(STUDY_DIR, "tensorboard") 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

REWARD_SCHEDULE = (1, -1, -0.01)

# We use the Stochastic environment as it highlights instability best
PARAMS = {
    "learning_rate": 0.001, 
    "batch_size": 64, 
    "epsilon_decay": 0.995,    
    "is_slippery": True, 
    "episodes": 2000, 
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

def train_agent(strategy_name: str):
    print(f"\n Starting Experiment: {strategy_name}")

    # 1. Setup
    run_name = f"strategy_{strategy_name}"
    writer = SummaryWriter(log_dir=os.path.join(TB_DIR, run_name))

    env = make_frozenlake_env(
        is_slippery=PARAMS["is_slippery"], 
        reward_schedule=REWARD_SCHEDULE
    )
    
    # Fixed seed for fair comparison (Ceteris Paribus)
    seed = PARAMS["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    
    # Initialize Agent (Gamma 0.99 is placeholder, overwritten in loop)
    agent = DQNAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        gamma=0.99, 
        lr=PARAMS["learning_rate"],
        batch_size=PARAMS["batch_size"],
        epsilon_decay=PARAMS["epsilon_decay"],
        seed=seed
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
        state = one_hot_state(state, 16)
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_qs = []
        done = False

        while not done:
            # Capture Q-Values (Confidence Check)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.policy_net(state_tensor)
                episode_qs.append(q_vals.max().item())

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_vec = one_hot_state(next_state, 16)

            agent.memorize(state, action, reward, next_state_vec, done)
            
            # Capture Loss (Stability Check)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state_vec
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
    
    # Save Full Data Structure
    np.save(os.path.join(DATA_DIR, f"results_{strategy_name}.npy"), history)
    print(f" Completed {strategy_name}")

if __name__ == "__main__":
    # 1. Run the Baseline (Standard Fixed Gamma)
    train_agent("fixed")
    
    # 2. Run the Scientific Proposal (Adaptive Gamma)
    train_agent("adaptive")