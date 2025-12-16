import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agents.dqn_agent import DQNAgent
from src.environments.frozenlake_env import make_frozenlake_env
from src.utils.utils import one_hot_state

# =============================================================================
# Configuration
# =============================================================================
# CHANGE 1: Define Map Name Here
MAP_NAME = "8x8"  # Options: "4x4", "8x8"

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
    # CHANGE 2: 8x8 is harder, so we might need more episodes (optional but recommended)
    "episodes": 3000 if MAP_NAME == "8x8" else 2000, 
    "seed": 25
}

def get_gamma(episode, total_episodes, method="fixed"):
    if method == "fixed":
        return 0.99
    elif method == "adaptive":
        gamma_start = 0.8
        gamma_end = 0.99
        fraction = 0.5 
        schedule_duration = total_episodes * fraction
        
        if episode >= schedule_duration:
            return gamma_end
        
        progress = episode / schedule_duration
        return gamma_start + (gamma_end - gamma_start) * progress
    return 0.99

def train_agent(strategy_name: str):
    print(f"\n Starting Experiment: {strategy_name} on {MAP_NAME}")

    run_name = f"strategy_{strategy_name}_{MAP_NAME}"
    writer = SummaryWriter(log_dir=os.path.join(TB_DIR, run_name))

    # CHANGE 3: Pass map_name to the environment factory
    env = make_frozenlake_env(
        is_slippery=PARAMS["is_slippery"], 
        map_name=MAP_NAME,
        reward_schedule=REWARD_SCHEDULE
    )
    
    # CHANGE 4: Dynamically get state size (16 for 4x4, 64 for 8x8)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    seed = PARAMS["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99, 
        lr=PARAMS["learning_rate"],
        batch_size=PARAMS["batch_size"],
        epsilon_decay=PARAMS["epsilon_decay"],
        seed=seed
    )

    history = {"rewards": [], "gammas": [], "loss": [], "q_values": [], "steps": []}
    
    for episode in range(1, PARAMS["episodes"] + 1):
        current_gamma = get_gamma(episode, PARAMS["episodes"], strategy_name)
        agent.gamma = current_gamma 

        state, _ = env.reset(seed=seed if episode == 1 else None)
        
        # CHANGE 5: Use dynamic state_size instead of hardcoded 16
        state = one_hot_state(state, state_size)
        
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_qs = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.policy_net(state_tensor)
                episode_qs.append(q_vals.max().item())

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # CHANGE 6: Dynamic state size here too
            next_state_vec = one_hot_state(next_state, state_size)

            agent.memorize(state, action, reward, next_state_vec, done)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state_vec
            total_reward += reward
            steps += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)
            
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_qs) if episode_qs else 0
        
        history["rewards"].append(total_reward)
        history["gammas"].append(current_gamma)
        history["loss"].append(avg_loss)
        history["q_values"].append(avg_q)
        history["steps"].append(steps)
        
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Gamma", current_gamma, episode)
        writer.add_scalar("Loss", avg_loss, episode)
        writer.add_scalar("Avg_Max_Q", avg_q, episode)
        
        if episode % 100 == 0:
            avg_rew = np.mean(history["rewards"][-50:])
            print(f"   > Ep {episode}: Reward={avg_rew:.3f} | Gamma={current_gamma:.3f} | Steps={steps}")

    writer.close()
    
    # CHANGE 7: Include map name in filename to prevent overwriting
    np.save(os.path.join(DATA_DIR, f"results_{strategy_name}_{MAP_NAME}.npy"), history)
    
    # Save model specifically for this map size
    torch.save(agent.policy_net.state_dict(), os.path.join(MODELS_DIR, f"dqn_{strategy_name}_{MAP_NAME}.pth"))
    
    print(f" Completed {strategy_name} on {MAP_NAME}")

if __name__ == "__main__":
    train_agent("fixed")
    train_agent("adaptive")