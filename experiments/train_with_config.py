import sys
import os
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agents import create_agent
from src.environments.frozenlake_env import make_frozenlake_env
from src.utils.utils import one_hot_state

def load_config(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(ROOT_DIR, "config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_with_config(config_path: str = None):
    config = load_config(config_path)
    
    env_config = config["environment"]
    agent_config = config["agent"]
    training_config = config["training"]
    
    env = make_frozenlake_env(
        is_slippery=env_config["is_slippery"],
        reward_schedule=tuple(env_config["reward_schedule"])
    )
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent_type = agent_config.get("type", "dqn")
    
    np.random.seed(42)
    torch.manual_seed(42)
    env.action_space.seed(42)
    
    agent_kwargs = {
        "lr": agent_config["learning_rate"],
        "epsilon_start": agent_config["epsilon_start"],
        "epsilon_end": agent_config["epsilon_end"],
        "epsilon_decay": agent_config["epsilon_decay"],
    }
    
    if agent_type.lower() == "dqn":
        agent_kwargs.update({
            "batch_size": agent_config["batch_size"],
            "device": training_config["device"]
        })
    
    agent = create_agent(
        agent_type=agent_type,
        state_size=state_size,
        action_size=action_size,
        gamma=training_config["gammas"][0],
        seed=42,
        **agent_kwargs
    )
    
    writer = SummaryWriter(log_dir=os.path.join(ROOT_DIR, "results", "config_training"))
    episode_rewards = []
    reward_window = deque(maxlen=100)
    
    for episode in range(1, training_config["episodes"] + 1):
        state, _ = env.reset(seed=42 if episode == 1 else None)
        
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
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        
        episode_rewards.append(total_reward)
        reward_window.append(total_reward)
        moving_avg = np.mean(reward_window)
        
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Moving_Avg_Reward", moving_avg, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)
        
        if episode % 50 == 0:
            print(f"Episode {episode}/{training_config['episodes']} | Avg Reward: {moving_avg:.3f}")
    
    writer.close()
    env.close()
    print(f"\nTraining complete. Average reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")

if __name__ == "__main__":
    train_with_config()

