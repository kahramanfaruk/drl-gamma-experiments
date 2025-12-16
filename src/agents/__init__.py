from src.agents.dqn_agent import DQNAgent
from src.agents.qlearning_agent import QLearningAgent


def create_agent(agent_type: str, state_size: int, action_size: int, gamma: float, seed: int, **kwargs):
    if agent_type.lower() == "dqn":
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            gamma=gamma,
            seed=seed,
            **kwargs
        )
    elif agent_type.lower() == "qlearning":
        device = kwargs.pop("device", "cpu")
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            gamma=gamma,
            seed=seed,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'dqn' or 'qlearning'")

