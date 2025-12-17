import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        seed: int,
        lr: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.device = device

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((state_size, action_size))
        random.seed(seed)
        np.random.seed(seed)

    def act(self, state: int) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)

    def memorize(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray):
            state = int(np.argmax(state))
        if isinstance(next_state, np.ndarray):
            next_state = int(np.argmax(next_state))
        self.update(state, action, reward, next_state, done)

    def replay(self):
        return None

