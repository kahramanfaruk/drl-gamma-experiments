import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    """
    Simple feed-forward network for Deep Q-Learning.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (state size)
    output_dim : int
        Dimension of the output (action size)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    """
    Deep Q-Learning agent.

    Parameters
    ----------
    state_size : int
        Size of the state space.
    action_size : int
        Size of the action space.
    gamma : float
        Discount factor.
    lr : float, optional
        Learning rate, by default 0.001
    batch_size : int, optional
        Mini-batch size for training, by default 64
    buffer_size : int, optional
        Replay buffer size, by default 10000
    epsilon_start : float, optional
        Initial epsilon for epsilon-greedy policy, by default 1.0
    epsilon_end : float, optional
        Minimum epsilon, by default 0.01
    epsilon_decay : float, optional
        Decay rate of epsilon per step, by default 0.995
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        seed: int,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=buffer_size)

        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        random.seed(seed)

        self.learn_step_counter = 0
        self.target_update_freq = 1000  # Steps to update target net

    def memorize(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next state.
        done : bool
            Whether episode terminated.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        int
            Action selected.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def replay(self):
        """
        Train the policy network with a batch from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-values for current states
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q-values for next states
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # Update target network periodically
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)


        return loss.item()