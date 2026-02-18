"""PyTorch RL utilities used by top and domain managers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(list(self._items), batch_size)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.model(state_batch)


class DQNPolicy:
    """Discrete-action DQN with softmax stochastic action selection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.02,
        batch_size: int = 32,
        replay_size: int = 5000,
        epsilon_start: float = 0.2,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.999,
        temperature: float = 1.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = max(1e-4, temperature)
        self.rng = random.Random(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_size, seed=seed)

    def select_action(self, state: np.ndarray, stochastic: bool = True) -> int:
        if stochastic and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).squeeze(0)

        if not stochastic:
            return int(torch.argmax(q_values).item())

        probs = torch.softmax(q_values / self.temperature, dim=0)
        action = torch.multinomial(probs, num_samples=1)
        return int(action.item())

    def store(
        self,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        transition = Transition(
            state=np.asarray(list(state), dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(list(next_state), dtype=np.float32),
            done=bool(done),
        )
        self.replay.push(transition)

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)

        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        done_mask = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + (1.0 - done_mask) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self._soft_update_target()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.copy_(self.tau * param + (1.0 - self.tau) * target_param)

