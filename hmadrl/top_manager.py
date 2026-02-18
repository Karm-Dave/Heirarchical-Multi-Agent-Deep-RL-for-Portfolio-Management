"""Top-layer portfolio manager implementations (RL or MoE-router)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .rl_core import DQNPolicy
from .spaces import TopLevelAction, TopLevelState, generate_top_level_templates


class TopManagerBase(ABC):
    domain_names: list[str]

    @abstractmethod
    def act(self, state: TopLevelState, stochastic: bool = True) -> TopLevelAction:
        raise NotImplementedError


class RLTopManager(TopManagerBase):
    """Top-layer DQN that allocates across domains and chooses hold duration."""

    def __init__(
        self,
        domain_names: Sequence[str],
        max_hold_steps: int,
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
    ) -> None:
        self.domain_names = list(domain_names)
        self.templates = generate_top_level_templates(self.domain_names, max_hold_steps)
        self._template_index = {action.action_id: i for i, action in enumerate(self.templates)}
        state_dim = TopLevelState.vector_size(len(self.domain_names))

        self.policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=len(self.templates),
            hidden_dims=hidden_dims,
            seed=seed,
        )

    def act(self, state: TopLevelState, stochastic: bool = True) -> TopLevelAction:
        vector = np.asarray(state.to_vector(self.domain_names), dtype=np.float32)
        index = self.policy.select_action(vector, stochastic=stochastic)
        return self.templates[index]

    def remember(
        self,
        state: TopLevelState,
        action: TopLevelAction,
        reward: float,
        next_state: TopLevelState,
        done: bool,
    ) -> None:
        index = self._template_index[action.action_id]
        self.policy.store(
            state=state.to_vector(self.domain_names),
            action=index,
            reward=reward,
            next_state=next_state.to_vector(self.domain_names),
            done=done,
        )

    def learn(self) -> float | None:
        return self.policy.train_step()


class RouterNetwork(nn.Module):
    def __init__(self, state_dim: int, num_experts: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.model(state_batch)


class MoERouterTopManager(TopManagerBase):
    """MoE-style router: a router selects one expert allocator."""

    def __init__(
        self,
        domain_names: Sequence[str],
        max_hold_steps: int,
        seed: int = 0,
        train_router: bool = False,
    ) -> None:
        self.domain_names = list(domain_names)
        self.max_hold_steps = max_hold_steps
        self.rng = random.Random(seed)
        self.train_router = train_router

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = TopLevelState.vector_size(len(self.domain_names))
        self.router = RouterNetwork(state_dim=state_dim, num_experts=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=1e-3)

        self._last_logprob: torch.Tensor | None = None
        self.experts: list[Callable[[TopLevelState], tuple[dict[str, float], int]]] = [
            self._momentum_expert,
            self._low_vol_expert,
            self._liquidity_expert,
        ]

    def act(self, state: TopLevelState, stochastic: bool = True) -> TopLevelAction:
        state_vec = torch.as_tensor(
            state.to_vector(self.domain_names),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.router(state_vec).squeeze(0)
        probs = torch.softmax(logits, dim=0)

        if stochastic:
            expert_index = int(torch.multinomial(probs, num_samples=1).item())
        else:
            expert_index = int(torch.argmax(probs).item())

        weights, hold_steps = self.experts[expert_index](state)
        hold_steps = max(1, min(hold_steps, self.max_hold_steps))
        self._last_logprob = torch.log(probs[expert_index] + 1e-8)

        return TopLevelAction(
            domain_weights=weights,
            hold_steps=hold_steps,
            action_id=f"moe_router_exp{expert_index}",
        ).normalized()

    def update_router(self, reward: float) -> None:
        if not self.train_router or self._last_logprob is None:
            return
        loss = -float(reward) * self._last_logprob
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._last_logprob = None

    def _momentum_expert(self, state: TopLevelState) -> tuple[dict[str, float], int]:
        raw = np.array([state.domain_momentum.get(d, 0.0) for d in self.domain_names], dtype=float)
        shifted = raw - raw.min() + 1e-4
        probs = shifted / shifted.sum() if shifted.sum() > 0 else np.ones_like(shifted) / len(shifted)
        weights = {d: float(probs[i]) for i, d in enumerate(self.domain_names)}
        hold = min(self.max_hold_steps, max(2, self.max_hold_steps))
        return weights, hold

    def _low_vol_expert(self, state: TopLevelState) -> tuple[dict[str, float], int]:
        vol = np.array([state.domain_volatility.get(d, 1.0) for d in self.domain_names], dtype=float)
        inv = 1.0 / np.clip(vol, 1e-4, None)
        probs = inv / inv.sum()
        weights = {d: float(probs[i]) for i, d in enumerate(self.domain_names)}
        hold = max(1, self.max_hold_steps // 2)
        return weights, hold

    def _liquidity_expert(self, state: TopLevelState) -> tuple[dict[str, float], int]:
        liq = np.array([state.domain_liquidity.get(d, 0.0) for d in self.domain_names], dtype=float)
        alloc = np.array([state.current_allocation.get(d, 0.0) for d in self.domain_names], dtype=float)
        score = np.maximum(0.0, liq + 0.1 * alloc)
        if score.sum() <= 0:
            score = np.ones_like(score)
        probs = score / score.sum()
        weights = {d: float(probs[i]) for i, d in enumerate(self.domain_names)}
        hold = max(1, min(self.max_hold_steps, state.remaining_horizon_steps // 3 or 1))
        return weights, hold

