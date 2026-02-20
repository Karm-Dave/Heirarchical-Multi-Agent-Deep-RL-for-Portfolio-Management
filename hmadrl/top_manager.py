"""Top-layer portfolio manager implementations (RL or MoE-router)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .rl_core import PPOContinuousPolicy
from .spaces import TopLevelAction, TopLevelState


class TopManagerBase(ABC):
    domain_names: list[str]

    @abstractmethod
    def act(self, state: TopLevelState, stochastic: bool = True) -> TopLevelAction:
        raise NotImplementedError


class RLTopManager(TopManagerBase):
    """Top-layer continuous PPO allocator with optional ensembles."""

    def __init__(
        self,
        domain_names: Sequence[str],
        max_hold_steps: int,
        num_global_features: int = 0,
        min_hold_steps: int = 2,
        hidden_dims: Sequence[int] = (128, 128),
        ensemble_size: int = 1,
        learning_rate: float = 3e-4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 1e-3,
        ppo_epochs: int = 4,
        network_type: str = "mlp",
        transformer_d_model: int = 96,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        hold_inertia: float = 0.75,
        seed: int = 0,
    ) -> None:
        self.domain_names = list(domain_names)
        state_dim = TopLevelState.vector_size(len(self.domain_names), num_global_features=num_global_features)
        self._state_dim = state_dim
        self._ensemble_size = max(1, int(ensemble_size))

        self.policies = [
            PPOContinuousPolicy(
                state_dim=state_dim,
                simplex_dim=len(self.domain_names),
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                epochs=ppo_epochs,
                network_type=network_type,
                transformer_d_model=transformer_d_model,
                transformer_nhead=transformer_nhead,
                transformer_layers=transformer_layers,
                transformer_dropout=transformer_dropout,
                min_hold_steps=min_hold_steps,
                max_hold_steps=max_hold_steps,
                hold_inertia=hold_inertia,
                seed=seed + i * 17,
            )
            for i in range(self._ensemble_size)
        ]
        self._last_action_vecs: list[np.ndarray] = []
        self._last_logprobs: list[float] = []
        self._last_values: list[float] = []

    def _extend_state(self, state: TopLevelState) -> np.ndarray:
        vector = np.asarray(state.to_vector(self.domain_names), dtype=np.float32)
        if vector.shape[0] < self._state_dim:
            padded = np.zeros(self._state_dim, dtype=np.float32)
            padded[: vector.shape[0]] = vector
            return padded
        if vector.shape[0] > self._state_dim:
            return vector[: self._state_dim]
        return vector

    def act(self, state: TopLevelState, stochastic: bool = True) -> TopLevelAction:
        vector = self._extend_state(state)
        weights_list: list[np.ndarray] = []
        holds: list[int] = []
        action_vecs: list[np.ndarray] = []
        logprobs: list[float] = []
        values: list[float] = []
        entropies: list[float] = []

        for policy in self.policies:
            weights, hold_steps, action_vec, logprob, value, entropy = policy.select_action(
                vector,
                stochastic=stochastic,
            )
            weights_list.append(weights)
            holds.append(hold_steps)
            action_vecs.append(action_vec)
            logprobs.append(logprob)
            values.append(value)
            entropies.append(entropy)

        avg_weights = np.mean(np.stack(weights_list, axis=0), axis=0)
        avg_weights = np.maximum(avg_weights, 1e-8)
        avg_weights = avg_weights / float(np.sum(avg_weights))
        hold_steps = int(round(float(np.mean(holds))))
        uncertainty = float(np.std(np.stack(weights_list, axis=0), axis=0).mean() + np.mean(entropies))

        self._last_action_vecs = action_vecs
        self._last_logprobs = logprobs
        self._last_values = values
        return TopLevelAction(
            domain_weights={domain: float(avg_weights[i]) for i, domain in enumerate(self.domain_names)},
            hold_steps=hold_steps,
            action_id="top_continuous_ppo",
            uncertainty=uncertainty,
        ).normalized()

    def remember(
        self,
        state: TopLevelState,
        action: TopLevelAction,
        reward: float,
        next_state: TopLevelState,
        done: bool,
    ) -> None:
        del action, next_state
        vector = self._extend_state(state)
        for idx, policy in enumerate(self.policies):
            if idx >= len(self._last_action_vecs):
                continue
            policy.remember(
                state=vector,
                action_vec=self._last_action_vecs[idx],
                logprob=self._last_logprobs[idx],
                value=self._last_values[idx],
                reward=reward,
                done=done,
            )

    def learn(self) -> float | None:
        losses = [policy.learn() for policy in self.policies]
        valid = [loss for loss in losses if loss is not None]
        if not valid:
            return None
        return float(np.mean(valid))


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
        num_global_features: int = 0,
        seed: int = 0,
        train_router: bool = False,
    ) -> None:
        self.domain_names = list(domain_names)
        self.max_hold_steps = max_hold_steps
        self.rng = random.Random(seed)
        self.train_router = train_router

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = TopLevelState.vector_size(len(self.domain_names), num_global_features=num_global_features)
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
            uncertainty=0.0,
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
