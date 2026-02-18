"""Domain-level stock allocation managers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .rl_core import DQNPolicy
from .spaces import DomainAction, DomainState, generate_domain_templates


class DomainRLManager:
    """DQN domain manager: allocates among stocks and picks stock hold steps."""

    def __init__(
        self,
        domain_name: str,
        stock_names: Sequence[str],
        max_hold_steps: int,
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
    ) -> None:
        self.domain_name = domain_name
        self.stock_names = list(stock_names)
        self.templates = generate_domain_templates(
            domain_name=domain_name,
            stock_names=self.stock_names,
            max_hold_steps=max_hold_steps,
        )
        self._template_index = {action.action_id: i for i, action in enumerate(self.templates)}
        state_dim = DomainState.vector_size(len(self.stock_names))
        self.policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=len(self.templates),
            hidden_dims=hidden_dims,
            seed=seed,
        )

    def act(
        self,
        state: DomainState,
        parent_hold_steps: int,
        stochastic: bool = True,
    ) -> DomainAction:
        vector = np.asarray(state.to_vector(self.stock_names), dtype=np.float32)
        idx = self.policy.select_action(vector, stochastic=stochastic)
        base_action = self.templates[idx]
        hold = min(base_action.hold_steps, max(1, parent_hold_steps))
        return DomainAction(
            stock_weights=base_action.stock_weights,
            hold_steps=hold,
            action_id=base_action.action_id,
        ).normalized()

    def remember(
        self,
        state: DomainState,
        action: DomainAction,
        reward: float,
        next_state: DomainState,
        done: bool,
    ) -> None:
        idx = self._template_index[action.action_id]
        self.policy.store(
            state=state.to_vector(self.stock_names),
            action=idx,
            reward=reward,
            next_state=next_state.to_vector(self.stock_names),
            done=done,
        )

    def learn(self) -> float | None:
        return self.policy.train_step()

