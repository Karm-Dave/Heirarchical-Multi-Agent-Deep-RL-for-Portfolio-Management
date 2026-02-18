"""Domain-level stock allocation managers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .rl_core import DQNPolicy
from .spaces import DomainAction, DomainState, generate_domain_templates
from .stochastic_control import DomainControlSignal


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        if not clipped:
            return {}
        equal = 1.0 / len(clipped)
        return {k: equal for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _cap_weights(weights: Mapping[str, float], max_weight: float) -> dict[str, float]:
    out = _normalize(weights)
    if not out:
        return {}

    n = len(out)
    max_weight = max(float(max_weight), 1.0 / n)
    max_weight = min(max_weight, 1.0)
    out = {k: min(v, max_weight) for k, v in out.items()}

    for _ in range(20):
        total = sum(out.values())
        gap = 1.0 - total
        if abs(gap) < 1e-8:
            break
        if gap < 0:
            out = _normalize(out)
            out = {k: min(v, max_weight) for k, v in out.items()}
            continue
        free = [k for k, v in out.items() if v < max_weight - 1e-10]
        if not free:
            out = _normalize(out)
            break
        add_per_name = gap / len(free)
        for name in free:
            room = max_weight - out[name]
            out[name] += min(room, add_per_name)
    return _normalize(out)


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
        control: DomainControlSignal | None = None,
        stochastic: bool = True,
    ) -> DomainAction:
        vector = np.asarray(state.to_vector(self.stock_names), dtype=np.float32)
        idx = self.policy.select_action(vector, stochastic=stochastic)
        base_action = self.templates[idx]

        hold = min(base_action.hold_steps, max(1, parent_hold_steps))
        stock_weights = dict(base_action.stock_weights)
        if control is not None:
            hold = min(hold, max(1, control.hold_steps))

            # Top-down risk control: blend RL action with inverse-volatility allocation.
            inv_vol = {
                stock: 1.0 / max(1e-4, state.stock_volatility.get(stock, 1.0))
                for stock in self.stock_names
            }
            inv_vol = _normalize(inv_vol)
            blend = max(0.0, min(1.0, control.risk_budget))
            stock_weights = {
                stock: (1.0 - blend) * stock_weights.get(stock, 0.0) + blend * inv_vol.get(stock, 0.0)
                for stock in self.stock_names
            }
            stock_weights = _cap_weights(stock_weights, control.max_stock_weight)

        return DomainAction(
            stock_weights=stock_weights,
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
