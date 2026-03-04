"""Domain-level stock allocation managers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .rl_core import SACContinuousPolicy
from .spaces import DomainAction, DomainState
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
    """Continuous SAC domain manager with optional policy ensembles."""

    def __init__(
        self,
        domain_name: str,
        stock_names: Sequence[str],
        max_hold_steps: int,
        num_domain_factors: int = 0,
        num_stock_features: int = 0,
        stock_feature_names: Sequence[str] | None = None,
        min_hold_steps: int = 2,
        hidden_dims: Sequence[int] = (128, 128),
        ensemble_size: int = 1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.15,
        auto_alpha: bool = True,
        target_entropy: float = -2.0,
        use_dueling_critic: bool = True,
        batch_size: int = 64,
        replay_size: int = 20000,
        hold_inertia: float = 0.75,
        uncertainty_alpha: float = 0.75,
        n_step: int = 3,
        seed: int = 0,
    ) -> None:
        self.domain_name = domain_name
        self.stock_names = list(stock_names)
        self.stock_feature_names = list(stock_feature_names or [])
        self._state_dim = DomainState.vector_size(
            len(self.stock_names),
            num_domain_factors=num_domain_factors,
            num_stock_features=num_stock_features,
        )
        self._ensemble_size = max(1, int(ensemble_size))
        self.policies = [
            SACContinuousPolicy(
                state_dim=self._state_dim,
                simplex_dim=len(self.stock_names),
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                auto_alpha=auto_alpha,
                target_entropy=target_entropy,
                use_dueling_critic=use_dueling_critic,
                batch_size=batch_size,
                replay_size=replay_size,
                min_hold_steps=min_hold_steps,
                max_hold_steps=max_hold_steps,
                hold_inertia=hold_inertia,
                n_step=n_step,
                seed=seed + i * 31,
            )
            for i in range(self._ensemble_size)
        ]
        self._last_action_vecs: list[np.ndarray] = []
        self.uncertainty_alpha = float(max(0.0, uncertainty_alpha))
        self._risk_lambda = 0.25
        self._risk_blend = 0.5

    def _extend_state(self, state: DomainState) -> np.ndarray:
        vector = np.asarray(
            state.to_vector(
                self.stock_names,
                stock_feature_names=self.stock_feature_names,
            ),
            dtype=np.float32,
        )
        if vector.shape[0] < self._state_dim:
            padded = np.zeros(self._state_dim, dtype=np.float32)
            padded[: vector.shape[0]] = vector
            return padded
        if vector.shape[0] > self._state_dim:
            return vector[: self._state_dim]
        return vector

    def act(
        self,
        state: DomainState,
        parent_hold_steps: int,
        control: DomainControlSignal | None = None,
        stochastic: bool = True,
    ) -> DomainAction:
        vector = self._extend_state(state)
        weights_list: list[np.ndarray] = []
        holds: list[int] = []
        uncertainties: list[float] = []
        action_vecs: list[np.ndarray] = []

        for policy in self.policies:
            weights, hold, action_vec, uncertainty = policy.select_action(vector, stochastic=stochastic)
            weights_list.append(weights)
            holds.append(hold)
            uncertainties.append(uncertainty)
            action_vecs.append(action_vec)

        stock_weights = np.mean(np.stack(weights_list, axis=0), axis=0)
        stock_weights = np.maximum(stock_weights, 1e-8)
        sigma = np.std(np.stack(weights_list, axis=0), axis=0)
        stock_weights = stock_weights * np.exp(-self.uncertainty_alpha * np.maximum(0.0, sigma))
        stock_weights = np.maximum(stock_weights, 1e-8)
        stock_weights = stock_weights / float(np.sum(stock_weights))
        hold = int(round(float(np.mean(holds))))
        hold = min(hold, max(1, parent_hold_steps))
        uncertainty = float(np.std(np.stack(weights_list, axis=0), axis=0).mean() + np.mean(uncertainties))
        self._last_action_vecs = action_vecs

        stock_weights_map = {
            stock: float(stock_weights[i]) for i, stock in enumerate(self.stock_names)
        }
        if control is not None:
            hold = min(hold, max(1, control.hold_steps))

            # Risk-constrained SAC blend via a lightweight Lagrangian penalty.
            vol_vec = np.asarray(
                [max(1e-4, float(state.stock_volatility.get(stock, 1.0))) for stock in self.stock_names],
                dtype=float,
            )
            w_vec = np.asarray([float(stock_weights_map.get(stock, 0.0)) for stock in self.stock_names], dtype=float)
            target_risk = max(1e-5, float(control.risk_budget) * float(np.mean(vol_vec)))
            realized_risk = float(np.dot(w_vec, vol_vec))
            violation = realized_risk - target_risk
            self._risk_lambda = float(max(0.0, min(10.0, self._risk_lambda + 0.05 * violation)))
            lagrangian_penalty = np.exp(-self._risk_lambda * (vol_vec / max(1e-6, float(np.mean(vol_vec)))))
            constrained = np.maximum(1e-8, w_vec * lagrangian_penalty)
            constrained = constrained / float(np.sum(constrained))
            blend = float(max(0.0, min(1.0, self._risk_blend)))
            mixed = (1.0 - blend) * w_vec + blend * constrained
            mixed = mixed / float(np.sum(mixed))
            stock_weights_map = {stock: float(mixed[i]) for i, stock in enumerate(self.stock_names)}
            stock_weights_map = _cap_weights(stock_weights_map, control.max_stock_weight)

        return DomainAction(
            stock_weights=stock_weights_map,
            hold_steps=hold,
            action_id=f"{self.domain_name}_continuous_sac",
            uncertainty=uncertainty,
        ).normalized()

    def remember(
        self,
        state: DomainState,
        action: DomainAction,
        reward: float,
        next_state: DomainState,
        done: bool,
    ) -> None:
        del action
        if reward < 0.0:
            self._risk_blend = float(min(1.0, self._risk_blend + 0.01))
        else:
            self._risk_blend = float(max(0.0, self._risk_blend - 0.005))
        state_vec = self._extend_state(state)
        next_state_vec = self._extend_state(next_state)
        for idx, policy in enumerate(self.policies):
            if idx >= len(self._last_action_vecs):
                continue
            policy.store(
                state=state_vec,
                action_vec=self._last_action_vecs[idx],
                reward=reward,
                next_state=next_state_vec,
                done=done,
            )

    def learn(self) -> float | None:
        losses = [policy.train_step() for policy in self.policies]
        valid = [loss for loss in losses if loss is not None]
        if not valid:
            return None
        return float(np.mean(valid))
