"""Stochastic-control signals passed from top manager to domain managers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

import numpy as np

from .config import StochasticControlConfig
from .spaces import TopLevelAction, TopLevelState


@dataclass(frozen=True)
class DomainControlSignal:
    """Top-down control signal for a domain manager."""

    capital_budget: float
    risk_budget: float
    max_stock_weight: float
    hold_steps: int
    merton_fraction: float
    uncertainty: float


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        if not clipped:
            return {}
        equal = 1.0 / len(clipped)
        return {k: equal for k in clipped}
    return {k: v / total for k, v in clipped.items()}


class StochasticController:
    """Computes domain controls from stochastic-control-inspired equations.

    Control equations:
    - Merton fraction proxy:
      w* = mu / (gamma * sigma^2)
    - Latent regime follows OU dynamics:
      x_{t+1} = x_t + kappa (w* - x_t) + sigma * epsilon_t
    - Hold duration scales with expected Sharpe-like term:
      h ~ exp(mu/sigma - lambda * uncertainty)
    """

    def __init__(
        self,
        domain_names: Sequence[str],
        config: StochasticControlConfig,
        max_hold_steps: int,
        domain_stock_count: Mapping[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.domain_names = list(domain_names)
        self.config = config
        self.max_hold_steps = max(1, int(max_hold_steps))
        self.domain_stock_count = {
            str(k): max(1, int(v))
            for k, v in dict(domain_stock_count or {}).items()
        }
        self.rng = random.Random(seed)
        self._latent_state = {domain: 0.0 for domain in self.domain_names}
        mid_hold = max(1, int(round(self.max_hold_steps * 0.5)))
        self._last_hold = {domain: mid_hold for domain in self.domain_names}

    def build_controls(
        self,
        state: TopLevelState,
        top_action: TopLevelAction,
    ) -> dict[str, DomainControlSignal]:
        base_budgets = _normalize(top_action.domain_weights)
        if not base_budgets:
            return {}

        momentum_mean = float(np.mean([state.domain_momentum.get(d, 0.0) for d in self.domain_names]))
        raw_budgets: dict[str, float] = {}
        raw_risk: dict[str, float] = {}
        raw_max_stock: dict[str, float] = {}
        raw_hold: dict[str, int] = {}
        raw_merton: dict[str, float] = {}
        raw_uncertainty: dict[str, float] = {}

        for domain in self.domain_names:
            if domain not in base_budgets:
                continue

            mu = float(state.domain_momentum.get(domain, 0.0))
            sigma = float(max(1e-4, state.domain_volatility.get(domain, 1.0)))
            liq = float(max(0.0, state.domain_liquidity.get(domain, 0.0)))
            prev_alloc = float(max(0.0, state.current_allocation.get(domain, 0.0)))
            action_uncertainty = float(max(0.0, top_action.uncertainty))
            global_features = state.global_features or {}
            market_vol = float(max(0.0, global_features.get("market_vol", 0.0)))
            market_dd = float(abs(min(0.0, global_features.get("market_drawdown", 0.0))))
            bear_flag = float(max(0.0, global_features.get("regime_bear", 0.0)))
            vol_flag = float(max(0.0, global_features.get("regime_volatile", 0.0)))
            market_stress = min(3.0, market_vol + 2.0 * market_dd + bear_flag + 0.5 * vol_flag)
            stress_scale = 1.0 / (1.0 + market_stress)

            # Merton fraction proxy for risky allocation intensity.
            merton = mu / (max(1e-4, self.config.risk_aversion) * sigma * sigma)
            # Uncertainty penalty increases with volatility and momentum dispersion.
            uncertainty = (
                sigma
                + self.config.uncertainty_penalty * abs(mu - momentum_mean)
                + 1.0 / (1.0 + liq)
                + 0.25 * action_uncertainty
                + 0.5 * market_stress
            )

            latent_prev = self._latent_state.get(domain, 0.0)
            epsilon = self.rng.gauss(0.0, 1.0)
            latent_new = (
                latent_prev
                + self.config.mean_reversion_speed * (merton - latent_prev)
                + sigma * 0.1 * epsilon
            )
            self._latent_state[domain] = latent_new

            confidence = 1.0 / (1.0 + uncertainty)
            control_term = 0.5 * math.tanh(latent_new) + 0.5 * confidence
            budget_score = (
                base_budgets[domain] * (0.6 + 0.4 * control_term) * (0.65 + 0.35 * stress_scale)
                + 0.1 * prev_alloc
            )
            budget_score = max(0.0, budget_score)

            risk_score = (1.0 / sigma) * (0.5 + 0.5 * stress_scale)
            hold_score = math.exp(
                self.config.hold_scale
                * ((mu / sigma) - self.config.uncertainty_penalty * uncertainty)
            )
            proposed_hold = int(round(top_action.hold_steps * hold_score))
            local_max_hold = max(1, min(self.max_hold_steps, top_action.hold_steps))
            min_ratio = max(0.0, min(1.0, self.config.min_hold_ratio))
            local_min_hold = max(1, min(local_max_hold, int(round(local_max_hold * min_ratio))))
            proposed_hold = max(local_min_hold, min(local_max_hold, proposed_hold))
            prev_hold = self._last_hold.get(domain, local_max_hold)
            inertia = max(0.0, min(0.99, self.config.hold_inertia))
            hold_steps = int(round(inertia * prev_hold + (1.0 - inertia) * proposed_hold))
            hold_steps = max(local_min_hold, min(local_max_hold, hold_steps))
            self._last_hold[domain] = hold_steps

            max_stock_weight = self.config.max_single_stock_weight * (0.65 + 0.35 * confidence) * (0.7 + 0.3 * stress_scale)
            min_feasible = 1.0 / float(self.domain_stock_count.get(domain, 2))
            max_stock_weight = float(min(0.95, max(min_feasible, max_stock_weight)))

            raw_budgets[domain] = budget_score
            raw_risk[domain] = risk_score
            raw_max_stock[domain] = max_stock_weight
            raw_hold[domain] = hold_steps
            raw_merton[domain] = float(merton)
            raw_uncertainty[domain] = float(uncertainty)

        capital_budgets = _normalize(raw_budgets)
        risk_budgets = _normalize(raw_risk)

        controls: dict[str, DomainControlSignal] = {}
        min_alloc = max(0.0, self.config.min_domain_allocation)
        for domain in self.domain_names:
            if domain not in capital_budgets:
                continue
            cap = capital_budgets[domain]
            if cap < min_alloc:
                cap = 0.0
            controls[domain] = DomainControlSignal(
                capital_budget=cap,
                risk_budget=risk_budgets.get(domain, 0.0),
                max_stock_weight=raw_max_stock[domain],
                hold_steps=raw_hold[domain],
                merton_fraction=raw_merton[domain],
                uncertainty=raw_uncertainty[domain],
            )

        # Renormalize positive capital budgets after min-alloc gating.
        gated = _normalize({d: c.capital_budget for d, c in controls.items()})
        for domain, control in list(controls.items()):
            controls[domain] = DomainControlSignal(
                capital_budget=gated.get(domain, 0.0),
                risk_budget=control.risk_budget,
                max_stock_weight=control.max_stock_weight,
                hold_steps=control.hold_steps,
                merton_fraction=control.merton_fraction,
                uncertainty=control.uncertainty,
            )
        return controls

