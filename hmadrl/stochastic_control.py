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
        seed: int = 0,
    ) -> None:
        self.domain_names = list(domain_names)
        self.config = config
        self.max_hold_steps = max(1, int(max_hold_steps))
        self.rng = random.Random(seed)
        self._latent_state = {domain: 0.0 for domain in self.domain_names}

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

            # Merton fraction proxy for risky allocation intensity.
            merton = mu / (max(1e-4, self.config.risk_aversion) * sigma * sigma)
            # Uncertainty penalty increases with volatility and momentum dispersion.
            uncertainty = sigma + self.config.uncertainty_penalty * abs(mu - momentum_mean) + 1.0 / (1.0 + liq)

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
            budget_score = base_budgets[domain] * (0.6 + 0.4 * control_term) + 0.1 * prev_alloc
            budget_score = max(0.0, budget_score)

            risk_score = 1.0 / sigma
            hold_score = math.exp(
                self.config.hold_scale
                * ((mu / sigma) - self.config.uncertainty_penalty * uncertainty)
            )
            hold_steps = int(round(top_action.hold_steps * hold_score))
            hold_steps = max(1, min(self.max_hold_steps, min(top_action.hold_steps, hold_steps)))

            max_stock_weight = self.config.max_single_stock_weight * (0.75 + 0.25 * confidence)
            max_stock_weight = float(min(0.95, max(0.1, max_stock_weight)))

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

