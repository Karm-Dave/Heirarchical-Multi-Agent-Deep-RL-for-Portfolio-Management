"""Two-layer hierarchical coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domain_manager import DomainRLManager
from .spaces import DomainAction, DomainState, TopLevelAction, TopLevelState
from .stochastic_control import DomainControlSignal, StochasticController
from .top_manager import TopManagerBase


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        if not weights:
            return {}
        equal = 1.0 / len(weights)
        return {k: equal for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def _vol_target_weights(
    weights: Mapping[str, float],
    stock_volatility: Mapping[str, float],
    target_vol: float,
) -> dict[str, float]:
    out = _normalize(weights)
    if not out:
        return {}
    target = max(1e-4, float(target_vol))
    realized = 0.0
    for key, w in out.items():
        realized += float(w) * max(1e-4, float(stock_volatility.get(key, 0.2)))
    if realized <= target:
        return out
    scale = max(0.0, min(1.0, target / realized))
    equal = 1.0 / len(out)
    blended = {k: scale * float(v) + (1.0 - scale) * equal for k, v in out.items()}
    return _normalize(blended)


@dataclass(frozen=True)
class HierarchicalDecision:
    top_action: TopLevelAction
    domain_controls: Mapping[str, DomainControlSignal]
    final_domain_weights: Mapping[str, float]
    domain_actions: Mapping[str, DomainAction]
    final_stock_weights: Mapping[str, float]
    domain_rebalance_after_steps: int
    stock_rebalance_after_steps: Mapping[str, int]


class HierarchicalPortfolioAgent:
    """Coordinates top and domain policies."""

    def __init__(
        self,
        top_manager: TopManagerBase,
        domain_managers: Mapping[str, DomainRLManager],
        stochastic_controller: StochasticController | None = None,
        target_portfolio_vol: float = 0.2,
    ) -> None:
        self.top_manager = top_manager
        self.domain_managers = dict(domain_managers)
        self.stochastic_controller = stochastic_controller
        self.target_portfolio_vol = float(max(1e-4, target_portfolio_vol))

    def decide(
        self,
        top_state: TopLevelState,
        domain_states: Mapping[str, DomainState],
        stochastic: bool = True,
    ) -> HierarchicalDecision:
        top_action = self.top_manager.act(top_state, stochastic=stochastic).normalized()
        if self.stochastic_controller is not None:
            controls = self.stochastic_controller.build_controls(top_state, top_action)
        else:
            controls = {
                domain: DomainControlSignal(
                    capital_budget=weight,
                    risk_budget=weight,
                    max_stock_weight=1.0,
                    hold_steps=top_action.hold_steps,
                    merton_fraction=0.0,
                    uncertainty=0.0,
                )
                for domain, weight in top_action.domain_weights.items()
            }

        domain_actions: dict[str, DomainAction] = {}
        final_domain_weights: dict[str, float] = {}
        final_stock_weights: dict[str, float] = {}
        stock_rebalance: dict[str, int] = {}

        for domain, control in controls.items():
            domain_weight = control.capital_budget
            if domain_weight <= 0:
                continue
            manager = self.domain_managers.get(domain)
            state = domain_states.get(domain)
            if manager is None or state is None:
                continue

            action = manager.act(
                state=state,
                parent_hold_steps=top_action.hold_steps,
                control=control,
                stochastic=stochastic,
            )
            final_domain_weights[domain] = domain_weight
            domain_actions[domain] = action
            stock_rebalance[domain] = action.hold_steps
            for stock, stock_weight in action.stock_weights.items():
                key = f"{domain}:{stock}"
                final_stock_weights[key] = final_stock_weights.get(key, 0.0) + domain_weight * stock_weight

        final_domain_weights = _normalize(final_domain_weights)
        final_stock_weights = _normalize(final_stock_weights)
        stock_vol_map: dict[str, float] = {}
        for domain, state in domain_states.items():
            for stock, vol in state.stock_volatility.items():
                stock_vol_map[f"{domain}:{stock}"] = float(max(1e-4, vol))
        market_vol = float((top_state.global_features or {}).get("market_vol", self.target_portfolio_vol))
        dynamic_target = max(0.05, min(0.6, market_vol if market_vol > 0.0 else self.target_portfolio_vol))
        final_stock_weights = _vol_target_weights(final_stock_weights, stock_vol_map, dynamic_target)
        return HierarchicalDecision(
            top_action=top_action,
            domain_controls=controls,
            final_domain_weights=final_domain_weights,
            domain_actions=domain_actions,
            final_stock_weights=final_stock_weights,
            domain_rebalance_after_steps=top_action.hold_steps,
            stock_rebalance_after_steps=stock_rebalance,
        )
