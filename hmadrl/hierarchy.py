"""Two-layer hierarchical coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domain_manager import DomainRLManager
from .spaces import DomainAction, DomainState, TopLevelAction, TopLevelState
from .top_manager import TopManagerBase


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        if not weights:
            return {}
        equal = 1.0 / len(weights)
        return {k: equal for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


@dataclass(frozen=True)
class HierarchicalDecision:
    top_action: TopLevelAction
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
    ) -> None:
        self.top_manager = top_manager
        self.domain_managers = dict(domain_managers)

    def decide(
        self,
        top_state: TopLevelState,
        domain_states: Mapping[str, DomainState],
        stochastic: bool = True,
    ) -> HierarchicalDecision:
        top_action = self.top_manager.act(top_state, stochastic=stochastic).normalized()

        domain_actions: dict[str, DomainAction] = {}
        final_stock_weights: dict[str, float] = {}
        stock_rebalance: dict[str, int] = {}

        for domain, domain_weight in top_action.domain_weights.items():
            if domain_weight <= 0:
                continue
            manager = self.domain_managers.get(domain)
            state = domain_states.get(domain)
            if manager is None or state is None:
                continue

            action = manager.act(
                state=state,
                parent_hold_steps=top_action.hold_steps,
                stochastic=stochastic,
            )
            domain_actions[domain] = action
            stock_rebalance[domain] = action.hold_steps
            for stock, stock_weight in action.stock_weights.items():
                key = f"{domain}:{stock}"
                final_stock_weights[key] = final_stock_weights.get(key, 0.0) + domain_weight * stock_weight

        final_stock_weights = _normalize(final_stock_weights)
        return HierarchicalDecision(
            top_action=top_action,
            domain_actions=domain_actions,
            final_stock_weights=final_stock_weights,
            domain_rebalance_after_steps=top_action.hold_steps,
            stock_rebalance_after_steps=stock_rebalance,
        )

