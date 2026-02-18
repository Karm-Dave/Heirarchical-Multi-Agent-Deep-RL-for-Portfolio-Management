"""State/action spaces for the hierarchical portfolio agent."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        if not clipped:
            return {}
        equal = 1.0 / len(clipped)
        return {k: equal for k in clipped}
    return {k: v / total for k, v in clipped.items()}


@dataclass(frozen=True)
class TopLevelState:
    """Portfolio-manager state at the domain allocation level."""

    domain_momentum: Mapping[str, float]
    domain_volatility: Mapping[str, float]
    domain_liquidity: Mapping[str, float]
    current_allocation: Mapping[str, float]
    remaining_horizon_steps: int
    step_index: int
    cash_ratio: float

    def to_vector(self, domain_names: Sequence[str]) -> list[float]:
        out: list[float] = []
        for domain in domain_names:
            out.extend(
                [
                    float(self.domain_momentum.get(domain, 0.0)),
                    float(self.domain_volatility.get(domain, 1.0)),
                    float(self.domain_liquidity.get(domain, 0.0)),
                    float(self.current_allocation.get(domain, 0.0)),
                ]
            )
        out.extend(
            [
                float(self.remaining_horizon_steps),
                float(self.step_index),
                float(self.cash_ratio),
            ]
        )
        return out

    @staticmethod
    def vector_size(num_domains: int) -> int:
        return 4 * num_domains + 3


@dataclass(frozen=True)
class TopLevelAction:
    """Portfolio-manager action: domain allocation + domain hold period."""

    domain_weights: Mapping[str, float]
    hold_steps: int
    action_id: str

    def normalized(self) -> "TopLevelAction":
        return TopLevelAction(
            domain_weights=_normalize(self.domain_weights),
            hold_steps=max(1, int(self.hold_steps)),
            action_id=self.action_id,
        )


@dataclass(frozen=True)
class DomainState:
    """Domain-manager state at the stock allocation level."""

    domain_name: str
    stock_momentum: Mapping[str, float]
    stock_volatility: Mapping[str, float]
    stock_liquidity: Mapping[str, float]
    current_stock_allocation: Mapping[str, float]
    remaining_domain_steps: int
    step_index: int

    def to_vector(self, stock_names: Sequence[str]) -> list[float]:
        out: list[float] = []
        for stock in stock_names:
            out.extend(
                [
                    float(self.stock_momentum.get(stock, 0.0)),
                    float(self.stock_volatility.get(stock, 1.0)),
                    float(self.stock_liquidity.get(stock, 0.0)),
                    float(self.current_stock_allocation.get(stock, 0.0)),
                ]
            )
        out.extend([float(self.remaining_domain_steps), float(self.step_index)])
        return out

    @staticmethod
    def vector_size(num_stocks: int) -> int:
        return 4 * num_stocks + 2


@dataclass(frozen=True)
class DomainAction:
    """Domain-manager action: stock allocation + stock hold period."""

    stock_weights: Mapping[str, float]
    hold_steps: int
    action_id: str

    def normalized(self) -> "DomainAction":
        return DomainAction(
            stock_weights=_normalize(self.stock_weights),
            hold_steps=max(1, int(self.hold_steps)),
            action_id=self.action_id,
        )


def _focused_allocation(
    names: Sequence[str],
    focus: str,
    focus_weight: float,
) -> dict[str, float]:
    if len(names) == 1:
        return {focus: 1.0}
    remainder = max(0.0, 1.0 - focus_weight)
    other_weight = remainder / (len(names) - 1)
    return {name: focus_weight if name == focus else other_weight for name in names}


def generate_top_level_templates(
    domain_names: Sequence[str],
    max_hold_steps: int,
    focus_weight: float = 0.8,
) -> list[TopLevelAction]:
    """Discrete template action space for top-layer DQN."""

    if not domain_names:
        raise ValueError("domain_names cannot be empty")
    if max_hold_steps < 1:
        raise ValueError("max_hold_steps must be >= 1")

    actions: list[TopLevelAction] = []
    equal = {d: 1.0 / len(domain_names) for d in domain_names}

    for hold in range(1, max_hold_steps + 1):
        actions.append(
            TopLevelAction(equal, hold, action_id=f"top_equal_h{hold}").normalized()
        )
        for domain in domain_names:
            actions.append(
                TopLevelAction(
                    _focused_allocation(domain_names, domain, focus_weight),
                    hold,
                    action_id=f"top_focus_{domain}_h{hold}",
                ).normalized()
            )
        for left, right in combinations(domain_names, 2):
            pair = {d: 0.0 for d in domain_names}
            pair[left] = 0.5
            pair[right] = 0.5
            actions.append(
                TopLevelAction(
                    pair,
                    hold,
                    action_id=f"top_pair_{left}_{right}_h{hold}",
                ).normalized()
            )
    return actions


def generate_domain_templates(
    domain_name: str,
    stock_names: Sequence[str],
    max_hold_steps: int,
    focus_weight: float = 0.85,
) -> list[DomainAction]:
    """Discrete template action space for each domain manager."""

    if not stock_names:
        raise ValueError("stock_names cannot be empty")
    if max_hold_steps < 1:
        raise ValueError("max_hold_steps must be >= 1")

    actions: list[DomainAction] = []
    equal = {s: 1.0 / len(stock_names) for s in stock_names}

    for hold in range(1, max_hold_steps + 1):
        actions.append(
            DomainAction(
                equal,
                hold,
                action_id=f"{domain_name}_equal_h{hold}",
            ).normalized()
        )
        for stock in stock_names:
            actions.append(
                DomainAction(
                    _focused_allocation(stock_names, stock, focus_weight),
                    hold,
                    action_id=f"{domain_name}_focus_{stock}_h{hold}",
                ).normalized()
            )
    return actions

