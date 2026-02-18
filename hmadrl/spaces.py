"""State/action definitions for the hierarchical portfolio system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
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
    domain_momentum: Mapping[str, float]
    domain_volatility: Mapping[str, float]
    current_allocation: Mapping[str, float]
    remaining_horizon_steps: int
    elapsed_steps: int
    cash_buffer: float = 0.0

    def to_vector(self, domain_names: Sequence[str]) -> list[float]:
        vector: list[float] = []
        for domain in domain_names:
            vector.append(float(self.domain_momentum.get(domain, 0.0)))
            vector.append(float(self.domain_volatility.get(domain, 1.0)))
            vector.append(float(self.current_allocation.get(domain, 0.0)))
        vector.append(float(self.remaining_horizon_steps))
        vector.append(float(self.elapsed_steps))
        vector.append(float(self.cash_buffer))
        return vector

    @staticmethod
    def vector_size(num_domains: int) -> int:
        return 3 * num_domains + 3


@dataclass(frozen=True)
class TopLevelAction:
    domain_weights: Mapping[str, float]
    hold_steps: int
    template_id: str = "manual"

    def normalized(self) -> "TopLevelAction":
        return TopLevelAction(
            domain_weights=_normalize_weights(self.domain_weights),
            hold_steps=max(1, int(self.hold_steps)),
            template_id=self.template_id,
        )


@dataclass(frozen=True)
class DomainState:
    domain_name: str
    stock_momentum: Mapping[str, float]
    stock_volatility: Mapping[str, float]
    current_stock_allocation: Mapping[str, float]
    remaining_domain_horizon_steps: int
    elapsed_steps: int

    def to_vector(self, stock_names: Sequence[str]) -> list[float]:
        vector: list[float] = []
        for stock in stock_names:
            vector.append(float(self.stock_momentum.get(stock, 0.0)))
            vector.append(float(self.stock_volatility.get(stock, 1.0)))
            vector.append(float(self.current_stock_allocation.get(stock, 0.0)))
        vector.append(float(self.remaining_domain_horizon_steps))
        vector.append(float(self.elapsed_steps))
        return vector

    @staticmethod
    def vector_size(num_stocks: int) -> int:
        return 3 * num_stocks + 2


@dataclass(frozen=True)
class DomainAction:
    stock_weights: Mapping[str, float]
    hold_steps: int
    template_id: str = "manual"

    def normalized(self) -> "DomainAction":
        return DomainAction(
            stock_weights=_normalize_weights(self.stock_weights),
            hold_steps=max(1, int(self.hold_steps)),
            template_id=self.template_id,
        )


def _focused_template(
    names: Sequence[str],
    focus_name: str,
    focus_weight: float,
) -> Dict[str, float]:
    count = len(names)
    if count == 1:
        return {focus_name: 1.0}
    remainder = max(0.0, 1.0 - focus_weight)
    other_weight = remainder / (count - 1)
    return {
        name: focus_weight if name == focus_name else other_weight
        for name in names
    }


def generate_top_level_action_templates(
    domain_names: Sequence[str],
    max_hold_steps: int,
    focus_weight: float = 0.8,
) -> list[TopLevelAction]:
    if not domain_names:
        raise ValueError("domain_names must not be empty")
    if max_hold_steps < 1:
        raise ValueError("max_hold_steps must be >= 1")

    actions: list[TopLevelAction] = []
    equal_weights = {d: 1.0 / len(domain_names) for d in domain_names}
    for hold in range(1, max_hold_steps + 1):
        actions.append(
            TopLevelAction(
                domain_weights=equal_weights,
                hold_steps=hold,
                template_id=f"top_equal_h{hold}",
            )
        )
        for domain in domain_names:
            actions.append(
                TopLevelAction(
                    domain_weights=_focused_template(
                        domain_names, domain, focus_weight
                    ),
                    hold_steps=hold,
                    template_id=f"top_focus_{domain}_h{hold}",
                )
            )
    return [a.normalized() for a in actions]


def generate_domain_action_templates(
    domain_name: str,
    stock_names: Sequence[str],
    max_hold_steps: int,
    focus_weight: float = 0.85,
) -> list[DomainAction]:
    if not stock_names:
        raise ValueError("stock_names must not be empty")
    if max_hold_steps < 1:
        raise ValueError("max_hold_steps must be >= 1")

    actions: list[DomainAction] = []
    equal_weights = {s: 1.0 / len(stock_names) for s in stock_names}
    for hold in range(1, max_hold_steps + 1):
        actions.append(
            DomainAction(
                stock_weights=equal_weights,
                hold_steps=hold,
                template_id=f"{domain_name}_equal_h{hold}",
            )
        )
        for stock in stock_names:
            actions.append(
                DomainAction(
                    stock_weights=_focused_template(
                        stock_names, stock, focus_weight
                    ),
                    hold_steps=hold,
                    template_id=f"{domain_name}_focus_{stock}_h{hold}",
                )
            )
    return [a.normalized() for a in actions]


def ensure_weights_sum_to_one(values: Mapping[str, float]) -> bool:
    total = sum(values.values())
    return abs(total - 1.0) < 1e-8


def keys_match(keys: Iterable[str], values: Mapping[str, float]) -> bool:
    return set(keys) == set(values.keys())

