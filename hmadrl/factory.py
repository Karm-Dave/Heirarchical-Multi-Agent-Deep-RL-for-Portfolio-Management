"""Factory helpers for building switchable top-level manager + hierarchy."""

from __future__ import annotations

from typing import Mapping, Sequence

from .domain_manager import DomainRLManager
from .hierarchy import HierarchicalPortfolioAgent
from .top_manager import MoERouterTopManager, RLTopManager, TopManagerBase


def build_top_manager(
    mode: str,
    domain_names: Sequence[str],
    max_hold_steps: int,
    seed: int = 0,
) -> TopManagerBase:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "rl":
        return RLTopManager(
            domain_names=domain_names,
            max_hold_steps=max_hold_steps,
            seed=seed,
        )
    if normalized_mode in {"moe", "router", "moe_router"}:
        return MoERouterTopManager(
            domain_names=domain_names,
            max_hold_steps=max_hold_steps,
            seed=seed,
            train_router=False,
        )
    raise ValueError(f"Unsupported top manager mode: {mode}")


def build_hierarchical_agent(
    mode: str,
    domain_to_stocks: Mapping[str, Sequence[str]],
    max_domain_hold_steps: int = 8,
    max_stock_hold_steps: int = 5,
    seed: int = 0,
) -> HierarchicalPortfolioAgent:
    domain_names = list(domain_to_stocks.keys())
    top_manager = build_top_manager(
        mode=mode,
        domain_names=domain_names,
        max_hold_steps=max_domain_hold_steps,
        seed=seed,
    )

    domain_managers = {
        domain: DomainRLManager(
            domain_name=domain,
            stock_names=stocks,
            max_hold_steps=max_stock_hold_steps,
            seed=seed,
        )
        for domain, stocks in domain_to_stocks.items()
    }

    return HierarchicalPortfolioAgent(top_manager=top_manager, domain_managers=domain_managers)

