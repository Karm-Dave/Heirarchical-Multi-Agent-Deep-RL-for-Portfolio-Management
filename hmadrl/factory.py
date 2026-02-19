"""Factory helpers for building switchable top-level manager + hierarchy."""

from __future__ import annotations

from typing import Mapping, Sequence

from .config import RLConfig, StochasticControlConfig
from .domain_manager import DomainRLManager
from .hierarchy import HierarchicalPortfolioAgent
from .stochastic_control import StochasticController
from .top_manager import MoERouterTopManager, RLTopManager, TopManagerBase


def build_top_manager(
    mode: str,
    domain_names: Sequence[str],
    max_hold_steps: int,
    num_global_features: int = 0,
    min_hold_steps: int = 2,
    hidden_dims: Sequence[int] = (128, 128),
    ensemble_size: int = 1,
    learning_rate: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 1e-3,
    ppo_epochs: int = 4,
    hold_inertia: float = 0.75,
    seed: int = 0,
) -> TopManagerBase:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "rl":
        return RLTopManager(
            domain_names=domain_names,
            max_hold_steps=max_hold_steps,
            num_global_features=num_global_features,
            min_hold_steps=min_hold_steps,
            hidden_dims=hidden_dims,
            ensemble_size=ensemble_size,
            learning_rate=learning_rate,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            ppo_epochs=ppo_epochs,
            hold_inertia=hold_inertia,
            seed=seed,
        )
    if normalized_mode in {"moe", "router", "moe_router"}:
        return MoERouterTopManager(
            domain_names=domain_names,
            max_hold_steps=max_hold_steps,
            num_global_features=num_global_features,
            seed=seed,
            train_router=False,
        )
    raise ValueError(f"Unsupported top manager mode: {mode}")


def build_hierarchical_agent(
    mode: str,
    domain_to_stocks: Mapping[str, Sequence[str]],
    max_domain_hold_steps: int = 8,
    max_stock_hold_steps: int = 5,
    num_global_features: int = 0,
    num_domain_factors: int = 0,
    rl_config: RLConfig | None = None,
    stochastic_control: StochasticControlConfig | None = None,
    seed: int = 0,
) -> HierarchicalPortfolioAgent:
    domain_names = list(domain_to_stocks.keys())
    rl = rl_config
    top_manager = build_top_manager(
        mode=mode,
        domain_names=domain_names,
        max_hold_steps=rl.max_domain_hold_steps if rl is not None else max_domain_hold_steps,
        num_global_features=num_global_features,
        min_hold_steps=rl.min_domain_hold_steps if rl is not None else 2,
        hidden_dims=tuple(rl.hidden_dims) if rl is not None else (128, 128),
        ensemble_size=rl.ensemble_size if rl is not None else 1,
        learning_rate=rl.ppo_lr if rl is not None else 3e-4,
        clip_ratio=rl.ppo_clip if rl is not None else 0.2,
        entropy_coef=rl.ppo_entropy_coef if rl is not None else 1e-3,
        ppo_epochs=rl.ppo_epochs if rl is not None else 4,
        hold_inertia=stochastic_control.hold_inertia if stochastic_control is not None else 0.75,
        seed=seed,
    )

    domain_managers = {
        domain: DomainRLManager(
            domain_name=domain,
            stock_names=stocks,
            max_hold_steps=rl.max_stock_hold_steps if rl is not None else max_stock_hold_steps,
            num_domain_factors=num_domain_factors,
            min_hold_steps=rl.min_stock_hold_steps if rl is not None else 2,
            hidden_dims=tuple(rl.hidden_dims) if rl is not None else (128, 128),
            ensemble_size=rl.ensemble_size if rl is not None else 1,
            learning_rate=rl.sac_lr if rl is not None else 3e-4,
            gamma=rl.sac_gamma if rl is not None else 0.99,
            alpha=rl.sac_alpha if rl is not None else 0.15,
            batch_size=rl.sac_batch_size if rl is not None else 64,
            replay_size=rl.sac_replay_size if rl is not None else 20000,
            hold_inertia=stochastic_control.hold_inertia if stochastic_control is not None else 0.75,
            seed=seed,
        )
        for domain, stocks in domain_to_stocks.items()
    }

    controller = None
    if stochastic_control is not None:
        controller = StochasticController(
            domain_names=domain_names,
            config=stochastic_control,
            max_hold_steps=max_domain_hold_steps,
            seed=seed,
        )

    return HierarchicalPortfolioAgent(
        top_manager=top_manager,
        domain_managers=domain_managers,
        stochastic_controller=controller,
    )
