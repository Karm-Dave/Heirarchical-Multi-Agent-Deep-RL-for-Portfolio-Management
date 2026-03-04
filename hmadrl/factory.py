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
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 1e-3,
    ppo_epochs: int = 4,
    network_type: str = "mlp",
    transformer_d_model: int = 96,
    transformer_nhead: int = 4,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    hold_inertia: float = 0.75,
    uncertainty_alpha: float = 0.75,
    router_entropy_coef: float = 0.01,
    router_load_balance_coef: float = 0.03,
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
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            ppo_epochs=ppo_epochs,
            network_type=network_type,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
            hold_inertia=hold_inertia,
            uncertainty_alpha=uncertainty_alpha,
            seed=seed,
        )
    if normalized_mode in {"moe", "router", "moe_router"}:
        return MoERouterTopManager(
            domain_names=domain_names,
            max_hold_steps=max_hold_steps,
            num_global_features=num_global_features,
            seed=seed,
            train_router=True,
            entropy_coef=router_entropy_coef,
            load_balance_coef=router_load_balance_coef,
        )
    raise ValueError(f"Unsupported top manager mode: {mode}")


def build_hierarchical_agent(
    mode: str,
    domain_to_stocks: Mapping[str, Sequence[str]],
    max_domain_hold_steps: int = 8,
    max_stock_hold_steps: int = 5,
    num_global_features: int = 0,
    num_domain_factors: int = 0,
    num_stock_features: int = 0,
    stock_feature_names: Sequence[str] | None = None,
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
        gamma=rl.ppo_gamma if rl is not None else 0.995,
        gae_lambda=rl.ppo_gae_lambda if rl is not None else 0.95,
        clip_ratio=rl.ppo_clip if rl is not None else 0.2,
        entropy_coef=rl.ppo_entropy_coef if rl is not None else 1e-3,
        ppo_epochs=rl.ppo_epochs if rl is not None else 4,
        network_type=rl.top_network_type if rl is not None else "mlp",
        transformer_d_model=rl.transformer_d_model if rl is not None else 96,
        transformer_nhead=rl.transformer_nhead if rl is not None else 4,
        transformer_layers=rl.transformer_layers if rl is not None else 2,
        transformer_dropout=rl.transformer_dropout if rl is not None else 0.1,
        hold_inertia=stochastic_control.hold_inertia if stochastic_control is not None else 0.75,
        uncertainty_alpha=rl.uncertainty_alpha if rl is not None else 0.75,
        router_entropy_coef=rl.router_entropy_coef if rl is not None else 0.01,
        router_load_balance_coef=rl.router_load_balance_coef if rl is not None else 0.03,
        seed=seed,
    )

    domain_managers = {
        domain: DomainRLManager(
            domain_name=domain,
            stock_names=stocks,
            max_hold_steps=rl.max_stock_hold_steps if rl is not None else max_stock_hold_steps,
            num_domain_factors=num_domain_factors,
            num_stock_features=num_stock_features,
            stock_feature_names=stock_feature_names,
            min_hold_steps=rl.min_stock_hold_steps if rl is not None else 2,
            hidden_dims=tuple(rl.hidden_dims) if rl is not None else (128, 128),
            ensemble_size=rl.ensemble_size if rl is not None else 1,
            learning_rate=rl.sac_lr if rl is not None else 3e-4,
            gamma=rl.sac_gamma if rl is not None else 0.99,
            tau=rl.sac_tau if rl is not None else 0.005,
            alpha=rl.sac_alpha if rl is not None else 0.15,
            auto_alpha=rl.sac_auto_alpha if rl is not None else True,
            target_entropy=rl.sac_target_entropy if rl is not None else -2.0,
            use_dueling_critic=rl.sac_use_dueling if rl is not None else True,
            batch_size=rl.sac_batch_size if rl is not None else 64,
            replay_size=rl.sac_replay_size if rl is not None else 20000,
            hold_inertia=stochastic_control.hold_inertia if stochastic_control is not None else 0.75,
            uncertainty_alpha=rl.uncertainty_alpha if rl is not None else 0.75,
            n_step=rl.sac_n_step if rl is not None else 3,
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
            domain_stock_count={domain: len(stocks) for domain, stocks in domain_to_stocks.items()},
            seed=seed,
        )

    return HierarchicalPortfolioAgent(
        top_manager=top_manager,
        domain_managers=domain_managers,
        stochastic_controller=controller,
    )
