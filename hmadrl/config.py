"""Config loader for experimentation and training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RLConfig:
    top_algorithm: str
    domain_algorithm: str
    top_network_type: str
    max_domain_hold_steps: int
    max_stock_hold_steps: int
    min_domain_hold_steps: int
    min_stock_hold_steps: int
    random_seed: int
    training_steps: int
    stochastic_actions: bool
    ensemble_size: int
    hidden_dims: list[int]
    ppo_epochs: int
    ppo_clip: float
    ppo_lr: float
    ppo_gamma: float
    ppo_gae_lambda: float
    ppo_entropy_coef: float
    transformer_d_model: int
    transformer_nhead: int
    transformer_layers: int
    transformer_dropout: float
    sac_lr: float
    sac_gamma: float
    sac_tau: float
    sac_alpha: float
    sac_auto_alpha: bool
    sac_target_entropy: float
    sac_use_dueling: bool
    sac_batch_size: int
    sac_replay_size: int
    sac_n_step: int
    uncertainty_alpha: float
    router_entropy_coef: float
    router_load_balance_coef: float


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str
    interval: str
    provider: str
    lookback: int
    train_split: float
    cache_dir: str
    walk_forward_train: int
    walk_forward_test: int
    walk_forward_step: int
    market_symbol: str
    rate_symbols: list[str]
    volatility_symbol: str
    cross_asset_symbols: list[str]
    sector_etfs: dict[str, str]
    use_learned_clusters: bool
    learned_cluster_count: int


@dataclass(frozen=True)
class RewardConfig:
    transaction_cost_bps: float
    slippage_bps: float
    turnover_penalty: float
    downside_penalty: float
    cvar_alpha: float
    cvar_penalty: float
    reward_window: int
    rank_loss_coef: float = 0.0
    domain_entropy_coef: float = 0.0
    hold_penalty_coef: float = 0.0
    max_reasonable_hold: int = 8


@dataclass(frozen=True)
class StochasticControlConfig:
    risk_aversion: float
    hold_scale: float
    uncertainty_penalty: float
    mean_reversion_speed: float
    max_single_stock_weight: float
    min_domain_allocation: float
    min_hold_ratio: float = 0.25
    hold_inertia: float = 0.75


@dataclass(frozen=True)
class ExperimentConfig:
    results_dir: str
    run_name: str
    methods: list[str]
    modes: list[str]
    seeds: list[int]
    fine_tune_rounds: int


@dataclass(frozen=True)
class ProjectConfig:
    top_mode: str
    domain_to_stocks: dict[str, list[str]]
    rl: RLConfig
    data: DataConfig
    reward: RewardConfig
    stochastic_control: StochasticControlConfig
    experiments: ExperimentConfig


def _require_keys(payload: dict[str, Any], keys: list[str], scope: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"Missing keys in {scope}: {missing}")


def _as_int_list(values: list[Any]) -> list[int]:
    return [int(v) for v in values]


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    _require_keys(raw, ["top_mode", "domain_to_stocks", "rl", "data"], "root")
    _require_keys(
        raw["rl"],
        [
            "max_domain_hold_steps",
            "max_stock_hold_steps",
            "random_seed",
            "training_steps",
            "stochastic_actions",
        ],
        "rl",
    )
    _require_keys(raw["data"], ["start_date", "end_date", "interval"], "data")

    rl_raw = dict(raw["rl"])
    data_raw = dict(raw["data"])
    reward_raw = dict(raw.get("reward", {}))
    stochastic_raw = dict(raw.get("stochastic_control", {}))
    experiments_raw = dict(raw.get("experiments", {}))

    default_methods = [
        "rl",
        "moe_router",
    ]
    methods = [str(m) for m in experiments_raw.get("methods", experiments_raw.get("modes", default_methods))]
    modes = [str(m) for m in experiments_raw.get("modes", [str(raw["top_mode"])])]
    seeds = _as_int_list(experiments_raw.get("seeds", [int(rl_raw["random_seed"])]))

    return ProjectConfig(
        top_mode=str(raw["top_mode"]),
        domain_to_stocks={
            str(domain): [str(symbol) for symbol in symbols]
            for domain, symbols in dict(raw["domain_to_stocks"]).items()
        },
        rl=RLConfig(
            top_algorithm=str(rl_raw.get("top_algorithm", "ppo")).lower(),
            domain_algorithm=str(rl_raw.get("domain_algorithm", "sac")).lower(),
            top_network_type=str(rl_raw.get("top_network_type", "mlp")).lower(),
            max_domain_hold_steps=int(rl_raw["max_domain_hold_steps"]),
            max_stock_hold_steps=int(rl_raw["max_stock_hold_steps"]),
            min_domain_hold_steps=int(rl_raw.get("min_domain_hold_steps", 2)),
            min_stock_hold_steps=int(rl_raw.get("min_stock_hold_steps", 2)),
            random_seed=int(rl_raw["random_seed"]),
            training_steps=int(rl_raw["training_steps"]),
            stochastic_actions=bool(rl_raw["stochastic_actions"]),
            ensemble_size=max(1, int(rl_raw.get("ensemble_size", 3))),
            hidden_dims=[int(v) for v in rl_raw.get("hidden_dims", [128, 128])],
            ppo_epochs=max(1, int(rl_raw.get("ppo_epochs", 4))),
            ppo_clip=float(rl_raw.get("ppo_clip", 0.2)),
            ppo_lr=float(rl_raw.get("ppo_lr", 3e-4)),
            ppo_gamma=float(rl_raw.get("ppo_gamma", 0.995)),
            ppo_gae_lambda=float(rl_raw.get("ppo_gae_lambda", 0.95)),
            ppo_entropy_coef=float(rl_raw.get("ppo_entropy_coef", 1e-3)),
            transformer_d_model=max(32, int(rl_raw.get("transformer_d_model", 96))),
            transformer_nhead=max(1, int(rl_raw.get("transformer_nhead", 4))),
            transformer_layers=max(1, int(rl_raw.get("transformer_layers", 2))),
            transformer_dropout=float(rl_raw.get("transformer_dropout", 0.1)),
            sac_lr=float(rl_raw.get("sac_lr", 3e-4)),
            sac_gamma=float(rl_raw.get("sac_gamma", 0.995)),
            sac_tau=float(rl_raw.get("sac_tau", 0.005)),
            sac_alpha=float(rl_raw.get("sac_alpha", 0.15)),
            sac_auto_alpha=bool(rl_raw.get("sac_auto_alpha", True)),
            sac_target_entropy=float(rl_raw.get("sac_target_entropy", -2.0)),
            sac_use_dueling=bool(rl_raw.get("sac_use_dueling", True)),
            sac_batch_size=max(8, int(rl_raw.get("sac_batch_size", 64))),
            sac_replay_size=max(256, int(rl_raw.get("sac_replay_size", 20000))),
            sac_n_step=max(1, int(rl_raw.get("sac_n_step", 3))),
            uncertainty_alpha=float(rl_raw.get("uncertainty_alpha", 0.75)),
            router_entropy_coef=float(rl_raw.get("router_entropy_coef", 0.01)),
            router_load_balance_coef=float(rl_raw.get("router_load_balance_coef", 0.03)),
        ),
        data=DataConfig(
            start_date=str(data_raw["start_date"]),
            end_date=str(data_raw["end_date"]),
            interval=str(data_raw["interval"]),
            provider=str(data_raw.get("provider", "auto")),
            lookback=max(2, int(data_raw.get("lookback", 20))),
            train_split=float(data_raw.get("train_split", 0.7)),
            cache_dir=str(data_raw.get("cache_dir", "data_cache")),
            walk_forward_train=max(60, int(data_raw.get("walk_forward_train", 220))),
            walk_forward_test=max(20, int(data_raw.get("walk_forward_test", 80))),
            walk_forward_step=max(10, int(data_raw.get("walk_forward_step", 80))),
            market_symbol=str(data_raw.get("market_symbol", "SPY")),
            rate_symbols=[str(v) for v in data_raw.get("rate_symbols", ["TLT", "SHY"])],
            volatility_symbol=str(data_raw.get("volatility_symbol", "VXX")),
            cross_asset_symbols=[str(v) for v in data_raw.get("cross_asset_symbols", ["GLD", "USO", "UUP"])],
            sector_etfs={
                str(domain): str(symbol)
                for domain, symbol in dict(
                    data_raw.get(
                        "sector_etfs",
                        {
                            "technology": "XLK",
                            "finance": "XLF",
                            "healthcare": "XLV",
                            "energy": "XLE",
                            "industrials": "XLI",
                            "consumer": "XLP",
                        },
                    )
                ).items()
            },
            use_learned_clusters=bool(data_raw.get("use_learned_clusters", True)),
            learned_cluster_count=max(1, int(data_raw.get("learned_cluster_count", len(raw["domain_to_stocks"])))),
        ),
        reward=RewardConfig(
            transaction_cost_bps=float(reward_raw.get("transaction_cost_bps", 5.0)),
            slippage_bps=float(reward_raw.get("slippage_bps", 2.5)),
            turnover_penalty=float(reward_raw.get("turnover_penalty", 0.0015)),
            downside_penalty=float(reward_raw.get("downside_penalty", 2.0)),
            cvar_alpha=float(reward_raw.get("cvar_alpha", 0.1)),
            cvar_penalty=float(reward_raw.get("cvar_penalty", 1.5)),
            reward_window=max(10, int(reward_raw.get("reward_window", 60))),
            rank_loss_coef=float(reward_raw.get("rank_loss_coef", 0.05)),
            domain_entropy_coef=float(reward_raw.get("domain_entropy_coef", 0.02)),
            hold_penalty_coef=float(reward_raw.get("hold_penalty_coef", 0.01)),
            max_reasonable_hold=max(1, int(reward_raw.get("max_reasonable_hold", 8))),
        ),
        stochastic_control=StochasticControlConfig(
            risk_aversion=float(stochastic_raw.get("risk_aversion", 3.0)),
            hold_scale=float(stochastic_raw.get("hold_scale", 1.1)),
            uncertainty_penalty=float(stochastic_raw.get("uncertainty_penalty", 0.35)),
            mean_reversion_speed=float(stochastic_raw.get("mean_reversion_speed", 0.18)),
            max_single_stock_weight=float(stochastic_raw.get("max_single_stock_weight", 0.45)),
            min_domain_allocation=float(stochastic_raw.get("min_domain_allocation", 0.01)),
            min_hold_ratio=float(stochastic_raw.get("min_hold_ratio", 0.25)),
            hold_inertia=float(stochastic_raw.get("hold_inertia", 0.75)),
        ),
        experiments=ExperimentConfig(
            results_dir=str(experiments_raw.get("results_dir", "results")),
            run_name=str(experiments_raw.get("run_name", "hmadrl_continuous_batch")),
            methods=methods,
            modes=modes,
            seeds=seeds,
            fine_tune_rounds=max(1, int(experiments_raw.get("fine_tune_rounds", 2))),
        ),
    )
