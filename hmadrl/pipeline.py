"""Training, evaluation, and artifact export pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig, load_config
from .data_api import AutoMarketDataProvider, StooqProvider, YFinanceProvider
from .factory import build_hierarchical_agent
from .hierarchy import HierarchicalDecision, HierarchicalPortfolioAgent
from .spaces import DomainState, TopLevelState


@dataclass(frozen=True)
class BacktestMetrics:
    num_periods: int
    mean_period_return: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float


@dataclass(frozen=True)
class TrainingResult:
    provider: str
    mode: str
    seed: int
    train_rewards: list[float]
    train_period_returns: list[float]
    test_period_returns: list[float]
    losses: list[dict[str, float]]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    train_dates: list[str]
    test_dates: list[str]
    train_domain_allocations: list[dict[str, float]]
    test_domain_allocations: list[dict[str, float]]
    last_decision: HierarchicalDecision
    output_dir: str | None = None


@dataclass(frozen=True)
class BatchResult:
    provider: str
    batch_dir: str
    runs: list[TrainingResult]


def _choose_provider(config: ProjectConfig):
    mode = config.data.provider.strip().lower()
    if mode == "yfinance":
        return YFinanceProvider(cache_dir=config.data.cache_dir), "yfinance"
    if mode == "stooq":
        return StooqProvider(cache_dir=config.data.cache_dir), "stooq"
    provider = AutoMarketDataProvider(
        providers=[
            StooqProvider(cache_dir=config.data.cache_dir),
            YFinanceProvider(cache_dir=config.data.cache_dir),
        ]
    )
    return provider, "auto(stooq->yfinance)"


def _build_market_frames(
    config: ProjectConfig,
    provider: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted({s for stocks in config.domain_to_stocks.values() for s in stocks})
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}

    for symbol in symbols:
        frame = provider.fetch(
            symbol=symbol,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            interval=config.data.interval,
        )
        price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        closes[symbol] = frame[price_col].rename(symbol)
        volumes[symbol] = frame["Volume"].rename(symbol)

    close_df = pd.concat(closes.values(), axis=1).sort_index().ffill().dropna()
    volume_df = pd.concat(volumes.values(), axis=1).sort_index().ffill()
    volume_df = volume_df.reindex(close_df.index).fillna(0.0)
    return close_df, volume_df


def _build_states_for_date(
    config: ProjectConfig,
    date: pd.Timestamp,
    momentum_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    remaining_horizon_steps: int,
    step_index: int,
    current_domain_allocation: Mapping[str, float],
    current_stock_allocations: Mapping[str, Mapping[str, float]],
) -> tuple[TopLevelState, dict[str, DomainState]]:
    domain_momentum: dict[str, float] = {}
    domain_volatility: dict[str, float] = {}
    domain_liquidity: dict[str, float] = {}
    domain_states: dict[str, DomainState] = {}

    for domain, symbols in config.domain_to_stocks.items():
        stock_momentum = {s: float(momentum_df.at[date, s]) for s in symbols}
        stock_volatility = {s: float(volatility_df.at[date, s]) for s in symbols}
        stock_liquidity = {s: float(liquidity_df.at[date, s]) for s in symbols}

        domain_momentum[domain] = float(np.mean(list(stock_momentum.values())))
        domain_volatility[domain] = float(np.mean(list(stock_volatility.values())))
        domain_liquidity[domain] = float(np.mean(list(stock_liquidity.values())))

        stock_alloc = dict(current_stock_allocations.get(domain, {}))
        if not stock_alloc:
            equal = 1.0 / len(symbols)
            stock_alloc = {s: equal for s in symbols}

        domain_states[domain] = DomainState(
            domain_name=domain,
            stock_momentum=stock_momentum,
            stock_volatility=stock_volatility,
            stock_liquidity=stock_liquidity,
            current_stock_allocation=stock_alloc,
            remaining_domain_steps=remaining_horizon_steps,
            step_index=step_index,
        )

    top_state = TopLevelState(
        domain_momentum=domain_momentum,
        domain_volatility=domain_volatility,
        domain_liquidity=domain_liquidity,
        current_allocation=current_domain_allocation,
        remaining_horizon_steps=remaining_horizon_steps,
        step_index=step_index,
        cash_ratio=0.0,
    )
    return top_state, domain_states


def _portfolio_period_return(
    decision: HierarchicalDecision,
    next_returns_row: pd.Series,
) -> float:
    period_return = 0.0
    for composite_symbol, weight in decision.final_stock_weights.items():
        _, stock = composite_symbol.split(":", 1)
        period_return += weight * float(next_returns_row.get(stock, 0.0))
    return float(period_return)


def _train_step(
    agent: HierarchicalPortfolioAgent,
    top_state: TopLevelState,
    domain_states: Mapping[str, DomainState],
    next_top_state: TopLevelState,
    next_domain_states: Mapping[str, DomainState],
    decision: HierarchicalDecision,
    reward: float,
    done: bool,
) -> dict[str, float]:
    losses: dict[str, float] = {}
    top_manager = agent.top_manager

    if hasattr(top_manager, "remember"):
        top_manager.remember(top_state, decision.top_action, reward, next_top_state, done)
    if hasattr(top_manager, "learn"):
        loss = top_manager.learn()
        if loss is not None:
            losses["top"] = float(loss)
    if hasattr(top_manager, "update_router"):
        top_manager.update_router(reward)

    for domain, action in decision.domain_actions.items():
        manager = agent.domain_managers[domain]
        manager.remember(domain_states[domain], action, reward, next_domain_states[domain], done)
        d_loss = manager.learn()
        if d_loss is not None:
            losses[f"domain:{domain}"] = float(d_loss)
    return losses


def _compute_metrics(period_returns: list[float]) -> BacktestMetrics:
    if not period_returns:
        return BacktestMetrics(
            num_periods=0,
            mean_period_return=0.0,
            cumulative_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
        )

    returns = np.asarray(period_returns, dtype=float)
    curve = np.cumprod(1.0 + returns)
    cumulative_return = float(curve[-1] - 1.0)
    mean_period = float(np.mean(returns))
    annualized_return = float((1.0 + cumulative_return) ** (252.0 / len(returns)) - 1.0)
    annualized_volatility = float(np.std(returns, ddof=0) * np.sqrt(252.0))
    sharpe = float(annualized_return / annualized_volatility) if annualized_volatility > 1e-12 else 0.0
    peaks = np.maximum.accumulate(curve)
    drawdowns = (curve - peaks) / peaks
    max_drawdown = float(abs(drawdowns.min()))

    return BacktestMetrics(
        num_periods=len(period_returns),
        mean_period_return=mean_period,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
    )


def _equity_curve(period_returns: list[float]) -> list[float]:
    equity = [1.0]
    for value in period_returns:
        equity.append(equity[-1] * (1.0 + value))
    return equity


def _prepare_frames(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.Timestamp], str]:
    provider, provider_label = _choose_provider(config)
    probe_symbol = next(iter(next(iter(config.domain_to_stocks.values()))))
    provider.fetch(
        symbol=probe_symbol,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        interval=config.data.interval,
    )
    close_df, volume_df = _build_market_frames(config, provider=provider)

    lookback = max(2, config.data.lookback)
    returns_df = close_df.pct_change()
    momentum_df = close_df.pct_change(lookback)
    volatility_df = returns_df.rolling(lookback).std() * np.sqrt(252.0)
    liquidity_df = volume_df.rolling(lookback).mean()
    next_returns_df = returns_df.shift(-1)

    valid_mask = (
        ~momentum_df.isna().any(axis=1)
        & ~volatility_df.isna().any(axis=1)
        & ~liquidity_df.isna().any(axis=1)
        & ~next_returns_df.isna().any(axis=1)
    )
    valid_dates = list(close_df.index[valid_mask])
    if len(valid_dates) < 10:
        raise ValueError("Not enough clean history after lookback for training/testing")

    return (
        momentum_df,
        volatility_df,
        liquidity_df,
        next_returns_df,
        close_df,
        valid_dates,
        provider_label,
    )


def _run_single(
    config: ProjectConfig,
    mode: str,
    seed: int,
    momentum_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    next_returns_df: pd.DataFrame,
    valid_dates: list[pd.Timestamp],
    provider_label: str,
) -> TrainingResult:
    split = max(0.5, min(0.95, config.data.train_split))
    split_idx = int(len(valid_dates) * split)
    split_idx = max(2, min(split_idx, len(valid_dates) - 2))

    train_dates = valid_dates[:split_idx]
    if config.rl.training_steps > 0:
        train_dates = train_dates[: min(len(train_dates), config.rl.training_steps)]
    test_dates = valid_dates[split_idx:]
    if not test_dates:
        raise ValueError("No test window available; adjust train_split or date range")

    agent = build_hierarchical_agent(
        mode=mode,
        domain_to_stocks=config.domain_to_stocks,
        max_domain_hold_steps=config.rl.max_domain_hold_steps,
        max_stock_hold_steps=config.rl.max_stock_hold_steps,
        stochastic_control=config.stochastic_control,
        seed=seed,
    )

    domain_allocation = {
        domain: 1.0 / len(config.domain_to_stocks) for domain in config.domain_to_stocks
    }
    stock_allocations = {
        domain: {s: 1.0 / len(stocks) for s in stocks}
        for domain, stocks in config.domain_to_stocks.items()
    }

    train_rewards: list[float] = []
    train_returns: list[float] = []
    losses: list[dict[str, float]] = []
    train_domain_allocs: list[dict[str, float]] = []
    test_domain_allocs: list[dict[str, float]] = []
    decision: HierarchicalDecision | None = None

    for step, date in enumerate(train_dates):
        top_state, domain_states = _build_states_for_date(
            config=config,
            date=date,
            momentum_df=momentum_df,
            volatility_df=volatility_df,
            liquidity_df=liquidity_df,
            remaining_horizon_steps=len(train_dates) - step,
            step_index=step,
            current_domain_allocation=domain_allocation,
            current_stock_allocations=stock_allocations,
        )
        decision = agent.decide(
            top_state=top_state,
            domain_states=domain_states,
            stochastic=config.rl.stochastic_actions,
        )
        train_domain_allocs.append(dict(decision.final_domain_weights))
        period_return = _portfolio_period_return(decision, next_returns_df.loc[date])
        train_returns.append(period_return)
        reward = period_return
        train_rewards.append(reward)

        next_domain_allocation = dict(decision.final_domain_weights)
        next_stock_allocations = dict(stock_allocations)
        for domain, action in decision.domain_actions.items():
            next_stock_allocations[domain] = dict(action.stock_weights)

        next_date = train_dates[min(step + 1, len(train_dates) - 1)]
        next_top_state, next_domain_states = _build_states_for_date(
            config=config,
            date=next_date,
            momentum_df=momentum_df,
            volatility_df=volatility_df,
            liquidity_df=liquidity_df,
            remaining_horizon_steps=max(0, len(train_dates) - step - 1),
            step_index=step + 1,
            current_domain_allocation=next_domain_allocation,
            current_stock_allocations=next_stock_allocations,
        )
        done = step == len(train_dates) - 1
        losses.append(
            _train_step(
                agent=agent,
                top_state=top_state,
                domain_states=domain_states,
                next_top_state=next_top_state,
                next_domain_states=next_domain_states,
                decision=decision,
                reward=reward,
                done=done,
            )
        )
        domain_allocation = next_domain_allocation
        stock_allocations = next_stock_allocations

    assert decision is not None
    test_returns: list[float] = []
    for step, date in enumerate(test_dates):
        top_state, domain_states = _build_states_for_date(
            config=config,
            date=date,
            momentum_df=momentum_df,
            volatility_df=volatility_df,
            liquidity_df=liquidity_df,
            remaining_horizon_steps=len(test_dates) - step,
            step_index=step,
            current_domain_allocation=domain_allocation,
            current_stock_allocations=stock_allocations,
        )
        decision = agent.decide(top_state=top_state, domain_states=domain_states, stochastic=False)
        test_domain_allocs.append(dict(decision.final_domain_weights))
        period_return = _portfolio_period_return(decision, next_returns_df.loc[date])
        test_returns.append(period_return)
        domain_allocation = dict(decision.final_domain_weights)
        for domain, action in decision.domain_actions.items():
            stock_allocations[domain] = dict(action.stock_weights)

    return TrainingResult(
        provider=provider_label,
        mode=mode,
        seed=seed,
        train_rewards=train_rewards,
        train_period_returns=train_returns,
        test_period_returns=test_returns,
        losses=losses,
        train_metrics=_compute_metrics(train_returns),
        test_metrics=_compute_metrics(test_returns),
        train_dates=[str(pd.Timestamp(d).date()) for d in train_dates],
        test_dates=[str(pd.Timestamp(d).date()) for d in test_dates],
        train_domain_allocations=train_domain_allocs,
        test_domain_allocations=test_domain_allocs,
        last_decision=decision,
    )


def _plot_rewards(train_rewards: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_rewards, label="Train reward", color="#0f766e")
    ax.set_title("Training Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_equity(train_returns: list[float], test_returns: list[float], path: Path) -> None:
    train_curve = _equity_curve(train_returns)
    test_curve = _equity_curve(test_returns)
    combined = train_curve + [train_curve[-1] * x for x in test_curve[1:]]
    split_idx = len(train_curve) - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(combined, color="#1d4ed8", label="Portfolio equity")
    ax.axvline(split_idx, color="#dc2626", linestyle="--", label="Train/Test split")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_drawdown(train_returns: list[float], test_returns: list[float], path: Path) -> None:
    train_curve = np.array(_equity_curve(train_returns))
    test_curve = np.array(_equity_curve(test_returns))
    combined = np.concatenate([train_curve, train_curve[-1] * test_curve[1:]])
    peaks = np.maximum.accumulate(combined)
    drawdown = (combined - peaks) / peaks

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(drawdown)), drawdown, color="#b91c1c", alpha=0.35)
    ax.plot(drawdown, color="#7f1d1d")
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_domain_allocations(
    allocations: list[dict[str, float]],
    title: str,
    path: Path,
) -> None:
    if not allocations:
        return
    frame = pd.DataFrame(allocations).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        range(len(frame)),
        [frame[col].values for col in frame.columns],
        labels=list(frame.columns),
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _metrics_to_dict(metrics: BacktestMetrics) -> dict[str, float | int]:
    return {
        "num_periods": metrics.num_periods,
        "mean_period_return": metrics.mean_period_return,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
    }


def save_training_result(result: TrainingResult, output_dir: str | Path) -> TrainingResult:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    train_equity = _equity_curve(result.train_period_returns)
    test_equity = _equity_curve(result.test_period_returns)
    pd.DataFrame(
        {
            "date": result.train_dates[: len(result.train_period_returns)],
            "period_return": result.train_period_returns,
            "reward": result.train_rewards[: len(result.train_period_returns)],
            "equity": train_equity[1:],
        }
    ).to_csv(path / "train_returns.csv", index=False)
    pd.DataFrame(
        {
            "date": result.test_dates[: len(result.test_period_returns)],
            "period_return": result.test_period_returns,
            "equity": test_equity[1:],
        }
    ).to_csv(path / "test_returns.csv", index=False)
    if result.losses:
        pd.DataFrame(result.losses).to_csv(path / "losses.csv", index=False)
    pd.DataFrame(result.train_domain_allocations).to_csv(path / "train_domain_allocations.csv", index=False)
    pd.DataFrame(result.test_domain_allocations).to_csv(path / "test_domain_allocations.csv", index=False)

    summary = {
        "provider": result.provider,
        "mode": result.mode,
        "seed": result.seed,
        "train_metrics": _metrics_to_dict(result.train_metrics),
        "test_metrics": _metrics_to_dict(result.test_metrics),
        "last_top_hold_steps": result.last_decision.top_action.hold_steps,
        "last_top_weights": dict(result.last_decision.top_action.domain_weights),
        "last_final_domain_weights": dict(result.last_decision.final_domain_weights),
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_rewards(result.train_rewards, path / "reward_curve.png")
    _plot_equity(result.train_period_returns, result.test_period_returns, path / "equity_curve.png")
    _plot_drawdown(result.train_period_returns, result.test_period_returns, path / "drawdown_curve.png")
    _plot_domain_allocations(
        allocations=result.train_domain_allocations,
        title="Train Domain Allocation",
        path=path / "train_domain_allocation.png",
    )
    _plot_domain_allocations(
        allocations=result.test_domain_allocations,
        title="Test Domain Allocation",
        path=path / "test_domain_allocation.png",
    )

    return TrainingResult(
        provider=result.provider,
        mode=result.mode,
        seed=result.seed,
        train_rewards=result.train_rewards,
        train_period_returns=result.train_period_returns,
        test_period_returns=result.test_period_returns,
        losses=result.losses,
        train_metrics=result.train_metrics,
        test_metrics=result.test_metrics,
        train_dates=result.train_dates,
        test_dates=result.test_dates,
        train_domain_allocations=result.train_domain_allocations,
        test_domain_allocations=result.test_domain_allocations,
        last_decision=result.last_decision,
        output_dir=str(path),
    )


def _plot_batch_cumulative_returns(results: list[TrainingResult], path: Path) -> None:
    labels = [f"{r.mode}-s{r.seed}" for r in results]
    train_vals = [r.train_metrics.cumulative_return for r in results]
    test_vals = [r.test_metrics.cumulative_return for r in results]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
    ax.bar(x - width / 2, train_vals, width=width, label="Train", color="#0f766e")
    ax.bar(x + width / 2, test_vals, width=width, label="Test", color="#1d4ed8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns By Run")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_training(config_path: str | Path) -> TrainingResult:
    config = load_config(config_path)
    (
        momentum_df,
        volatility_df,
        liquidity_df,
        next_returns_df,
        _,
        valid_dates,
        provider_label,
    ) = _prepare_frames(config)
    return _run_single(
        config=config,
        mode=config.top_mode,
        seed=config.rl.random_seed,
        momentum_df=momentum_df,
        volatility_df=volatility_df,
        liquidity_df=liquidity_df,
        next_returns_df=next_returns_df,
        valid_dates=valid_dates,
        provider_label=provider_label,
    )


def run_multiple_experiments(config_path: str | Path) -> BatchResult:
    config = load_config(config_path)
    (
        momentum_df,
        volatility_df,
        liquidity_df,
        next_returns_df,
        _,
        valid_dates,
        provider_label,
    ) = _prepare_frames(config)

    modes = config.experiments.modes or [config.top_mode]
    seeds = config.experiments.seeds or [config.rl.random_seed]
    run_grid = [(mode, seed) for mode in modes for seed in seeds]
    if not run_grid:
        run_grid = [(config.top_mode, config.rl.random_seed)]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(config.experiments.results_dir) / f"{config.experiments.run_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs: list[TrainingResult] = []
    summary_rows: list[dict[str, Any]] = []
    for index, (mode, seed) in enumerate(run_grid):
        print(f"[{index + 1}/{len(run_grid)}] running mode={mode}, seed={seed}")
        run_result = _run_single(
            config=config,
            mode=mode,
            seed=seed,
            momentum_df=momentum_df,
            volatility_df=volatility_df,
            liquidity_df=liquidity_df,
            next_returns_df=next_returns_df,
            valid_dates=valid_dates,
            provider_label=provider_label,
        )
        run_dir = batch_dir / f"run_{index:02d}_{mode}_seed{seed}"
        run_result = save_training_result(run_result, run_dir)
        print(f"[{index + 1}/{len(run_grid)}] saved artifacts: {run_dir}")
        runs.append(run_result)
        summary_rows.append(
            {
                "run": run_dir.name,
                "mode": run_result.mode,
                "seed": run_result.seed,
                "train_cumulative_return": run_result.train_metrics.cumulative_return,
                "test_cumulative_return": run_result.test_metrics.cumulative_return,
                "train_sharpe": run_result.train_metrics.sharpe,
                "test_sharpe": run_result.test_metrics.sharpe,
                "train_max_drawdown": run_result.train_metrics.max_drawdown,
                "test_max_drawdown": run_result.test_metrics.max_drawdown,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(batch_dir / "batch_summary.csv", index=False)
    _plot_batch_cumulative_returns(runs, batch_dir / "batch_cumulative_returns.png")
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    return BatchResult(
        provider=provider_label,
        batch_dir=str(batch_dir),
        runs=runs,
    )
