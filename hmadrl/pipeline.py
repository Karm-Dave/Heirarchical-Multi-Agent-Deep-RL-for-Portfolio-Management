"""End-to-end helper to run the hierarchy with API-backed data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import ProjectConfig, load_config
from .data_api import (
    AutoMarketDataProvider,
    StooqProvider,
    YFinanceProvider,
)
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
    train_rewards: list[float]
    losses: list[dict[str, float]]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    last_decision: HierarchicalDecision


def _choose_provider(config: ProjectConfig):
    mode = config.data.provider.strip().lower()
    if mode == "yfinance":
        return YFinanceProvider(cache_dir=config.data.cache_dir), "yfinance"
    if mode == "stooq":
        return StooqProvider(cache_dir=config.data.cache_dir), "stooq"
    provider = AutoMarketDataProvider(
        providers=[
            YFinanceProvider(cache_dir=config.data.cache_dir),
            StooqProvider(cache_dir=config.data.cache_dir),
        ]
    )
    return provider, "auto(yfinance->stooq)"


def _build_market_frames(
    config: ProjectConfig,
    provider,
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


def run_training(config_path: str | Path) -> TrainingResult:
    config: ProjectConfig = load_config(config_path)
    provider, provider_label = _choose_provider(config)
    # Provider probe ensures we fail fast if data access is unavailable.
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
        mode=config.top_mode,
        domain_to_stocks=config.domain_to_stocks,
        max_domain_hold_steps=config.rl.max_domain_hold_steps,
        max_stock_hold_steps=config.rl.max_stock_hold_steps,
        seed=config.rl.random_seed,
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
        period_return = _portfolio_period_return(decision, next_returns_df.loc[date])
        train_returns.append(period_return)
        reward = period_return
        train_rewards.append(reward)

        next_domain_allocation = dict(decision.top_action.domain_weights)
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
        period_return = _portfolio_period_return(decision, next_returns_df.loc[date])
        test_returns.append(period_return)
        domain_allocation = dict(decision.top_action.domain_weights)
        for domain, action in decision.domain_actions.items():
            stock_allocations[domain] = dict(action.stock_weights)

    return TrainingResult(
        provider=provider_label,
        train_rewards=train_rewards,
        losses=losses,
        train_metrics=_compute_metrics(train_returns),
        test_metrics=_compute_metrics(test_returns),
        last_decision=decision,
    )
