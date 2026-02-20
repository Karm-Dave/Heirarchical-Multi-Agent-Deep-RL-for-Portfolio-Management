"""Training/evaluation pipeline with walk-forward and risk-aware rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig, RewardConfig, load_config
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
    downside_deviation: float = 0.0
    cvar: float = 0.0


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
    reward_components: list[dict[str, float | str | int]] = field(default_factory=list)
    walk_forward_windows: list[dict[str, Any]] = field(default_factory=list)
    regime_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)
    output_dir: str | None = None


@dataclass(frozen=True)
class BatchResult:
    provider: str
    batch_dir: str
    runs: list[TrainingResult]


@dataclass(frozen=True)
class PreparedData:
    domain_to_stocks: dict[str, list[str]]
    momentum_df: pd.DataFrame
    volatility_df: pd.DataFrame
    liquidity_df: pd.DataFrame
    stock_feature_frames: dict[str, pd.DataFrame]
    stock_feature_names: list[str]
    next_returns_df: pd.DataFrame
    global_feature_df: pd.DataFrame
    domain_factor_frames: dict[str, pd.DataFrame]
    valid_dates: list[pd.Timestamp]
    regimes: pd.Series
    provider_label: str


def _choose_provider(config: ProjectConfig):
    mode = config.data.provider.strip().lower()
    if mode == "yfinance":
        return YFinanceProvider(cache_dir=config.data.cache_dir), "yfinance"
    if mode == "stooq":
        return StooqProvider(cache_dir=config.data.cache_dir), "stooq"
    return (
        AutoMarketDataProvider(
            providers=[
                StooqProvider(cache_dir=config.data.cache_dir),
                YFinanceProvider(cache_dir=config.data.cache_dir),
            ]
        ),
        "auto(stooq->yfinance)",
    )


def _extract_ohlcv(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    close = frame[price_col].astype(float)
    volume = frame["Volume"].astype(float) if "Volume" in frame.columns else pd.Series(0.0, index=frame.index)
    high = frame["High"].astype(float) if "High" in frame.columns else close
    low = frame["Low"].astype(float) if "Low" in frame.columns else close
    return close, volume, high, low


def _build_market_frames(
    config: ProjectConfig,
    provider: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_symbols = sorted({s for stocks in config.domain_to_stocks.values() for s in stocks})
    factors = [config.data.market_symbol, config.data.volatility_symbol]
    factors += list(config.data.rate_symbols) + list(config.data.cross_asset_symbols) + list(config.data.sector_etfs.values())
    factors = list(dict.fromkeys([s for s in factors if s]))

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    spreads: dict[str, pd.Series] = {}
    for symbol in stock_symbols:
        frame = provider.fetch(symbol=symbol, start_date=config.data.start_date, end_date=config.data.end_date, interval=config.data.interval)
        close, volume, high, low = _extract_ohlcv(frame)
        closes[symbol] = close.rename(symbol)
        volumes[symbol] = volume.rename(symbol)
        spread_proxy = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        spreads[symbol] = spread_proxy.rename(symbol)

    close_df = pd.concat(closes.values(), axis=1).sort_index().ffill().dropna()
    volume_df = pd.concat(volumes.values(), axis=1).sort_index().ffill().reindex(close_df.index).fillna(0.0)
    spread_df = pd.concat(spreads.values(), axis=1).sort_index().ffill().reindex(close_df.index).fillna(0.0)

    factor_closes: dict[str, pd.Series] = {}
    for symbol in factors:
        try:
            frame = provider.fetch(symbol=symbol, start_date=config.data.start_date, end_date=config.data.end_date, interval=config.data.interval)
            close, _, _, _ = _extract_ohlcv(frame)
            factor_closes[symbol] = close.rename(symbol)
        except Exception:
            continue

    factor_df = pd.concat(factor_closes.values(), axis=1).sort_index().ffill().bfill() if factor_closes else pd.DataFrame(index=close_df.index)
    factor_df = factor_df.reindex(close_df.index).ffill().bfill()
    fallback = close_df.mean(axis=1)
    for symbol in factors:
        if symbol not in factor_df.columns:
            factor_df[symbol] = fallback
    return close_df, volume_df, factor_df.sort_index(axis=1), spread_df


def _rolling_z(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std().replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _classify_regime(ret: pd.Series, vol: pd.Series) -> pd.Series:
    q50, q75 = float(vol.quantile(0.5)), float(vol.quantile(0.75))
    labels = pd.Series(index=ret.index, dtype=object)
    for date in ret.index:
        r = float(ret.loc[date])
        v = float(vol.loc[date])
        if r >= 0.0 and v <= q50:
            labels.loc[date] = "bull"
        elif r < 0.0 and v >= q75:
            labels.loc[date] = "bear"
        elif v >= q75:
            labels.loc[date] = "volatile"
        else:
            labels.loc[date] = "sideways"
    return labels


def _learned_cluster_map(
    domain_to_stocks: Mapping[str, list[str]],
    returns_df: pd.DataFrame,
    cluster_count: int,
) -> dict[str, list[str]]:
    stocks = sorted({s for symbols in domain_to_stocks.values() for s in symbols})
    if not stocks:
        return {}
    if len(stocks) <= 2:
        return {"cluster_00": stocks}

    k = max(1, min(int(cluster_count), len(stocks)))
    corr = returns_df[stocks].corr().fillna(0.0)
    dist = 1.0 - corr
    variances = returns_df[stocks].var().fillna(0.0)
    first_seed = str(variances.sort_values(ascending=False).index[0])
    seeds = [first_seed]
    while len(seeds) < k:
        best_stock = None
        best_score = -1.0
        for stock in stocks:
            if stock in seeds:
                continue
            nearest_seed = min(float(dist.loc[stock, seed]) for seed in seeds)
            if nearest_seed > best_score:
                best_score = nearest_seed
                best_stock = stock
        if best_stock is None:
            break
        seeds.append(str(best_stock))

    cluster_assignments: dict[str, list[str]] = {f"cluster_{i:02d}": [] for i in range(len(seeds))}
    for stock in stocks:
        best_cluster = 0
        best_dist = float("inf")
        for idx, seed in enumerate(seeds):
            d = float(dist.loc[stock, seed])
            if d < best_dist:
                best_dist = d
                best_cluster = idx
        cluster_assignments[f"cluster_{best_cluster:02d}"].append(stock)

    # Ensure each cluster has at least one stock.
    empty = [name for name, symbols in cluster_assignments.items() if not symbols]
    if empty:
        donors = sorted(cluster_assignments.items(), key=lambda x: len(x[1]), reverse=True)
        for target in empty:
            for donor_name, donor_symbols in donors:
                if len(donor_symbols) > 1:
                    cluster_assignments[target].append(donor_symbols.pop())
                    break

    # Drop any empty clusters that still remain.
    return {name: symbols for name, symbols in cluster_assignments.items() if symbols}


def _build_features(
    config: ProjectConfig,
    domain_to_stocks: Mapping[str, list[str]],
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.Series,
]:
    lb = max(2, config.data.lookback)
    ret = close_df.pct_change().replace([np.inf, -np.inf], np.nan)
    next_ret = ret.shift(-1)

    momentum_horizons = [5, 20, 60, 120, 252]
    volatility_horizons = [10, 30, 90, 252]
    momentum_map = {
        h: close_df.pct_change(h).replace([np.inf, -np.inf], np.nan)
        for h in momentum_horizons
    }
    vol_map = {
        h: (ret.rolling(h).std() * np.sqrt(252.0)).replace([np.inf, -np.inf], np.nan)
        for h in volatility_horizons
    }

    vol_30 = vol_map[30]
    vol_of_vol_30 = vol_30.rolling(30).std().replace([np.inf, -np.inf], np.nan)
    vol_10 = vol_map[10]
    vol_90 = vol_map[90]
    vol_252 = vol_map[252]
    momentum = (
        0.15 * momentum_map[5]
        + 0.30 * momentum_map[20]
        + 0.30 * momentum_map[60]
        + 0.15 * momentum_map[120]
        + 0.10 * momentum_map[252]
    ).replace([np.inf, -np.inf], np.nan)
    volatility = (
        0.20 * vol_10 + 0.35 * vol_30 + 0.30 * vol_90 + 0.15 * vol_252
    ).replace([np.inf, -np.inf], np.nan)

    vol_mean_30 = volume_df.rolling(30).mean()
    vol_std_20 = volume_df.rolling(20).std().replace(0.0, np.nan)
    volume_z_20 = ((volume_df - volume_df.rolling(20).mean()) / vol_std_20).replace([np.inf, -np.inf], np.nan)
    turnover_ratio_30 = (volume_df / vol_mean_30).replace([np.inf, -np.inf], np.nan)
    amihud_20 = (
        ret.abs() / (volume_df * close_df).replace(0.0, np.nan)
    ).rolling(20).mean().replace([np.inf, -np.inf], np.nan)
    spread_20 = spread_df.rolling(20).mean().replace([np.inf, -np.inf], np.nan)
    liquidity = (
        np.log1p(vol_mean_30).replace([np.inf, -np.inf], np.nan)
        / (1.0 + amihud_20 * 1e6)
    ).replace([np.inf, -np.inf], np.nan)

    ema20 = close_df.ewm(span=20, adjust=False).mean()
    ema60 = close_df.ewm(span=60, adjust=False).mean()
    ema120 = close_df.ewm(span=120, adjust=False).mean()
    ema_gap_20_60 = (ema20 / ema60 - 1.0).replace([np.inf, -np.inf], np.nan)
    ema_gap_60_120 = (ema60 / ema120 - 1.0).replace([np.inf, -np.inf], np.nan)

    factor_ret = factor_df.pct_change().replace([np.inf, -np.inf], np.nan)
    m = config.data.market_symbol
    v = config.data.volatility_symbol
    mret = factor_ret[m]
    mvol = (mret.rolling(lb).std() * np.sqrt(252.0)).replace([np.inf, -np.inf], np.nan)
    regimes = _classify_regime(mret, mvol)
    market_curve = (1.0 + mret.fillna(0.0)).cumprod()
    market_peaks = np.maximum.accumulate(market_curve)
    market_drawdown = (market_curve - market_peaks) / market_peaks

    above_50 = (close_df > close_df.rolling(50).mean()).mean(axis=1)
    above_200 = (close_df > close_df.rolling(200).mean()).mean(axis=1)
    adv = (ret > 0).sum(axis=1).astype(float)
    dec = (ret < 0).sum(axis=1).astype(float)
    advance_decline = adv / (dec + 1e-6)
    market_dispersion = ret.std(axis=1)

    g = pd.DataFrame(index=close_df.index)
    g["market_ret"] = mret
    g["market_vol"] = mvol
    g["market_trend"] = (factor_df[m] / factor_df[m].rolling(lb).mean()) - 1.0
    g["market_drawdown"] = market_drawdown
    g["market_max_drawdown_90d"] = market_drawdown.rolling(90).min()
    g["market_skew_30d"] = mret.rolling(30).skew()
    g["market_kurtosis_30d"] = mret.rolling(30).kurt()
    g["market_tail_risk_30d"] = mret.rolling(30).quantile(0.05)
    g["breadth_above_50dma"] = above_50
    g["breadth_above_200dma"] = above_200
    g["advance_decline_ratio"] = advance_decline
    g["market_dispersion"] = market_dispersion
    g["vol_to_equity_ratio"] = (factor_df[v] / factor_df[m]).replace([np.inf, -np.inf], np.nan)
    g["vol_momentum_20d"] = factor_df[v].pct_change(20)
    g[f"vol_ret_{v}"] = factor_ret[v]
    g[f"vol_z_{v}"] = _rolling_z(factor_df[v], lb)

    bond_symbol = config.data.rate_symbols[0] if config.data.rate_symbols else m
    g["equity_bond_ratio"] = (factor_df[m] / factor_df[bond_symbol]).replace([np.inf, -np.inf], np.nan)
    if len(config.data.rate_symbols) >= 2:
        left = factor_ret[config.data.rate_symbols[0]]
        right = factor_ret[config.data.rate_symbols[1]]
        g["yield_curve_slope_proxy"] = left - right
    else:
        g["yield_curve_slope_proxy"] = factor_ret[bond_symbol]

    for s in config.data.rate_symbols:
        g[f"rate_ret_{s}"] = factor_ret[s]
    for s in config.data.cross_asset_symbols:
        g[f"cross_ret_{s}"] = factor_ret[s]

    g["gold_momentum_60d"] = factor_df["GLD"].pct_change(60) if "GLD" in factor_df.columns else 0.0
    g["oil_momentum_60d"] = factor_df["USO"].pct_change(60) if "USO" in factor_df.columns else 0.0
    market_ema20 = factor_df[m].ewm(span=20, adjust=False).mean()
    market_ema60 = factor_df[m].ewm(span=60, adjust=False).mean()
    market_ema200 = factor_df[m].ewm(span=200, adjust=False).mean()
    g["market_ema_gap_20_60"] = (market_ema20 / market_ema60 - 1.0).replace([np.inf, -np.inf], np.nan)
    g["market_ema_gap_60_200"] = (market_ema60 / market_ema200 - 1.0).replace([np.inf, -np.inf], np.nan)
    g["market_trend_slope_60"] = factor_df[m].pct_change(60) / 60.0

    for label in ("bull", "bear", "volatile", "sideways"):
        g[f"regime_{label}"] = (regimes == label).astype(float)

    m_mom_20 = factor_df[m].pct_change(20).replace([np.inf, -np.inf], np.nan)
    ret_mean = ret.mean(axis=1)
    ret_std = ret.std(axis=1).replace(0.0, np.nan)
    ret_zscore = ret.sub(ret_mean, axis=0).div(ret_std, axis=0).replace([np.inf, -np.inf], np.nan)

    stock_feature_frames: dict[str, pd.DataFrame] = {
        "mom_5d": momentum_map[5],
        "mom_20d": momentum_map[20],
        "mom_60d": momentum_map[60],
        "mom_120d": momentum_map[120],
        "mom_252d": momentum_map[252],
        "vol_10d": vol_10,
        "vol_30d": vol_30,
        "vol_90d": vol_90,
        "vol_252d": vol_252,
        "vol_of_vol_30d": vol_of_vol_30,
        "amihud_20d": amihud_20,
        "turnover_ratio_30d": turnover_ratio_30,
        "spread_proxy_20d": spread_20,
        "volume_zscore_20d": volume_z_20,
        "ema_gap_20_60": ema_gap_20_60,
        "ema_gap_60_120": ema_gap_60_120,
        "relative_strength_vs_market_20d": momentum_map[20].sub(m_mom_20, axis=0),
        "zscore_return_cross_sectional": ret_zscore,
    }
    stock_feature_names = sorted(stock_feature_frames.keys())

    domain_factors: dict[str, pd.DataFrame] = {}
    for domain, symbols in domain_to_stocks.items():
        subset = ret[symbols]
        dret = subset.mean(axis=1)
        dcurve = (1.0 + dret.fillna(0.0)).cumprod()
        ddrawdown = (dcurve - np.maximum.accumulate(dcurve)) / np.maximum.accumulate(dcurve)
        sector = config.data.sector_etfs.get(domain, m)
        sret = factor_ret[sector] if sector in factor_ret.columns else dret

        top_bottom = subset.apply(
            lambda row: float(
                row.sort_values().tail(max(1, len(symbols) // 5)).mean()
                - row.sort_values().head(max(1, len(symbols) // 5)).mean()
            ),
            axis=1,
        )
        intra_corr = 1.0 - (
            subset.std(axis=1) / (subset.abs().mean(axis=1) + 1e-6)
        )
        f = pd.DataFrame(index=close_df.index)
        f["sector_ret"] = sret
        f["relative_strength"] = sret - mret
        f["beta_proxy"] = dret.rolling(lb).corr(mret).replace([np.inf, -np.inf], np.nan)
        f["dispersion"] = subset.std(axis=1)
        f["cluster_momentum"] = close_df[symbols].mean(axis=1).pct_change(20)
        f["cluster_volatility"] = subset.mean(axis=1).rolling(30).std() * np.sqrt(252.0)
        f["cluster_drawdown"] = ddrawdown
        sharpe_roll = dret.rolling(30).mean() / (dret.rolling(30).std().replace(0.0, np.nan))
        f["cluster_sharpe_rolling"] = sharpe_roll * np.sqrt(252.0)
        f["cluster_correlation_to_market"] = dret.rolling(30).corr(mret).replace([np.inf, -np.inf], np.nan)
        f["cross_sectional_std_returns"] = subset.std(axis=1)
        f["top_minus_bottom_quintile_return"] = top_bottom
        f["intra_cluster_correlation"] = intra_corr
        domain_factors[domain] = f.replace([np.inf, -np.inf], np.nan)

    return (
        momentum,
        volatility,
        liquidity,
        stock_feature_frames,
        stock_feature_names,
        next_ret,
        g.replace([np.inf, -np.inf], np.nan),
        domain_factors,
        regimes,
    )


def _prepare_data(config: ProjectConfig) -> PreparedData:
    provider, provider_label = _choose_provider(config)
    probe = next(iter(next(iter(config.domain_to_stocks.values()))))
    provider.fetch(symbol=probe, start_date=config.data.start_date, end_date=config.data.end_date, interval=config.data.interval)
    close_df, volume_df, factor_df, spread_df = _build_market_frames(config, provider)
    provisional_returns = close_df.pct_change().replace([np.inf, -np.inf], np.nan)
    if config.data.use_learned_clusters:
        learned_clusters = _learned_cluster_map(
            config.domain_to_stocks,
            provisional_returns,
            cluster_count=config.data.learned_cluster_count,
        )
        domain_map = learned_clusters or {
            str(name): list(symbols) for name, symbols in config.domain_to_stocks.items()
        }
    else:
        domain_map = {
            str(name): list(symbols) for name, symbols in config.domain_to_stocks.items()
        }

    (
        momentum,
        volatility,
        liquidity,
        stock_feature_frames,
        stock_feature_names,
        next_ret,
        global_df,
        domain_factors,
        regimes,
    ) = _build_features(
        config,
        domain_map,
        close_df,
        volume_df,
        spread_df,
        factor_df,
    )
    mask = (
        ~momentum.isna().any(axis=1)
        & ~volatility.isna().any(axis=1)
        & ~liquidity.isna().any(axis=1)
        & ~next_ret.isna().any(axis=1)
        & ~global_df.isna().any(axis=1)
        & ~regimes.isna()
    )
    for frame in domain_factors.values():
        mask &= ~frame.isna().any(axis=1)
    for frame in stock_feature_frames.values():
        mask &= ~frame.isna().any(axis=1)
    dates = list(close_df.index[mask])
    if len(dates) < 80:
        raise ValueError("Not enough clean history for walk-forward training/testing")
    return PreparedData(
        domain_to_stocks=domain_map,
        momentum_df=momentum,
        volatility_df=volatility,
        liquidity_df=liquidity,
        stock_feature_frames=stock_feature_frames,
        stock_feature_names=stock_feature_names,
        next_returns_df=next_ret,
        global_feature_df=global_df,
        domain_factor_frames=domain_factors,
        valid_dates=dates,
        regimes=regimes,
        provider_label=provider_label,
    )


def _build_states(
    prepared: PreparedData,
    date: pd.Timestamp,
    remaining: int,
    step_idx: int,
    domain_alloc: Mapping[str, float],
    stock_allocs: Mapping[str, Mapping[str, float]],
) -> tuple[TopLevelState, dict[str, DomainState]]:
    dm, dv, dl = {}, {}, {}
    domain_states: dict[str, DomainState] = {}
    global_features = {
        c: float(prepared.global_feature_df.at[date, c]) for c in sorted(prepared.global_feature_df.columns)
    }
    for domain, symbols in prepared.domain_to_stocks.items():
        sm = {s: float(prepared.momentum_df.at[date, s]) for s in symbols}
        sv = {s: float(prepared.volatility_df.at[date, s]) for s in symbols}
        sl = {s: float(prepared.liquidity_df.at[date, s]) for s in symbols}
        sf = {
            c: float(prepared.domain_factor_frames[domain].at[date, c])
            for c in sorted(prepared.domain_factor_frames[domain].columns)
        }

        mom_series = pd.Series(sm, dtype=float)
        vol_series = pd.Series(sv, dtype=float)
        mom_rank = mom_series.rank(pct=True, method="average").to_dict()
        vol_rank = vol_series.rank(pct=True, method="average").to_dict()
        mean_mom = float(mom_series.mean())
        stock_features: dict[str, dict[str, float]] = {}
        for stock in symbols:
            base_features = {
                name: float(prepared.stock_feature_frames[name].at[date, stock])
                for name in prepared.stock_feature_names
            }
            base_features["stock_mom_rank_in_cluster"] = float(mom_rank.get(stock, 0.0))
            base_features["stock_vol_rank"] = float(vol_rank.get(stock, 0.0))
            base_features["relative_momentum"] = float(sm.get(stock, 0.0) - mean_mom)
            stock_features[stock] = base_features

        dm[domain] = float(sf.get("cluster_momentum", float(np.mean(list(sm.values())))))
        dv[domain] = float(sf.get("cluster_volatility", float(np.mean(list(sv.values())))))
        dl[domain] = float(np.mean(list(sl.values())))
        alloc = dict(stock_allocs.get(domain, {}))
        if not alloc:
            eq = 1.0 / len(symbols)
            alloc = {s: eq for s in symbols}
        domain_states[domain] = DomainState(
            domain_name=domain,
            stock_momentum=sm,
            stock_volatility=sv,
            stock_liquidity=sl,
            current_stock_allocation=alloc,
            remaining_domain_steps=remaining,
            step_index=step_idx,
            domain_factors=sf,
            stock_features=stock_features,
        )
    top_state = TopLevelState(
        domain_momentum=dm,
        domain_volatility=dv,
        domain_liquidity=dl,
        current_allocation=domain_alloc,
        remaining_horizon_steps=remaining,
        step_index=step_idx,
        cash_ratio=0.0,
        global_features=global_features,
    )
    return top_state, domain_states


def _portfolio_return(decision: HierarchicalDecision, next_row: pd.Series) -> float:
    value = 0.0
    for comp, w in decision.final_stock_weights.items():
        _, stock = comp.split(":", 1)
        value += float(w) * float(next_row.get(stock, 0.0))
    return float(value)


def _turnover(prev_w: Mapping[str, float], new_w: Mapping[str, float]) -> float:
    keys = set(prev_w) | set(new_w)
    return float(sum(abs(float(new_w.get(k, 0.0)) - float(prev_w.get(k, 0.0))) for k in keys))


def _compute_cvar(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    a = max(1e-3, min(0.5, float(alpha)))
    arr = np.sort(np.asarray(values, dtype=float))
    cutoff = max(1, int(math.ceil(a * len(arr))))
    return float(np.mean(arr[:cutoff]))


def _reward(
    gross: float,
    prev_w: Mapping[str, float],
    new_w: Mapping[str, float],
    hist: list[float],
    cfg: RewardConfig,
) -> dict[str, float]:
    t = _turnover(prev_w, new_w)
    tx = t * (cfg.transaction_cost_bps / 10000.0)
    sl = t * (cfg.slippage_bps / 10000.0)
    tp = cfg.turnover_penalty * t
    down = cfg.downside_penalty * max(0.0, -gross)
    win = (hist + [gross])[-max(5, int(cfg.reward_window)) :]
    cvar = _compute_cvar(win, cfg.cvar_alpha)
    cp = cfg.cvar_penalty * max(0.0, -cvar)
    r = gross - tx - sl - tp - down - cp
    return {
        "gross_return": float(gross),
        "turnover": float(t),
        "transaction_cost": float(tx),
        "slippage_cost": float(sl),
        "turnover_penalty": float(tp),
        "downside_penalty": float(down),
        "cvar": float(cvar),
        "cvar_penalty": float(cp),
        "reward": float(r),
    }


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
    top = agent.top_manager
    if hasattr(top, "remember"):
        top.remember(top_state, decision.top_action, reward, next_top_state, done)
    if hasattr(top, "learn"):
        top_loss = top.learn()
        if top_loss is not None:
            losses["top"] = float(top_loss)
    if hasattr(top, "update_router"):
        top.update_router(reward)
    for domain, action in decision.domain_actions.items():
        manager = agent.domain_managers[domain]
        manager.remember(domain_states[domain], action, reward, next_domain_states[domain], done)
        dloss = manager.learn()
        if dloss is not None:
            losses[f"domain:{domain}"] = float(dloss)
    return losses


def _metrics(values: list[float], cvar_alpha: float = 0.1) -> BacktestMetrics:
    if not values:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    curve = np.cumprod(1.0 + arr)
    cum = float(curve[-1] - 1.0)
    mean = float(np.mean(arr))
    ann = float((1.0 + cum) ** (252.0 / len(arr)) - 1.0)
    vol = float(np.std(arr, ddof=0) * np.sqrt(252.0))
    sharpe = float(ann / vol) if vol > 1e-12 else 0.0
    peaks = np.maximum.accumulate(curve)
    mdd = float(abs(((curve - peaks) / peaks).min()))
    downside = float(np.sqrt(np.mean(np.square(np.minimum(arr, 0.0)))) * np.sqrt(252.0))
    cvar = _compute_cvar(values, cvar_alpha)
    return BacktestMetrics(len(values), mean, cum, ann, vol, sharpe, mdd, downside, cvar)


def _metrics_dict(m: BacktestMetrics) -> dict[str, float | int]:
    return {
        "num_periods": m.num_periods,
        "mean_period_return": m.mean_period_return,
        "cumulative_return": m.cumulative_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe": m.sharpe,
        "max_drawdown": m.max_drawdown,
        "downside_deviation": m.downside_deviation,
        "cvar": m.cvar,
    }


def _equity(values: list[float]) -> list[float]:
    out = [1.0]
    for r in values:
        out.append(out[-1] * (1.0 + r))
    return out


def _init_allocations(domain_to_stocks: Mapping[str, list[str]]) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    da = {d: 1.0 / len(domain_to_stocks) for d in domain_to_stocks}
    sa = {d: {s: 1.0 / len(stocks) for s in stocks} for d, stocks in domain_to_stocks.items()}
    comp = {}
    for d, stocks in domain_to_stocks.items():
        for s in stocks:
            comp[f"{d}:{s}"] = da[d] * sa[d][s]
    return da, sa, comp


def _walk_splits(config: ProjectConfig, dates: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp], int]]:
    train_len = max(60, int(config.data.walk_forward_train))
    test_len = max(20, int(config.data.walk_forward_test))
    step = max(10, int(config.data.walk_forward_step))
    out: list[tuple[list[pd.Timestamp], list[pd.Timestamp], int]] = []
    if len(dates) >= train_len + test_len + 1:
        last = len(dates) - train_len - test_len
        for start in range(0, last + 1, step):
            tr = dates[start : start + train_len]
            te = dates[start + train_len : start + train_len + test_len]
            if len(tr) >= 40 and len(te) >= 10:
                out.append((tr, te, start))
    if out:
        return out
    split = max(0.5, min(0.95, config.data.train_split))
    idx = max(2, min(int(len(dates) * split), len(dates) - 2))
    return [(dates[:idx], dates[idx:], 0)]


def _run_single(config: ProjectConfig, mode: str, seed: int, prepared: PreparedData) -> TrainingResult:
    g_count = len(prepared.global_feature_df.columns)
    d_count = len(next(iter(prepared.domain_factor_frames.values())).columns) if prepared.domain_factor_frames else 0
    s_count = len(prepared.stock_feature_names)
    agent = build_hierarchical_agent(
        mode=mode,
        domain_to_stocks=prepared.domain_to_stocks,
        max_domain_hold_steps=config.rl.max_domain_hold_steps,
        max_stock_hold_steps=config.rl.max_stock_hold_steps,
        num_global_features=g_count,
        num_domain_factors=d_count,
        num_stock_features=s_count,
        stock_feature_names=prepared.stock_feature_names,
        rl_config=config.rl,
        stochastic_control=config.stochastic_control,
        seed=seed,
    )
    domain_alloc, stock_allocs, prev_weights = _init_allocations(prepared.domain_to_stocks)
    splits = _walk_splits(config, prepared.valid_dates)

    train_rewards: list[float] = []
    train_returns: list[float] = []
    test_returns: list[float] = []
    losses: list[dict[str, float]] = []
    reward_components: list[dict[str, float | str | int]] = []
    train_domain_allocs: list[dict[str, float]] = []
    test_domain_allocs: list[dict[str, float]] = []
    train_dates: list[str] = []
    test_dates: list[str] = []
    walk: list[dict[str, Any]] = []
    regime_returns: dict[str, list[float]] = {}
    last_decision: HierarchicalDecision | None = None
    train_step = 0
    test_step = 0
    max_training_steps = int(config.rl.training_steps) if int(config.rl.training_steps) > 0 else None
    total_windows = len(splits)
    cluster_labels = ",".join(sorted(prepared.domain_to_stocks.keys()))
    print(
        f"[run] mode={mode} seed={seed} windows={total_windows} fine_tune_rounds={max(1, int(config.experiments.fine_tune_rounds))}",
        flush=True,
    )
    print(
        f"[domains] using={len(prepared.domain_to_stocks)} groups -> {cluster_labels}",
        flush=True,
    )

    stop_training = False
    for window_idx, (tr_dates, te_dates, _start) in enumerate(splits):
        tr_offset = len(train_returns)
        te_offset = len(test_returns)
        rounds = max(1, int(config.experiments.fine_tune_rounds))
        print(
            f"[window {window_idx + 1}/{total_windows}] train_steps={len(tr_dates)} test_steps={len(te_dates)}",
            flush=True,
        )
        train_log_every = max(1, len(tr_dates) // 10)
        test_log_every = max(1, len(te_dates) // 5)

        for round_idx in range(rounds):
            print(f"  [epoch {round_idx + 1}/{rounds}] training...", flush=True)
            active_train_decision: HierarchicalDecision | None = None
            train_hold_remaining = 0
            for local_idx, date in enumerate(tr_dates):
                if max_training_steps is not None and train_step >= max_training_steps:
                    stop_training = True
                    break
                top_state, domain_states = _build_states(
                    prepared=prepared,
                    date=date,
                    remaining=len(tr_dates) - local_idx,
                    step_idx=train_step,
                    domain_alloc=domain_alloc,
                    stock_allocs=stock_allocs,
                )
                did_rebalance = active_train_decision is None or train_hold_remaining <= 0
                if did_rebalance:
                    decision = agent.decide(
                        top_state=top_state,
                        domain_states=domain_states,
                        stochastic=config.rl.stochastic_actions,
                    )
                    active_train_decision = decision
                    train_hold_remaining = max(1, int(decision.top_action.hold_steps)) - 1
                else:
                    decision = active_train_decision
                    train_hold_remaining = max(0, train_hold_remaining - 1)

                assert decision is not None
                last_decision = decision
                train_domain_allocs.append(dict(decision.final_domain_weights))
                gross = _portfolio_return(decision, prepared.next_returns_df.loc[date])
                rb = _reward(gross, prev_weights, decision.final_stock_weights, train_returns, config.reward)
                reward = float(rb["reward"])
                train_returns.append(gross)
                train_rewards.append(reward)
                train_dates.append(str(pd.Timestamp(date).date()))
                reward_components.append(
                    {"phase": "train", "window": window_idx, "round": round_idx, "step": train_step, **rb}
                )

                next_domain_alloc = dict(decision.final_domain_weights)
                next_stock_allocs = dict(stock_allocs)
                for domain, action in decision.domain_actions.items():
                    next_stock_allocs[domain] = dict(action.stock_weights)
                next_date = tr_dates[min(local_idx + 1, len(tr_dates) - 1)]
                next_top_state, next_domain_states = _build_states(
                    prepared=prepared,
                    date=next_date,
                    remaining=max(0, len(tr_dates) - local_idx - 1),
                    step_idx=train_step + 1,
                    domain_alloc=next_domain_alloc,
                    stock_allocs=next_stock_allocs,
                )
                done = local_idx == len(tr_dates) - 1
                if did_rebalance:
                    step_losses = _train_step(
                        agent=agent,
                        top_state=top_state,
                        domain_states=domain_states,
                        next_top_state=next_top_state,
                        next_domain_states=next_domain_states,
                        decision=decision,
                        reward=reward,
                        done=done,
                    )
                    if step_losses:
                        row = {"window": float(window_idx), "round": float(round_idx), "step": float(train_step)}
                        row.update(step_losses)
                        losses.append(row)
                domain_alloc = next_domain_alloc
                stock_allocs = next_stock_allocs
                prev_weights = dict(decision.final_stock_weights)
                train_step += 1
                if (local_idx + 1) % train_log_every == 0 or local_idx + 1 == len(tr_dates):
                    print(
                        f"    epoch={round_idx + 1}/{rounds} step={local_idx + 1}/{len(tr_dates)} "
                        f"reward={reward:.5f} gross={gross:.5f} rebalance={int(did_rebalance)} hold_left={train_hold_remaining}",
                        flush=True,
                    )
            if stop_training:
                break
        if stop_training:
            print("  [training cap reached] stopping further train updates for this run", flush=True)

        print("  [eval] testing...", flush=True)
        active_test_decision: HierarchicalDecision | None = None
        test_hold_remaining = 0
        for local_idx, date in enumerate(te_dates):
            top_state, domain_states = _build_states(
                prepared=prepared,
                date=date,
                remaining=len(te_dates) - local_idx,
                step_idx=test_step,
                domain_alloc=domain_alloc,
                stock_allocs=stock_allocs,
            )
            did_rebalance = active_test_decision is None or test_hold_remaining <= 0
            if did_rebalance:
                decision = agent.decide(top_state=top_state, domain_states=domain_states, stochastic=False)
                active_test_decision = decision
                test_hold_remaining = max(1, int(decision.top_action.hold_steps)) - 1
            else:
                decision = active_test_decision
                test_hold_remaining = max(0, test_hold_remaining - 1)

            assert decision is not None
            last_decision = decision
            test_domain_allocs.append(dict(decision.final_domain_weights))
            gross = _portfolio_return(decision, prepared.next_returns_df.loc[date])
            rb = _reward(gross, prev_weights, decision.final_stock_weights, test_returns, config.reward)
            test_returns.append(gross)
            test_dates.append(str(pd.Timestamp(date).date()))
            reward_components.append(
                {"phase": "test", "window": window_idx, "round": -1, "step": test_step, **rb}
            )
            regime = str(prepared.regimes.loc[date])
            regime_returns.setdefault(regime, []).append(gross)
            domain_alloc = dict(decision.final_domain_weights)
            for domain, action in decision.domain_actions.items():
                stock_allocs[domain] = dict(action.stock_weights)
            prev_weights = dict(decision.final_stock_weights)
            test_step += 1
            if (local_idx + 1) % test_log_every == 0 or local_idx + 1 == len(te_dates):
                print(
                    f"    test_step={local_idx + 1}/{len(te_dates)} gross={gross:.5f} rebalance={int(did_rebalance)} hold_left={test_hold_remaining}",
                    flush=True,
                )

        walk.append(
            {
                "window": window_idx,
                "train_start": str(pd.Timestamp(tr_dates[0]).date()) if tr_dates else "",
                "train_end": str(pd.Timestamp(tr_dates[-1]).date()) if tr_dates else "",
                "test_start": str(pd.Timestamp(te_dates[0]).date()) if te_dates else "",
                "test_end": str(pd.Timestamp(te_dates[-1]).date()) if te_dates else "",
                "train_metrics": _metrics_dict(_metrics(train_returns[tr_offset:], config.reward.cvar_alpha)),
                "test_metrics": _metrics_dict(_metrics(test_returns[te_offset:], config.reward.cvar_alpha)),
            }
        )
        print(
            "  [window done] "
            f"train_cum={_metrics(train_returns[tr_offset:], config.reward.cvar_alpha).cumulative_return:.5f} "
            f"test_cum={_metrics(test_returns[te_offset:], config.reward.cvar_alpha).cumulative_return:.5f}",
            flush=True,
        )
        if stop_training:
            # Continue evaluating remaining windows but skip further train rounds.
            pass

    if last_decision is None:
        raise ValueError("No decision generated in training run")
    regime_metrics = {
        name: _metrics_dict(_metrics(vals, config.reward.cvar_alpha))
        for name, vals in regime_returns.items()
        if vals
    }
    return TrainingResult(
        provider=prepared.provider_label,
        mode=mode,
        seed=seed,
        train_rewards=train_rewards,
        train_period_returns=train_returns,
        test_period_returns=test_returns,
        losses=losses,
        train_metrics=_metrics(train_returns, config.reward.cvar_alpha),
        test_metrics=_metrics(test_returns, config.reward.cvar_alpha),
        train_dates=train_dates,
        test_dates=test_dates,
        train_domain_allocations=train_domain_allocs,
        test_domain_allocations=test_domain_allocs,
        last_decision=last_decision,
        reward_components=reward_components,
        walk_forward_windows=walk,
        regime_metrics=regime_metrics,
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
    train_curve = _equity(train_returns)
    test_curve = _equity(test_returns)
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
    tr = np.array(_equity(train_returns))
    te = np.array(_equity(test_returns))
    combined = np.concatenate([tr, tr[-1] * te[1:]])
    peaks = np.maximum.accumulate(combined)
    dd = (combined - peaks) / peaks
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(dd)), dd, color="#b91c1c", alpha=0.35)
    ax.plot(dd, color="#7f1d1d")
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_domain_allocations(allocations: list[dict[str, float]], title: str, path: Path) -> None:
    if not allocations:
        return
    frame = pd.DataFrame(allocations).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(range(len(frame)), [frame[c].values for c in frame.columns], labels=list(frame.columns), alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_regimes(regime_metrics: Mapping[str, Mapping[str, float | int]], path: Path) -> None:
    if not regime_metrics:
        return
    labels = list(regime_metrics.keys())
    vals = [float(regime_metrics[k].get("cumulative_return", 0.0)) for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.bar(labels, vals, color="#334155")
    ax.set_title("Cumulative Return By Regime")
    ax.set_ylabel("Cumulative Return")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_training_result(result: TrainingResult, output_dir: str | Path) -> TrainingResult:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    teq = _equity(result.train_period_returns)
    seq = _equity(result.test_period_returns)
    pd.DataFrame(
        {
            "date": result.train_dates[: len(result.train_period_returns)],
            "period_return": result.train_period_returns,
            "reward": result.train_rewards[: len(result.train_period_returns)],
            "equity": teq[1:],
        }
    ).to_csv(path / "train_returns.csv", index=False)
    pd.DataFrame(
        {
            "date": result.test_dates[: len(result.test_period_returns)],
            "period_return": result.test_period_returns,
            "equity": seq[1:],
        }
    ).to_csv(path / "test_returns.csv", index=False)
    if result.losses:
        pd.DataFrame(result.losses).to_csv(path / "losses.csv", index=False)
    pd.DataFrame(result.train_domain_allocations).to_csv(path / "train_domain_allocations.csv", index=False)
    pd.DataFrame(result.test_domain_allocations).to_csv(path / "test_domain_allocations.csv", index=False)
    if result.reward_components:
        pd.DataFrame(result.reward_components).to_csv(path / "reward_components.csv", index=False)
    if result.walk_forward_windows:
        pd.DataFrame(result.walk_forward_windows).to_csv(path / "walk_forward_windows.csv", index=False)

    summary = {
        "provider": result.provider,
        "mode": result.mode,
        "seed": result.seed,
        "train_metrics": _metrics_dict(result.train_metrics),
        "test_metrics": _metrics_dict(result.test_metrics),
        "regime_metrics": result.regime_metrics,
        "walk_forward_windows": result.walk_forward_windows,
        "last_top_hold_steps": result.last_decision.top_action.hold_steps,
        "last_top_weights": dict(result.last_decision.top_action.domain_weights),
        "last_final_domain_weights": dict(result.last_decision.final_domain_weights),
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_rewards(result.train_rewards, path / "reward_curve.png")
    _plot_equity(result.train_period_returns, result.test_period_returns, path / "equity_curve.png")
    _plot_drawdown(result.train_period_returns, result.test_period_returns, path / "drawdown_curve.png")
    _plot_domain_allocations(result.train_domain_allocations, "Train Domain Allocation", path / "train_domain_allocation.png")
    _plot_domain_allocations(result.test_domain_allocations, "Test Domain Allocation", path / "test_domain_allocation.png")
    _plot_regimes(result.regime_metrics, path / "regime_returns.png")

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
        reward_components=result.reward_components,
        walk_forward_windows=result.walk_forward_windows,
        regime_metrics=result.regime_metrics,
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
    prepared = _prepare_data(config)
    return _run_single(config=config, mode=config.top_mode, seed=config.rl.random_seed, prepared=prepared)


def run_multiple_experiments(config_path: str | Path) -> BatchResult:
    config = load_config(config_path)
    prepared = _prepare_data(config)
    modes = config.experiments.modes or [config.top_mode]
    seeds = config.experiments.seeds or [config.rl.random_seed]
    grid = [(m, s) for m in modes for s in seeds] or [(config.top_mode, config.rl.random_seed)]

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(config.experiments.results_dir) / f"{config.experiments.run_name}_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs: list[TrainingResult] = []
    summary_rows: list[dict[str, Any]] = []
    for idx, (mode, seed) in enumerate(grid):
        print(f"[{idx + 1}/{len(grid)}] running mode={mode}, seed={seed}")
        run_result = _run_single(config=config, mode=mode, seed=seed, prepared=prepared)
        run_dir = batch_dir / f"run_{idx:02d}_{mode}_seed{seed}"
        run_result = save_training_result(run_result, run_dir)
        print(f"[{idx + 1}/{len(grid)}] saved artifacts: {run_dir}")
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
                "test_cvar": run_result.test_metrics.cvar,
            }
        )

    pd.DataFrame(summary_rows).to_csv(batch_dir / "batch_summary.csv", index=False)
    _plot_batch_cumulative_returns(runs, batch_dir / "batch_cumulative_returns.png")
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return BatchResult(provider=prepared.provider_label, batch_dir=str(batch_dir), runs=runs)
