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
    evaluation_extras: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None


@dataclass(frozen=True)
class BatchResult:
    provider: str
    batch_dir: str
    runs: list[TrainingResult]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    top_mode: str = "rl"
    use_learned_clusters: bool = True
    use_stochastic_control: bool = True
    enforce_hold: bool = True
    use_global_macro_features: bool = True
    use_liquidity_features: bool = True


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
    close_df: pd.DataFrame
    returns_df: pd.DataFrame
    factor_returns_df: pd.DataFrame
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
    """Past-only regime classification using expanding volatility quantiles."""

    hist_vol = vol.shift(1)
    q50 = hist_vol.expanding(min_periods=40).quantile(0.50).ffill()
    q75 = hist_vol.expanding(min_periods=40).quantile(0.75).ffill()
    labels = pd.Series("sideways", index=ret.index, dtype=object)
    for date in ret.index:
        r = float(ret.loc[date]) if pd.notna(ret.loc[date]) else np.nan
        v = float(vol.loc[date]) if pd.notna(vol.loc[date]) else np.nan
        median = float(q50.loc[date]) if pd.notna(q50.loc[date]) else np.nan
        high = float(q75.loc[date]) if pd.notna(q75.loc[date]) else np.nan
        if np.isnan(r) or np.isnan(v) or np.isnan(median) or np.isnan(high):
            labels.loc[date] = "sideways"
            continue
        if r >= 0.0 and v <= median:
            labels.loc[date] = "bull"
        elif r < 0.0 and v >= high:
            labels.loc[date] = "bear"
        elif v >= high:
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
    corr = returns_df[stocks].corr(min_periods=20).fillna(0.0)
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

    # Keep deterministic label space for rolling windows.
    return {name: symbols for name, symbols in cluster_assignments.items()}


@dataclass(frozen=True)
class _RegimeNormalizer:
    base_mean: pd.Series
    base_std: pd.Series
    by_regime: dict[str, tuple[pd.Series, pd.Series]]


def _fit_regime_normalizer(
    frame: pd.DataFrame,
    regimes: pd.Series,
    train_dates: list[pd.Timestamp],
) -> _RegimeNormalizer:
    train_idx = pd.Index(train_dates)
    train_frame = frame.loc[train_idx]
    train_regimes = regimes.loc[train_idx]
    base_mean = train_frame.mean(axis=0)
    base_std = train_frame.std(axis=0).replace(0.0, np.nan).fillna(1.0)

    by_regime: dict[str, tuple[pd.Series, pd.Series]] = {}
    for regime_name in sorted(set(str(v) for v in train_regimes.values)):
        mask = train_regimes == regime_name
        subset = train_frame.loc[mask]
        if subset.empty:
            continue
        mean = subset.mean(axis=0)
        std = subset.std(axis=0).replace(0.0, np.nan).fillna(1.0)
        by_regime[regime_name] = (mean, std)
    return _RegimeNormalizer(base_mean=base_mean, base_std=base_std, by_regime=by_regime)


def _apply_regime_normalizer(
    frame: pd.DataFrame,
    regimes: pd.Series,
    normalizer: _RegimeNormalizer,
) -> pd.DataFrame:
    out = frame.sub(normalizer.base_mean, axis=1).div(normalizer.base_std, axis=1)
    for regime_name, (mean, std) in normalizer.by_regime.items():
        mask = regimes == regime_name
        if not bool(mask.any()):
            continue
        out.loc[mask] = frame.loc[mask].sub(mean, axis=1).div(std, axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_frames_by_window(
    frames: Mapping[str, pd.DataFrame],
    regimes: pd.Series,
    train_dates: list[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        norm = _fit_regime_normalizer(frame, regimes, train_dates)
        normalized[name] = _apply_regime_normalizer(frame, regimes, norm)
    return normalized


def _normalize_frame_by_window(
    frame: pd.DataFrame,
    regimes: pd.Series,
    train_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    norm = _fit_regime_normalizer(frame, regimes, train_dates)
    return _apply_regime_normalizer(frame, regimes, norm)


def _kmeans_numpy(values: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros((k, values.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    k = max(1, min(k, values.shape[0]))
    if values.shape[0] == 1:
        return np.repeat(values, repeats=k, axis=0)
    idx = rng.choice(values.shape[0], size=k, replace=False)
    centers = values[idx].copy()
    for _ in range(max(3, iters)):
        dist = np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            members = values[labels == j]
            if members.shape[0] > 0:
                new_centers[j] = members.mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    return centers


def _build_market_state_embeddings(
    global_df: pd.DataFrame,
    train_dates: list[pd.Timestamp],
    latent_k: int,
    seed: int,
) -> pd.DataFrame:
    cols = [f"latent_regime_{i:02d}" for i in range(max(1, latent_k))]
    if global_df.empty:
        return pd.DataFrame(0.0, index=global_df.index, columns=cols)
    train_idx = pd.Index(train_dates)
    fit_x = global_df.loc[train_idx].to_numpy(dtype=float)
    fit_x = np.nan_to_num(fit_x, nan=0.0, posinf=0.0, neginf=0.0)
    centers = _kmeans_numpy(fit_x, k=max(1, latent_k), seed=seed)
    full_x = np.nan_to_num(global_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.square(full_x[:, None, :] - centers[None, :, :]).sum(axis=2)
    labels = np.argmin(dist, axis=1)
    out = pd.DataFrame(0.0, index=global_df.index, columns=cols)
    for i, label in enumerate(labels):
        col = cols[int(label)]
        out.iat[i, out.columns.get_loc(col)] = 1.0
    return out


def _build_domain_factors(
    config: ProjectConfig,
    domain_to_stocks: Mapping[str, list[str]],
    close_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    lb = max(2, config.data.lookback)
    m = config.data.market_symbol
    mret = factor_returns_df[m]
    domain_factors: dict[str, pd.DataFrame] = {}
    for domain, symbols in domain_to_stocks.items():
        clean_symbols = [s for s in symbols if s in returns_df.columns]
        if not clean_symbols:
            continue
        subset = returns_df[clean_symbols]
        dret = subset.mean(axis=1)
        dcurve = (1.0 + dret.fillna(0.0)).cumprod()
        ddrawdown = (dcurve - np.maximum.accumulate(dcurve)) / np.maximum.accumulate(dcurve)
        sector = config.data.sector_etfs.get(domain, m)
        sret = factor_returns_df[sector] if sector in factor_returns_df.columns else dret

        top_bottom = subset.apply(
            lambda row: float(
                row.sort_values().tail(max(1, len(clean_symbols) // 5)).mean()
                - row.sort_values().head(max(1, len(clean_symbols) // 5)).mean()
            ),
            axis=1,
        )
        intra_corr = 1.0 - (subset.std(axis=1) / (subset.abs().mean(axis=1) + 1e-6))
        f = pd.DataFrame(index=close_df.index)
        f["sector_ret"] = sret
        f["relative_strength"] = sret - mret
        f["beta_proxy"] = dret.rolling(lb).corr(mret).replace([np.inf, -np.inf], np.nan)
        f["dispersion"] = subset.std(axis=1)
        f["cluster_momentum"] = close_df[clean_symbols].mean(axis=1).pct_change(20)
        f["cluster_volatility"] = subset.mean(axis=1).rolling(30).std() * np.sqrt(252.0)
        f["cluster_drawdown"] = ddrawdown
        sharpe_roll = dret.rolling(30).mean() / (dret.rolling(30).std().replace(0.0, np.nan))
        f["cluster_sharpe_rolling"] = sharpe_roll * np.sqrt(252.0)
        f["cluster_correlation_to_market"] = dret.rolling(30).corr(mret).replace([np.inf, -np.inf], np.nan)
        f["cross_sectional_std_returns"] = subset.std(axis=1)
        f["top_minus_bottom_quintile_return"] = top_bottom
        f["intra_cluster_correlation"] = intra_corr
        domain_factors[domain] = f.replace([np.inf, -np.inf], np.nan)
    return domain_factors


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

    # Macro/global features and regime labels are lagged so decisions only use past information.
    g = g.shift(1)
    regimes = regimes.shift(1).fillna("sideways")
    domain_factors = {
        name: frame.shift(1)
        for name, frame in _build_domain_factors(
            config=config,
            domain_to_stocks=domain_to_stocks,
            close_df=close_df,
            returns_df=ret,
            factor_returns_df=factor_ret,
        ).items()
    }

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
    domain_map = {
        str(name): list(symbols) for name, symbols in config.domain_to_stocks.items()
    }
    returns_df = close_df.pct_change().replace([np.inf, -np.inf], np.nan)
    factor_returns_df = factor_df.pct_change().replace([np.inf, -np.inf], np.nan)

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
        close_df=close_df,
        returns_df=returns_df,
        factor_returns_df=factor_returns_df,
        provider_label=provider_label,
    )


def _build_states(
    prepared: PreparedData,
    date: pd.Timestamp,
    remaining: int,
    step_idx: int,
    domain_alloc: Mapping[str, float],
    stock_allocs: Mapping[str, Mapping[str, float]],
    domain_to_stocks: Mapping[str, list[str]] | None = None,
    domain_factor_frames: Mapping[str, pd.DataFrame] | None = None,
    global_feature_df: pd.DataFrame | None = None,
    stock_feature_frames: Mapping[str, pd.DataFrame] | None = None,
    stock_feature_names: list[str] | None = None,
    momentum_df: pd.DataFrame | None = None,
    volatility_df: pd.DataFrame | None = None,
    liquidity_df: pd.DataFrame | None = None,
    include_global_features: bool = True,
    include_liquidity_features: bool = True,
) -> tuple[TopLevelState, dict[str, DomainState]]:
    domain_map = {str(k): list(v) for k, v in (domain_to_stocks or prepared.domain_to_stocks).items()}
    d_factors = domain_factor_frames or prepared.domain_factor_frames
    g_frame = global_feature_df if global_feature_df is not None else prepared.global_feature_df
    s_frames = stock_feature_frames or prepared.stock_feature_frames
    s_names = stock_feature_names or prepared.stock_feature_names
    mom_df = momentum_df if momentum_df is not None else prepared.momentum_df
    vol_df = volatility_df if volatility_df is not None else prepared.volatility_df
    liq_df = liquidity_df if liquidity_df is not None else prepared.liquidity_df

    dm, dv, dl = {}, {}, {}
    domain_states: dict[str, DomainState] = {}
    global_features = (
        {c: float(g_frame.at[date, c]) for c in sorted(g_frame.columns)}
        if include_global_features
        else {}
    )
    for domain, symbols in domain_map.items():
        sm = {s: float(mom_df.at[date, s]) for s in symbols}
        sv = {s: float(vol_df.at[date, s]) for s in symbols}
        sl = (
            {s: float(liq_df.at[date, s]) for s in symbols}
            if include_liquidity_features
            else {s: 0.0 for s in symbols}
        )
        factor_frame = d_factors.get(domain)
        if factor_frame is None or factor_frame.empty:
            sf = {}
        else:
            sf = {
                c: float(factor_frame.at[date, c])
                for c in sorted(factor_frame.columns)
            }

        mom_series = pd.Series(sm, dtype=float)
        vol_series = pd.Series(sv, dtype=float)
        mom_rank = mom_series.rank(pct=True, method="average").to_dict()
        vol_rank = vol_series.rank(pct=True, method="average").to_dict()
        mean_mom = float(mom_series.mean())
        stock_features: dict[str, dict[str, float]] = {}
        for stock in symbols:
            base_features = {
                name: float(s_frames[name].at[date, stock])
                for name in s_names
            }
            base_features["stock_mom_rank_in_cluster"] = float(mom_rank.get(stock, 0.0))
            base_features["stock_vol_rank"] = float(vol_rank.get(stock, 0.0))
            base_features["relative_momentum"] = float(sm.get(stock, 0.0) - mean_mom)
            if not include_liquidity_features:
                for key in list(base_features.keys()):
                    if "liq" in key.lower() or "liquid" in key.lower():
                        base_features[key] = 0.0
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


def _domain_step_rewards(decision: HierarchicalDecision, next_row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for domain, action in decision.domain_actions.items():
        domain_ret = 0.0
        for stock, weight in action.stock_weights.items():
            domain_ret += float(weight) * float(next_row.get(stock, 0.0))
        out[domain] = float(domain_ret)
    return out


def _rank_loss(final_stock_weights: Mapping[str, float], next_row: pd.Series) -> float:
    if not final_stock_weights:
        return 0.0
    weights_by_stock: dict[str, float] = {}
    for key, weight in final_stock_weights.items():
        stock = str(key).split(":", 1)[-1]
        weights_by_stock[stock] = weights_by_stock.get(stock, 0.0) + float(weight)
    if len(weights_by_stock) < 2:
        return 0.0
    stocks = sorted(weights_by_stock.keys())
    w = pd.Series([weights_by_stock[s] for s in stocks], index=stocks, dtype=float)
    r = pd.Series([float(next_row.get(s, 0.0)) for s in stocks], index=stocks, dtype=float)
    w_rank = w.rank(pct=True, method="average")
    r_rank = r.rank(pct=True, method="average")
    if float(w_rank.std(ddof=0)) < 1e-12 or float(r_rank.std(ddof=0)) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(w_rank.to_numpy(dtype=float), r_rank.to_numpy(dtype=float))[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return float(max(0.0, 1.0 - corr))


def _turnover(prev_w: Mapping[str, float], new_w: Mapping[str, float]) -> float:
    def _aggregate(weights: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in weights.items():
            stock = str(key).split(":", 1)[-1]
            out[stock] = out.get(stock, 0.0) + float(value)
        return out

    prev_by_stock = _aggregate(prev_w)
    new_by_stock = _aggregate(new_w)
    keys = set(prev_by_stock) | set(new_by_stock)
    return float(
        sum(
            abs(float(new_by_stock.get(k, 0.0)) - float(prev_by_stock.get(k, 0.0)))
            for k in keys
        )
    )


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
    rank_loss: float = 0.0,
    domain_entropy_bonus: float = 0.0,
    hold_penalty: float = 0.0,
) -> dict[str, float]:
    t = _turnover(prev_w, new_w)
    tx = t * (cfg.transaction_cost_bps / 10000.0)
    sl = t * (cfg.slippage_bps / 10000.0)
    tp = cfg.turnover_penalty * t
    down = cfg.downside_penalty * max(0.0, -gross)
    win = (hist + [gross])[-max(5, int(cfg.reward_window)) :]
    cvar = _compute_cvar(win, cfg.cvar_alpha)
    cp = cfg.cvar_penalty * max(0.0, -cvar)
    cost_norm = (tx + sl + tp) / (1.0 + abs(gross))
    cvar_norm = max(0.0, -cvar) / (1.0 + abs(cvar))
    turnover_norm = t / (1.0 + t)
    rank_penalty = cfg.rank_loss_coef * max(0.0, float(rank_loss))
    hold_cost = cfg.hold_penalty_coef * max(0.0, float(hold_penalty))
    entropy_bonus = cfg.domain_entropy_coef * max(0.0, float(domain_entropy_bonus))
    r = gross - cost_norm - down - cvar_norm - turnover_norm - rank_penalty - hold_cost + entropy_bonus
    return {
        "gross_return": float(gross),
        "turnover": float(t),
        "transaction_cost": float(tx),
        "slippage_cost": float(sl),
        "turnover_penalty": float(tp),
        "downside_penalty": float(down),
        "cvar": float(cvar),
        "cvar_penalty": float(cp),
        "rank_loss": float(rank_loss),
        "rank_penalty": float(rank_penalty),
        "domain_entropy_bonus": float(entropy_bonus),
        "hold_penalty": float(hold_cost),
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
    domain_rewards: Mapping[str, float] | None = None,
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
    if agent.stochastic_controller is not None and hasattr(agent.stochastic_controller, "update_from_reward"):
        agent.stochastic_controller.update_from_reward(reward)
    for domain, action in decision.domain_actions.items():
        manager = agent.domain_managers[domain]
        d_reward = float(domain_rewards.get(domain, reward)) if domain_rewards is not None else float(reward)
        manager.remember(domain_states[domain], action, d_reward, next_domain_states[domain], done)
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


def _newey_west_t(values: list[float], lags: int = 5) -> float:
    if len(values) < 8:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    resid = arr - mu
    max_lag = max(1, min(int(lags), len(arr) - 1))
    gamma0 = float(np.dot(resid, resid) / len(arr))
    var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(resid[lag:], resid[:-lag]) / len(arr))
        weight = 1.0 - lag / (max_lag + 1.0)
        var += 2.0 * weight * cov
    se = math.sqrt(max(1e-12, var / len(arr)))
    return float(mu / se)


def _deflated_sharpe(values: list[float], num_trials: int) -> float:
    if len(values) < 8:
        return 0.0
    arr = np.asarray(values, dtype=float)
    period_sharpe = float(np.mean(arr) / (np.std(arr, ddof=0) + 1e-12) * np.sqrt(252.0))
    max_ref = math.sqrt(2.0 * math.log(max(2, int(num_trials)))) / math.sqrt(len(arr))
    return float(period_sharpe - max_ref)


def _bootstrap_ci(
    values: list[float],
    stat_fn: Any,
    n_boot: int = 400,
    seed: int = 7,
) -> tuple[float, float]:
    if len(values) < 5:
        s = float(stat_fn(values)) if values else 0.0
        return s, s
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    stats = []
    for _ in range(max(50, int(n_boot))):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        stats.append(float(stat_fn(sample.tolist())))
    lo = float(np.quantile(stats, 0.025))
    hi = float(np.quantile(stats, 0.975))
    return lo, hi


def _crisis_metrics(dates: list[str], values: list[float], cvar_alpha: float) -> dict[str, dict[str, float | int]]:
    if not dates or not values:
        return {}
    idx = pd.to_datetime(pd.Series(dates)).dt.date
    ret = pd.Series(values, index=idx, dtype=float)
    windows = [
        ("covid_crash_2020", pd.Timestamp("2020-02-15").date(), pd.Timestamp("2020-06-30").date()),
        ("inflation_shock_2022", pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-12-31").date()),
        ("regional_bank_stress_2023", pd.Timestamp("2023-03-01").date(), pd.Timestamp("2023-06-30").date()),
    ]
    out: dict[str, dict[str, float | int]] = {}
    for name, start, end in windows:
        sub = ret.loc[(ret.index >= start) & (ret.index <= end)]
        if sub.empty:
            continue
        out[name] = _metrics_dict(_metrics(sub.tolist(), cvar_alpha))
    return out


def _turnover_decomposition(components: list[dict[str, float | str | int]]) -> dict[str, float]:
    if not components:
        return {}
    frame = pd.DataFrame(components)
    if frame.empty or "turnover" not in frame.columns:
        return {}
    frame["rebalance"] = frame.get("rebalance", 0).fillna(0).astype(int)
    out = {
        "mean_turnover_all": float(frame["turnover"].mean()),
        "mean_turnover_rebalance": float(frame.loc[frame["rebalance"] == 1, "turnover"].mean())
        if not frame.loc[frame["rebalance"] == 1].empty
        else 0.0,
        "mean_turnover_hold": float(frame.loc[frame["rebalance"] == 0, "turnover"].mean())
        if not frame.loc[frame["rebalance"] == 0].empty
        else 0.0,
    }
    return out


def _capacity_liquidity_analysis(
    test_dates: list[str],
    test_returns: list[float],
    liquidity_df: pd.DataFrame,
    cvar_alpha: float,
) -> dict[str, dict[str, float | int]]:
    if not test_dates or not test_returns or liquidity_df.empty:
        return {}
    idx = pd.to_datetime(pd.Series(test_dates))
    liq = liquidity_df.reindex(idx).mean(axis=1).fillna(liquidity_df.mean(axis=1).mean())
    out: dict[str, dict[str, float | int]] = {}
    base = np.asarray(test_returns, dtype=float)
    for scale in [1.0, 5.0, 10.0]:
        impact = scale * 1e-5 / (liq.to_numpy(dtype=float) + 1e-6)
        adjusted = (base - impact).tolist()
        out[f"capacity_x{int(scale)}"] = _metrics_dict(_metrics(adjusted, cvar_alpha))
    return out


def _evaluation_extras(
    method_count: int,
    test_dates: list[str],
    test_returns: list[float],
    reward_components: list[dict[str, float | str | int]],
    liquidity_df: pd.DataFrame,
    cvar_alpha: float,
) -> dict[str, Any]:
    sharpe_stat = lambda vals: _metrics(vals, cvar_alpha).sharpe
    cum_stat = lambda vals: _metrics(vals, cvar_alpha).cumulative_return
    sharpe_ci = _bootstrap_ci(test_returns, sharpe_stat, n_boot=300, seed=17)
    cum_ci = _bootstrap_ci(test_returns, cum_stat, n_boot=300, seed=19)
    return {
        "newey_west_t_stat": _newey_west_t(test_returns, lags=5),
        "deflated_sharpe_ratio": _deflated_sharpe(test_returns, num_trials=max(1, method_count)),
        "bootstrap_ci": {
            "sharpe_95": [float(sharpe_ci[0]), float(sharpe_ci[1])],
            "cumulative_return_95": [float(cum_ci[0]), float(cum_ci[1])],
        },
        "crisis_subperiods": _crisis_metrics(test_dates, test_returns, cvar_alpha),
        "turnover_decomposition": _turnover_decomposition(reward_components),
        "capacity_liquidity_scaling": _capacity_liquidity_analysis(
            test_dates=test_dates,
            test_returns=test_returns,
            liquidity_df=liquidity_df,
            cvar_alpha=cvar_alpha,
        ),
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


def _method_from_name(method_id: str, config: ProjectConfig) -> MethodSpec:
    key = str(method_id).strip().lower()
    if key in {"rl", "hmadrl"}:
        return MethodSpec(method_id="rl", top_mode="rl")
    if key in {"moe_router", "moe"}:
        return MethodSpec(method_id="moe_router", top_mode="moe_router")

    default_mode = str(config.top_mode).strip().lower() or "rl"
    if key == default_mode and default_mode in {"rl", "moe_router"}:
        return MethodSpec(method_id=default_mode, top_mode=default_mode)
    raise ValueError(
        "Unsupported method '{method}'. System-only branch supports only 'rl' and 'moe_router'.".format(
            method=method_id
        )
    )


@dataclass(frozen=True)
class WindowFeatureContext:
    domain_to_stocks: dict[str, list[str]]
    momentum_df: pd.DataFrame
    volatility_df: pd.DataFrame
    liquidity_df: pd.DataFrame
    stock_feature_frames: dict[str, pd.DataFrame]
    stock_feature_names: list[str]
    global_feature_df: pd.DataFrame
    domain_factor_frames: dict[str, pd.DataFrame]
    regimes: pd.Series


def _resolve_window_domain_map(
    config: ProjectConfig,
    prepared: PreparedData,
    train_dates: list[pd.Timestamp],
    use_learned_clusters: bool,
) -> dict[str, list[str]]:
    static_map = {
        str(name): [s for s in symbols if s in prepared.returns_df.columns]
        for name, symbols in config.domain_to_stocks.items()
    }
    static_map = {name: symbols for name, symbols in static_map.items() if symbols}
    if not config.data.use_learned_clusters or not use_learned_clusters:
        return static_map
    train_returns = prepared.returns_df.loc[pd.Index(train_dates)]
    learned = _learned_cluster_map(
        config.domain_to_stocks,
        returns_df=train_returns,
        cluster_count=config.data.learned_cluster_count,
    )
    cleaned = {
        str(name): [s for s in symbols if s in prepared.returns_df.columns]
        for name, symbols in learned.items()
    }
    cleaned = {name: symbols for name, symbols in cleaned.items() if symbols}
    return cleaned or static_map


def _build_window_context(
    config: ProjectConfig,
    prepared: PreparedData,
    train_dates: list[pd.Timestamp],
    domain_map: Mapping[str, list[str]],
    seed: int,
) -> WindowFeatureContext:
    regimes = prepared.regimes.copy()

    momentum_df = _normalize_frame_by_window(prepared.momentum_df, regimes, train_dates)
    volatility_df = _normalize_frame_by_window(prepared.volatility_df, regimes, train_dates)
    liquidity_df = _normalize_frame_by_window(prepared.liquidity_df, regimes, train_dates)
    stock_feature_frames = _normalize_frames_by_window(prepared.stock_feature_frames, regimes, train_dates)
    global_feature_df = _normalize_frame_by_window(prepared.global_feature_df, regimes, train_dates)

    latent_states = _build_market_state_embeddings(
        global_df=global_feature_df,
        train_dates=train_dates,
        latent_k=4,
        seed=seed,
    )
    global_feature_df = pd.concat([global_feature_df, latent_states], axis=1)

    raw_domain_factors = _build_domain_factors(
        config=config,
        domain_to_stocks=domain_map,
        close_df=prepared.close_df,
        returns_df=prepared.returns_df,
        factor_returns_df=prepared.factor_returns_df,
    )
    shifted_domain_factors = {
        name: frame.shift(1)
        for name, frame in raw_domain_factors.items()
    }
    domain_factor_frames = _normalize_frames_by_window(shifted_domain_factors, regimes, train_dates)

    return WindowFeatureContext(
        domain_to_stocks={str(k): list(v) for k, v in domain_map.items()},
        momentum_df=momentum_df,
        volatility_df=volatility_df,
        liquidity_df=liquidity_df,
        stock_feature_frames=stock_feature_frames,
        stock_feature_names=list(prepared.stock_feature_names),
        global_feature_df=global_feature_df,
        domain_factor_frames=domain_factor_frames,
        regimes=regimes,
    )


def _run_single(config: ProjectConfig, mode: str, seed: int, prepared: PreparedData) -> TrainingResult:
    spec = _method_from_name(mode, config)
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
    print(
        f"[run] mode={spec.method_id} seed={seed} windows={total_windows} fine_tune_rounds={max(1, int(config.experiments.fine_tune_rounds))}",
        flush=True,
    )

    stop_training = False
    for window_idx, (tr_dates, te_dates, _start) in enumerate(splits):
        window_domain_map = _resolve_window_domain_map(
            config,
            prepared,
            tr_dates,
            use_learned_clusters=spec.use_learned_clusters,
        )
        window_context = _build_window_context(
            config=config,
            prepared=prepared,
            train_dates=tr_dates,
            domain_map=window_domain_map,
            seed=seed + window_idx * 101,
        )
        cluster_labels = ",".join(sorted(window_context.domain_to_stocks.keys()))
        print(
            f"[domains] window={window_idx + 1}/{total_windows} using={len(window_context.domain_to_stocks)} groups -> {cluster_labels}",
            flush=True,
        )
        g_count = len(window_context.global_feature_df.columns)
        d_count = (
            len(next(iter(window_context.domain_factor_frames.values())).columns)
            if window_context.domain_factor_frames
            else 0
        )
        s_count = len(window_context.stock_feature_names)
        agent = build_hierarchical_agent(
            mode=spec.top_mode,
            domain_to_stocks=window_context.domain_to_stocks,
            max_domain_hold_steps=config.rl.max_domain_hold_steps,
            max_stock_hold_steps=config.rl.max_stock_hold_steps,
            num_global_features=g_count,
            num_domain_factors=d_count,
            num_stock_features=s_count,
            stock_feature_names=window_context.stock_feature_names,
            rl_config=config.rl,
            stochastic_control=config.stochastic_control if spec.use_stochastic_control else None,
            seed=seed + window_idx * 13,
        )
        domain_alloc, stock_allocs, prev_weights = _init_allocations(window_context.domain_to_stocks)
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
                    domain_to_stocks=window_context.domain_to_stocks,
                    domain_factor_frames=window_context.domain_factor_frames,
                    global_feature_df=window_context.global_feature_df,
                    stock_feature_frames=window_context.stock_feature_frames,
                    stock_feature_names=window_context.stock_feature_names,
                    momentum_df=window_context.momentum_df,
                    volatility_df=window_context.volatility_df,
                    liquidity_df=window_context.liquidity_df,
                    include_global_features=spec.use_global_macro_features,
                    include_liquidity_features=spec.use_liquidity_features,
                )
                did_rebalance = (active_train_decision is None or train_hold_remaining <= 0) if spec.enforce_hold else True
                if did_rebalance:
                    decision = agent.decide(
                        top_state=top_state,
                        domain_states=domain_states,
                        stochastic=config.rl.stochastic_actions,
                    )
                    active_train_decision = decision
                    train_hold_remaining = max(1, int(decision.top_action.hold_steps)) - 1 if spec.enforce_hold else 0
                else:
                    decision = active_train_decision
                    train_hold_remaining = max(0, train_hold_remaining - 1)

                assert decision is not None
                last_decision = decision
                train_domain_allocs.append(dict(decision.final_domain_weights))
                next_row = prepared.next_returns_df.loc[date]
                gross = _portfolio_return(decision, next_row)
                rank_loss = _rank_loss(decision.final_stock_weights, next_row)
                entropy = -sum(
                    float(w) * math.log(max(1e-12, float(w)))
                    for w in decision.final_domain_weights.values()
                )
                hold_over = max(0, int(decision.top_action.hold_steps) - int(config.reward.max_reasonable_hold))
                rb = _reward(
                    gross,
                    prev_weights,
                    decision.final_stock_weights,
                    train_returns,
                    config.reward,
                    rank_loss=rank_loss,
                    domain_entropy_bonus=entropy,
                    hold_penalty=float(hold_over),
                )
                reward = float(rb["reward"])
                train_returns.append(gross)
                train_rewards.append(reward)
                train_dates.append(str(pd.Timestamp(date).date()))
                reward_components.append(
                    {
                        "phase": "train",
                        "window": window_idx,
                        "round": round_idx,
                        "step": train_step,
                        "rebalance": int(did_rebalance),
                        **rb,
                    }
                )
                d_rewards = _domain_step_rewards(decision, next_row)

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
                    domain_to_stocks=window_context.domain_to_stocks,
                    domain_factor_frames=window_context.domain_factor_frames,
                    global_feature_df=window_context.global_feature_df,
                    stock_feature_frames=window_context.stock_feature_frames,
                    stock_feature_names=window_context.stock_feature_names,
                    momentum_df=window_context.momentum_df,
                    volatility_df=window_context.volatility_df,
                    liquidity_df=window_context.liquidity_df,
                    include_global_features=spec.use_global_macro_features,
                    include_liquidity_features=spec.use_liquidity_features,
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
                        domain_rewards=d_rewards,
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
                domain_to_stocks=window_context.domain_to_stocks,
                domain_factor_frames=window_context.domain_factor_frames,
                global_feature_df=window_context.global_feature_df,
                stock_feature_frames=window_context.stock_feature_frames,
                stock_feature_names=window_context.stock_feature_names,
                momentum_df=window_context.momentum_df,
                volatility_df=window_context.volatility_df,
                liquidity_df=window_context.liquidity_df,
                include_global_features=spec.use_global_macro_features,
                include_liquidity_features=spec.use_liquidity_features,
            )
            did_rebalance = (active_test_decision is None or test_hold_remaining <= 0) if spec.enforce_hold else True
            if did_rebalance:
                decision = agent.decide(top_state=top_state, domain_states=domain_states, stochastic=False)
                active_test_decision = decision
                test_hold_remaining = max(1, int(decision.top_action.hold_steps)) - 1 if spec.enforce_hold else 0
            else:
                decision = active_test_decision
                test_hold_remaining = max(0, test_hold_remaining - 1)

            assert decision is not None
            last_decision = decision
            test_domain_allocs.append(dict(decision.final_domain_weights))
            next_row = prepared.next_returns_df.loc[date]
            gross = _portfolio_return(decision, next_row)
            rank_loss = _rank_loss(decision.final_stock_weights, next_row)
            entropy = -sum(
                float(w) * math.log(max(1e-12, float(w)))
                for w in decision.final_domain_weights.values()
            )
            hold_over = max(0, int(decision.top_action.hold_steps) - int(config.reward.max_reasonable_hold))
            rb = _reward(
                gross,
                prev_weights,
                decision.final_stock_weights,
                test_returns,
                config.reward,
                rank_loss=rank_loss,
                domain_entropy_bonus=entropy,
                hold_penalty=float(hold_over),
            )
            test_returns.append(gross)
            test_dates.append(str(pd.Timestamp(date).date()))
            reward_components.append(
                {
                    "phase": "test",
                    "window": window_idx,
                    "round": -1,
                    "step": test_step,
                    "rebalance": int(did_rebalance),
                    **rb,
                }
            )
            regime = str(window_context.regimes.loc[date])
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
    eval_extras = _evaluation_extras(
        method_count=max(1, len(config.experiments.methods or config.experiments.modes or [config.top_mode])),
        test_dates=test_dates,
        test_returns=test_returns,
        reward_components=reward_components,
        liquidity_df=prepared.liquidity_df,
        cvar_alpha=config.reward.cvar_alpha,
    )
    return TrainingResult(
        provider=prepared.provider_label,
        mode=spec.method_id,
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
        evaluation_extras=eval_extras,
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
        "evaluation_extras": result.evaluation_extras,
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
        evaluation_extras=result.evaluation_extras,
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
    methods = config.experiments.methods or config.experiments.modes or [config.top_mode]
    seeds = config.experiments.seeds or [config.rl.random_seed]
    grid = [(m, s) for m in methods for s in seeds] or [(config.top_mode, config.rl.random_seed)]

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(config.experiments.results_dir) / f"{config.experiments.run_name}_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs: list[TrainingResult] = []
    summary_rows: list[dict[str, Any]] = []
    for idx, (method_id, seed) in enumerate(grid):
        print(f"[{idx + 1}/{len(grid)}] running method={method_id}, seed={seed}")
        run_result = _run_single(config=config, mode=method_id, seed=seed, prepared=prepared)
        run_dir = batch_dir / str(method_id) / f"seed_{seed}"
        run_result = save_training_result(run_result, run_dir)
        print(f"[{idx + 1}/{len(grid)}] saved artifacts: {run_dir}")
        runs.append(run_result)
        summary_rows.append(
            {
                "run": run_dir.name,
                "method": run_result.mode,
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
