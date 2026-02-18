"""Market data ingestion and feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .spaces import DomainState, TopLevelState


class MarketDataProvider(Protocol):
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceProvider:
    """Yahoo Finance API provider."""

    def __init__(self, cache_dir: str | Path = "data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Path:
        safe = f"{symbol}_{start_date}_{end_date}_{interval}".replace(":", "-")
        return self.cache_dir / f"{safe}.csv"

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, start_date, end_date, interval)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached

        try:
            frame = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Yahoo download failed for {symbol}") from exc

        frame = frame.copy()
        frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
        if frame.empty:
            raise ValueError(f"No data returned for {symbol}")

        if "Adj Close" not in frame.columns and "Close" in frame.columns:
            frame["Adj Close"] = frame["Close"]
        if "Volume" not in frame.columns:
            frame["Volume"] = 0.0
        frame.to_csv(cache_path)
        return frame


class StooqProvider:
    """Stooq CSV endpoint provider (free, no key)."""

    def __init__(self, cache_dir: str | Path = "data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Path:
        safe = f"stooq_{symbol}_{start_date}_{end_date}_{interval}".replace(":", "-")
        return self.cache_dir / f"{safe}.csv"

    @staticmethod
    def _to_stooq_symbol(symbol: str) -> str:
        clean = symbol.strip().lower()
        if "." not in clean:
            clean = f"{clean}.us"
        return clean

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("StooqProvider currently supports interval='1d' only")

        cache_path = self._cache_path(symbol, start_date, end_date, interval)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached

        stooq_symbol = self._to_stooq_symbol(symbol)
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        if frame.empty or "Date" not in frame.columns:
            raise ValueError(f"No Stooq data returned for {symbol}")

        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
        frame = frame.loc[(frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))]
        if frame.empty:
            raise ValueError(f"No Stooq data in requested range for {symbol}")
        if "Adj Close" not in frame.columns and "Close" in frame.columns:
            frame["Adj Close"] = frame["Close"]
        if "Volume" not in frame.columns:
            frame["Volume"] = 0.0
        frame.to_csv(cache_path)
        return frame


class AutoMarketDataProvider:
    """Try multiple real-data providers in order."""

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("providers cannot be empty")
        self.providers = providers

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return provider.fetch(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"All providers failed for {symbol}") from last_error


@dataclass(frozen=True)
class SymbolFeatures:
    momentum: float
    volatility: float
    liquidity: float


@dataclass(frozen=True)
class FeatureBundle:
    domain_features: Mapping[str, SymbolFeatures]
    stock_features: Mapping[str, Mapping[str, SymbolFeatures]]


def _extract_series(frame: pd.DataFrame, preferred: list[str]) -> pd.Series:
    for name in preferred:
        if name in frame.columns:
            return frame[name].dropna()
    raise ValueError(f"Missing columns: expected one of {preferred}")


def compute_features(frame: pd.DataFrame, lookback: int = 20) -> SymbolFeatures:
    close = _extract_series(frame, ["Adj Close", "Close"])
    volume = _extract_series(frame, ["Volume"])
    if len(close) < 3:
        raise ValueError("Not enough rows to compute features")

    returns = close.pct_change().dropna()
    lookback = max(2, min(lookback, len(close) - 1))
    recent = close.iloc[-lookback:]
    momentum = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
    volatility = float(returns.iloc[-lookback:].std() * np.sqrt(252.0))
    liquidity = float(volume.iloc[-lookback:].mean())
    return SymbolFeatures(momentum=momentum, volatility=volatility, liquidity=liquidity)


def build_feature_bundle(
    provider: MarketDataProvider,
    domain_to_stocks: Mapping[str, list[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    lookback: int = 20,
) -> FeatureBundle:
    per_domain: dict[str, SymbolFeatures] = {}
    per_stock: dict[str, dict[str, SymbolFeatures]] = {}

    for domain, symbols in domain_to_stocks.items():
        stock_map: dict[str, SymbolFeatures] = {}
        for symbol in symbols:
            frame = provider.fetch(symbol, start_date=start_date, end_date=end_date, interval=interval)
            stock_map[symbol] = compute_features(frame, lookback=lookback)
        per_stock[domain] = stock_map

        domain_momentum = float(np.mean([v.momentum for v in stock_map.values()]))
        domain_volatility = float(np.mean([v.volatility for v in stock_map.values()]))
        domain_liquidity = float(np.mean([v.liquidity for v in stock_map.values()]))
        per_domain[domain] = SymbolFeatures(
            momentum=domain_momentum,
            volatility=domain_volatility,
            liquidity=domain_liquidity,
        )

    return FeatureBundle(domain_features=per_domain, stock_features=per_stock)


def build_states_from_features(
    domain_to_stocks: Mapping[str, list[str]],
    features: FeatureBundle,
    remaining_horizon_steps: int,
    step_index: int,
    current_domain_allocation: Mapping[str, float] | None = None,
    current_stock_allocations: Mapping[str, Mapping[str, float]] | None = None,
    cash_ratio: float = 0.0,
) -> tuple[TopLevelState, dict[str, DomainState]]:
    domain_alloc = dict(current_domain_allocation or {})
    if not domain_alloc:
        equal = 1.0 / len(domain_to_stocks)
        domain_alloc = {domain: equal for domain in domain_to_stocks}

    top_state = TopLevelState(
        domain_momentum={d: features.domain_features[d].momentum for d in domain_to_stocks},
        domain_volatility={d: features.domain_features[d].volatility for d in domain_to_stocks},
        domain_liquidity={d: features.domain_features[d].liquidity for d in domain_to_stocks},
        current_allocation=domain_alloc,
        remaining_horizon_steps=remaining_horizon_steps,
        step_index=step_index,
        cash_ratio=cash_ratio,
    )

    domain_states: dict[str, DomainState] = {}
    for domain, symbols in domain_to_stocks.items():
        stock_alloc = dict((current_stock_allocations or {}).get(domain, {}))
        if not stock_alloc:
            equal = 1.0 / len(symbols)
            stock_alloc = {symbol: equal for symbol in symbols}
        stock_map = features.stock_features[domain]
        domain_states[domain] = DomainState(
            domain_name=domain,
            stock_momentum={s: stock_map[s].momentum for s in symbols},
            stock_volatility={s: stock_map[s].volatility for s in symbols},
            stock_liquidity={s: stock_map[s].liquidity for s in symbols},
            current_stock_allocation=stock_alloc,
            remaining_domain_steps=remaining_horizon_steps,
            step_index=step_index,
        )
    return top_state, domain_states
