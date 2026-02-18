"""Config loader for easy experimentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RLConfig:
    max_domain_hold_steps: int
    max_stock_hold_steps: int
    random_seed: int
    training_steps: int
    stochastic_actions: bool


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str
    interval: str
    provider: str
    lookback: int
    train_split: float
    cache_dir: str


@dataclass(frozen=True)
class ProjectConfig:
    top_mode: str
    domain_to_stocks: dict[str, list[str]]
    rl: RLConfig
    data: DataConfig


def _require_keys(payload: dict[str, Any], keys: list[str], scope: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"Missing keys in {scope}: {missing}")


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

    return ProjectConfig(
        top_mode=str(raw["top_mode"]),
        domain_to_stocks={
            str(domain): [str(symbol) for symbol in symbols]
            for domain, symbols in dict(raw["domain_to_stocks"]).items()
        },
        rl=RLConfig(
            max_domain_hold_steps=int(raw["rl"]["max_domain_hold_steps"]),
            max_stock_hold_steps=int(raw["rl"]["max_stock_hold_steps"]),
            random_seed=int(raw["rl"]["random_seed"]),
            training_steps=int(raw["rl"]["training_steps"]),
            stochastic_actions=bool(raw["rl"]["stochastic_actions"]),
        ),
        data=DataConfig(
            start_date=str(raw["data"]["start_date"]),
            end_date=str(raw["data"]["end_date"]),
            interval=str(raw["data"]["interval"]),
            provider=str(raw["data"].get("provider", "auto")),
            lookback=int(raw["data"].get("lookback", 20)),
            train_split=float(raw["data"].get("train_split", 0.7)),
            cache_dir=str(raw["data"].get("cache_dir", "data_cache")),
        ),
    )
