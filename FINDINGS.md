# Findings Log

## Rebuild Pass

- Removed previous interrupted scaffold and rebuilt from scratch.
- Added dependency-managed RL stack (`torch`, `numpy`, `pandas`) plus API data layer (`yfinance`, `stooq`).
- Implemented two-layer hierarchy:
  - Top layer switchable between `RLTopManager` and `MoERouterTopManager`.
  - Domain layer uses `DomainRLManager` per domain.
- Added explicit state/action space classes and template action spaces including hold durations.
- Added config system (`config/default_config.json`, `hmadrl/config.py`) for tunable experiments.
- Added API data ingestion + feature engineering + state builder (`hmadrl/data_api.py`).
- Added stochastic-control layer (`hmadrl/stochastic_control.py`) with OU-style latent control + Merton-style allocation term.
- Enforced true hierarchy: top manager now constrains domain agents via control signals (`capital_budget`, `risk_budget`, `max_stock_weight`, `hold_steps`).
- Added executable batch experiment runner (`run_experiment.py`) and expanded pipeline (`hmadrl/pipeline.py`) with:
  - multi-run mode/seed sweeps
  - results folder artifacts
  - CSV/JSON metrics
  - matplotlib plots (reward, equity, drawdown, domain allocations)
- Confirmed in this environment: `yfinance` can fail due Yahoo consent/DNS; `stooq` endpoint is reachable and usable in `auto` mode.
- Switched `auto` provider order to `stooq -> yfinance` for faster and more reliable live runs in this environment.
- Expanded default universe to 6 domains with 30 US stocks total.

## Batch Run Example

- Latest batch folder: `results/hmadrl_batch_20260218_103754`
- Completed runs: 4 (`rl` and `moe_router` over seeds `42` and `84`)
- Best test cumulative return in that run: `0.162814` (`moe_router`, seed `42`)

## Test Status

- Full suite passes:
  - `test_spaces.py`
  - `test_factory.py`
  - `test_hierarchy.py`
  - `test_rl_core.py`
  - `test_config_and_data.py`
  - `test_stochastic_control.py`
  - `test_pipeline_artifacts.py`
