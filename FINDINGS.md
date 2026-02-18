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
- Added executable training entrypoint (`run_experiment.py`) and pipeline (`hmadrl/pipeline.py`) with train/test return backtesting metrics.
- Confirmed in this environment: `yfinance` can fail due Yahoo consent/DNS; `stooq` endpoint is reachable and usable in `auto` mode.

## Test Status

- Full suite passes:
  - `test_spaces.py`
  - `test_factory.py`
  - `test_hierarchy.py`
  - `test_rl_core.py`
  - `test_config_and_data.py`
