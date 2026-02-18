# HMADRL Portfolio Agent

Hierarchical multi-agent portfolio framework with:

- Top layer (`Portfolio Manager`): allocates capital across domains and chooses domain hold duration.
- Domain layer (`Domain Managers`): allocates capital among stocks within each domain and chooses stock hold duration.

Top layer is switchable:

- `rl`: DQN-based top allocator.
- `moe_router`: MoE-style router logic (router chooses expert allocator).

Top-down stochastic control is active:

- Portfolio manager emits domain control signals (`capital_budget`, `risk_budget`, `max_stock_weight`, `hold_steps`).
- Domain managers must satisfy those constraints when selecting stock actions.

## State And Action Spaces

### Top Layer

- State (`TopLevelState`):
  - `domain_momentum[d]`
  - `domain_volatility[d]`
  - `domain_liquidity[d]`
  - `current_allocation[d]`
  - `remaining_horizon_steps`
  - `step_index`
  - `cash_ratio`
- Action (`TopLevelAction`):
  - `domain_weights[d]`
  - `hold_steps` (domain-level investment duration)

### Domain Layer

- State (`DomainState`):
  - `stock_momentum[s]`
  - `stock_volatility[s]`
  - `stock_liquidity[s]`
  - `current_stock_allocation[s]`
  - `remaining_domain_steps`
  - `step_index`
- Action (`DomainAction`):
  - `stock_weights[s]`
  - `hold_steps` (stock-level investment duration, capped by top-layer hold)

## Data API

Real market data is fetched through free providers (`hmadrl/data_api.py`):

- `yfinance` (Yahoo)
- `stooq` CSV endpoint
- `auto` mode tries `stooq` first, then `yfinance`

## Config

Edit `config/default_config.json` to change:

- top mode (`rl` or `moe_router`)
- domains and ticker sets
- hold horizons
- training steps
- data range and interval
- provider mode (`yfinance`, `stooq`, or `auto`)
- lookback window / train-test split / cache directory
- stochastic-control parameters
- batch experiment settings (`modes`, `seeds`, `results_dir`, `run_name`)

## Run

Install dependencies in venv:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Run tests:

```bash
.venv\Scripts\python -m unittest discover -s tests -v
```

Run batch experiments (default):

```bash
.venv\Scripts\python run_experiment.py --config config/default_config.json
```

Run only one experiment:

```bash
.venv\Scripts\python run_experiment.py --config config/default_config.json --single
```

Outputs are saved under `results/<run_name>_<timestamp>/`:

- `batch_summary.csv`, `batch_summary.json`
- per-run folder with:
  - `summary.json`
  - `train_returns.csv`, `test_returns.csv`, `losses.csv`
  - `train_domain_allocations.csv`, `test_domain_allocations.csv`
  - plots: `reward_curve.png`, `equity_curve.png`, `drawdown_curve.png`, and domain allocation charts
