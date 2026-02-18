# HMADRL Portfolio Agent

Hierarchical multi-agent portfolio framework with:

- Top layer (`Portfolio Manager`): allocates capital across domains and chooses domain hold duration.
- Domain layer (`Domain Managers`): allocates capital among stocks within each domain and chooses stock hold duration.

Top layer is switchable:

- `rl`: DQN-based top allocator.
- `moe_router`: MoE-style router logic (router chooses expert allocator).

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
- `auto` mode tries `yfinance` first, then `stooq`

## Config

Edit `config/default_config.json` to change:

- top mode (`rl` or `moe_router`)
- domains and ticker sets
- hold horizons
- training steps
- data range and interval
- provider mode (`yfinance`, `stooq`, or `auto`)
- lookback window / train-test split / cache directory

## Run

Install dependencies in venv:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Run tests:

```bash
.venv\Scripts\python -m unittest discover -s tests -v
```

Run training:

```bash
.venv\Scripts\python run_experiment.py --config config/default_config.json
```

The run output includes:

- training rewards
- training return metrics
- test return metrics (`cumulative return`, `annualized return`, `sharpe`, `max drawdown`)
