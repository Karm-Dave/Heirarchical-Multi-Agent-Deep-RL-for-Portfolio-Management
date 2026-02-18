# HMADRL

Hierarchical multi-agent deep reinforcement learning portfolio framework with two layers:

- `Portfolio Manager` (top layer): allocates capital to domains and chooses domain holding period.
- `Domain Manager` (bottom layer): allocates within each domain to stocks and chooses stock holding period.

Top-layer strategy is switchable between:

- `RL` stochastic-control style decision making.
- `MoE` (Mixture of Experts) allocation.

## State And Action Spaces

### Top Layer: Portfolio Manager

- State (`TopLevelState`):
  - `domain_momentum[d]` for each domain.
  - `domain_volatility[d]` for each domain.
  - `current_allocation[d]` for each domain.
  - `remaining_horizon_steps`.
  - `elapsed_steps`.
  - `cash_buffer`.
- Action (`TopLevelAction`):
  - `domain_weights[d]` (capital split across domains).
  - `hold_steps` (how long to keep the domain allocation).

### Domain Layer: Domain Manager

- State (`DomainState`):
  - `stock_momentum[s]` for each stock in that domain.
  - `stock_volatility[s]` for each stock in that domain.
  - `current_stock_allocation[s]`.
  - `remaining_domain_horizon_steps`.
  - `elapsed_steps`.
- Action (`DomainAction`):
  - `stock_weights[s]` (capital split among domain stocks).
  - `hold_steps` (stock-level holding period, capped by top-layer `hold_steps`).

## Development

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

