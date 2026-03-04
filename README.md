# Hierarchical Multi-Agent Deep RL for Portfolio Management

End-to-end hierarchical portfolio system with:

- A top allocator (switchable RL or MoE router)
- Multiple lower allocation agents (domain/cluster level)
- Stochastic control constraints
- Walk-forward training/evaluation on real market data

The project is designed for experimentation and reproducible research with modular components in `hmadrl/`.

## What This Repo Does

1. Fetches real data (`stooq` and/or `yfinance`) for stocks + macro/factor symbols.
2. Builds rich multi-scale features (stock, domain, global).
3. Builds hierarchical states:
   - `TopLevelState` for the top manager
   - `DomainState` for each lower manager
4. Produces hierarchical actions:
   - top domain weights + hold horizon
   - per-domain stock weights + hold horizon
5. Applies stochastic control mediation (`capital_budget`, `risk_budget`, caps, hold constraints).
6. Runs leakage-safe walk-forward train/test windows with:
   - rolling adaptive clustering fit on train windows only
   - train-window-only normalization (including regime-conditional scaling)
   - lagged macro/global features
7. Saves metrics, CSVs, plots, and summaries to `results/`.

## Methods In Main Pipeline

The batch runner now executes method-level experiments (each saved under its own subfolder):

- Hierarchical: `rl`, `moe_router`
- Baselines: `equal_weight`, `risk_parity`, `hrp`, `flat_ppo`, `flat_sac`, `lstm_portfolio`, `transformer_portfolio`
- Ablations:
  - `ablation_no_learned_clusters`
  - `ablation_no_stochastic_control`
  - `ablation_no_hold_enforcement`
  - `ablation_no_global_macro`
  - `ablation_no_liquidity`
  - `ablation_static_sectors`

## Architecture

### Top Layer (Strategy Allocation)

- `mode=rl`: `RLTopManager` (continuous PPO ensemble; optional transformer top network).
- `mode=moe_router`: `MoERouterTopManager` (router selecting experts: momentum/low-vol/liquidity).

Output:
- domain/cluster allocation weights
- top hold duration

### Lower Layer (Stock Allocation)

- `DomainRLManager` uses continuous SAC ensemble.
- Produces stock-level weights + stock hold duration.
- Post-processed with stochastic control constraints.

### Control Layer

`StochasticController` computes dynamic per-domain controls from momentum/vol/liquidity + global stress:

- `capital_budget`
- `risk_budget`
- `max_stock_weight`
- `hold_steps`

## Data And Features

The pipeline supports:

- Multi-horizon momentum and volatility
- Liquidity and microstructure proxies
- Cross-sectional stock signals
- Domain/cluster factors
- Global regime and macro signals

Data providers:

- `stooq` (free)
- `yfinance` (free)
- `auto` fallback chain (`stooq -> yfinance`)

## Main Files

- `hmadrl/pipeline.py`: data prep, feature engineering, walk-forward train/test, artifacts
- `hmadrl/top_manager.py`: top-layer RL and MoE router logic
- `hmadrl/domain_manager.py`: lower-layer SAC manager
- `hmadrl/stochastic_control.py`: risk mediation/control equations
- `hmadrl/rl_core.py`: PPO/SAC implementations
- `hmadrl/factory.py`: manager/agent assembly by mode
- `run_experiment.py`: CLI entry point

## Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run a quick smoke batch:

```powershell
python run_experiment.py --config config/smoke_config.json
```

Run full batch:

```powershell
python run_experiment.py --config config/default_config.json
```

Run a single experiment only:

```powershell
python run_experiment.py --config config/default_config.json --single
```

## Outputs

Results are written to:

`results/<run_name>_<timestamp>/`

Batch-level artifacts:

- `batch_summary.csv`
- `batch_summary.json`
- `batch_cumulative_returns.png`

Per-run artifacts:

- `summary.json`
- `train_returns.csv`, `test_returns.csv`
- `losses.csv` (if available)
- `reward_components.csv`
- `walk_forward_windows.csv`
- evaluation extras in `summary.json`:
  - Newey-West t-stat
  - Deflated Sharpe estimate
  - Bootstrap confidence intervals
  - Crisis subperiod metrics
  - Turnover decomposition
  - Capacity/liquidity scaling analysis
- plots: reward, equity, drawdown, allocation, regime returns

## Branch Notes

- `master`: stable baseline
- `nondomain`: active research branch (newer modeling/feature experiments)

To switch:

```powershell
git switch master
# or
git switch nondomain
```

## Disclaimer

This repository is for research/engineering experimentation, not financial advice. Live trading requires additional execution, risk, and compliance controls.
