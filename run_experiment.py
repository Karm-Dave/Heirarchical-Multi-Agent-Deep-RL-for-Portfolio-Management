from __future__ import annotations

import argparse
from statistics import mean

from hmadrl.pipeline import run_training


def _print_metrics(label: str, metrics) -> None:
    print(f"{label} periods: {metrics.num_periods}")
    print(f"{label} mean period return: {metrics.mean_period_return:.6f}")
    print(f"{label} cumulative return: {metrics.cumulative_return:.6f}")
    print(f"{label} annualized return: {metrics.annualized_return:.6f}")
    print(f"{label} annualized volatility: {metrics.annualized_volatility:.6f}")
    print(f"{label} sharpe: {metrics.sharpe:.6f}")
    print(f"{label} max drawdown: {metrics.max_drawdown:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMADRL training with API data.")
    parser.add_argument(
        "--config",
        default="config/default_config.json",
        help="Path to config JSON file.",
    )
    args = parser.parse_args()

    result = run_training(args.config)
    print(f"Data provider: {result.provider}")
    print(f"Training steps: {len(result.train_rewards)}")
    avg_reward = mean(result.train_rewards) if result.train_rewards else 0.0
    print(f"Average reward: {avg_reward:.6f}")
    _print_metrics("Train", result.train_metrics)
    _print_metrics("Test", result.test_metrics)
    print(f"Last top hold steps: {result.last_decision.top_action.hold_steps}")
    print("Last top domain weights:")
    for domain, weight in result.last_decision.top_action.domain_weights.items():
        print(f"  {domain}: {weight:.4f}")


if __name__ == "__main__":
    main()
