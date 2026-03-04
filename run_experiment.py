from __future__ import annotations

import argparse
from statistics import mean

from hmadrl.pipeline import run_multiple_experiments, run_training


def _print_metrics(label: str, metrics) -> None:
    print(f"{label} periods: {metrics.num_periods}")
    print(f"{label} mean period return: {metrics.mean_period_return:.6f}")
    print(f"{label} cumulative return: {metrics.cumulative_return:.6f}")
    print(f"{label} annualized return: {metrics.annualized_return:.6f}")
    print(f"{label} annualized volatility: {metrics.annualized_volatility:.6f}")
    print(f"{label} sharpe: {metrics.sharpe:.6f}")
    print(f"{label} max drawdown: {metrics.max_drawdown:.6f}")


def _print_single(result) -> None:
    print(f"Data provider: {result.provider}")
    print(f"Mode/Seed: {result.mode}/{result.seed}")
    print(f"Training steps: {len(result.train_rewards)}")
    avg_reward = mean(result.train_rewards) if result.train_rewards else 0.0
    print(f"Average reward: {avg_reward:.6f}")
    _print_metrics("Train", result.train_metrics)
    _print_metrics("Test", result.test_metrics)
    print(f"Last top hold steps: {result.last_decision.top_action.hold_steps}")
    print("Last top domain weights:")
    for domain, weight in result.last_decision.top_action.domain_weights.items():
        print(f"  {domain}: {weight:.4f}")
    if result.output_dir:
        print(f"Saved run artifacts: {result.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMADRL experiments.")
    parser.add_argument(
        "--config",
        default="config/default_config.json",
        help="Path to config JSON file.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run one experiment only (top_mode + random_seed).",
    )
    args = parser.parse_args()

    if args.single:
        result = run_training(args.config)
        _print_single(result)
        return

    batch = run_multiple_experiments(args.config)
    print(f"Data provider: {batch.provider}")
    print(f"Batch results folder: {batch.batch_dir}")
    print(f"Runs completed: {len(batch.runs)}")
    best = max(batch.runs, key=lambda x: x.test_metrics.cumulative_return)
    print("Best run by test cumulative return:")
    print(f"  method={best.mode}, seed={best.seed}, test_cum_return={best.test_metrics.cumulative_return:.6f}")
    if best.output_dir:
        print(f"  artifacts={best.output_dir}")


if __name__ == "__main__":
    main()
