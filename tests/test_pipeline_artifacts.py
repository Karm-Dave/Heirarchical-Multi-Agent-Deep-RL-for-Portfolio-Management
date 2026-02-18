from pathlib import Path
import tempfile
import unittest

from hmadrl.hierarchy import HierarchicalDecision
from hmadrl.pipeline import BacktestMetrics, TrainingResult, save_training_result
from hmadrl.spaces import DomainAction, TopLevelAction
from hmadrl.stochastic_control import DomainControlSignal


class TestPipelineArtifacts(unittest.TestCase):
    def test_save_training_result_outputs_files(self) -> None:
        decision = HierarchicalDecision(
            top_action=TopLevelAction(
                domain_weights={"tech": 0.6, "finance": 0.4},
                hold_steps=3,
                action_id="unit",
            ),
            domain_controls={
                "tech": DomainControlSignal(
                    capital_budget=0.6,
                    risk_budget=0.6,
                    max_stock_weight=0.7,
                    hold_steps=3,
                    merton_fraction=0.2,
                    uncertainty=0.4,
                ),
                "finance": DomainControlSignal(
                    capital_budget=0.4,
                    risk_budget=0.4,
                    max_stock_weight=0.7,
                    hold_steps=2,
                    merton_fraction=0.1,
                    uncertainty=0.5,
                ),
            },
            final_domain_weights={"tech": 0.6, "finance": 0.4},
            domain_actions={
                "tech": DomainAction(
                    stock_weights={"AAPL": 0.5, "MSFT": 0.5},
                    hold_steps=3,
                    action_id="tech",
                ),
                "finance": DomainAction(
                    stock_weights={"JPM": 0.6, "BAC": 0.4},
                    hold_steps=2,
                    action_id="fin",
                ),
            },
            final_stock_weights={
                "tech:AAPL": 0.3,
                "tech:MSFT": 0.3,
                "finance:JPM": 0.24,
                "finance:BAC": 0.16,
            },
            domain_rebalance_after_steps=3,
            stock_rebalance_after_steps={"tech": 3, "finance": 2},
        )
        metrics = BacktestMetrics(
            num_periods=3,
            mean_period_return=0.01,
            cumulative_return=0.03,
            annualized_return=0.2,
            annualized_volatility=0.15,
            sharpe=1.2,
            max_drawdown=0.05,
        )
        result = TrainingResult(
            provider="test",
            mode="rl",
            seed=1,
            train_rewards=[0.01, 0.02, -0.01],
            train_period_returns=[0.01, 0.02, -0.01],
            test_period_returns=[0.01, -0.02],
            losses=[{"top": 0.1}, {"top": 0.09}, {"top": 0.08}],
            train_metrics=metrics,
            test_metrics=metrics,
            train_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            test_dates=["2024-01-04", "2024-01-05"],
            train_domain_allocations=[{"tech": 0.6, "finance": 0.4}] * 3,
            test_domain_allocations=[{"tech": 0.5, "finance": 0.5}] * 2,
            last_decision=decision,
        )

        with tempfile.TemporaryDirectory() as tmp:
            saved = save_training_result(result, Path(tmp) / "run")
            out = Path(saved.output_dir or "")
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "train_returns.csv").exists())
            self.assertTrue((out / "test_returns.csv").exists())
            self.assertTrue((out / "reward_curve.png").exists())
            self.assertTrue((out / "equity_curve.png").exists())
            self.assertTrue((out / "drawdown_curve.png").exists())
            self.assertTrue((out / "train_domain_allocation.png").exists())
            self.assertTrue((out / "test_domain_allocation.png").exists())


if __name__ == "__main__":
    unittest.main()

