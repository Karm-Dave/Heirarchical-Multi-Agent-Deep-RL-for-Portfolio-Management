import unittest

from hmadrl.config import StochasticControlConfig
from hmadrl.spaces import TopLevelAction, TopLevelState
from hmadrl.stochastic_control import StochasticController


class TestStochasticControl(unittest.TestCase):
    def test_domain_controls_normalized_and_bounded(self) -> None:
        controller = StochasticController(
            domain_names=["tech", "finance", "energy"],
            config=StochasticControlConfig(
                risk_aversion=2.5,
                hold_scale=1.2,
                uncertainty_penalty=0.4,
                mean_reversion_speed=0.2,
                max_single_stock_weight=0.5,
                min_domain_allocation=0.0,
            ),
            max_hold_steps=8,
            seed=9,
        )
        state = TopLevelState(
            domain_momentum={"tech": 0.2, "finance": 0.1, "energy": -0.02},
            domain_volatility={"tech": 0.35, "finance": 0.2, "energy": 0.5},
            domain_liquidity={"tech": 0.9, "finance": 0.8, "energy": 0.65},
            current_allocation={"tech": 0.4, "finance": 0.4, "energy": 0.2},
            remaining_horizon_steps=20,
            step_index=2,
            cash_ratio=0.0,
        )
        top_action = TopLevelAction(
            domain_weights={"tech": 0.5, "finance": 0.3, "energy": 0.2},
            hold_steps=6,
            action_id="test",
        )
        controls = controller.build_controls(state=state, top_action=top_action)
        self.assertEqual(set(controls.keys()), {"tech", "finance", "energy"})
        self.assertAlmostEqual(sum(c.capital_budget for c in controls.values()), 1.0, places=6)
        for control in controls.values():
            self.assertGreaterEqual(control.hold_steps, 1)
            self.assertLessEqual(control.hold_steps, 6)
            self.assertGreaterEqual(control.max_stock_weight, 0.1)
            self.assertLessEqual(control.max_stock_weight, 0.95)


if __name__ == "__main__":
    unittest.main()

