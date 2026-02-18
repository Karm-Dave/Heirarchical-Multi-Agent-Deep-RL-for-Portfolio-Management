import unittest

from hmadrl.spaces import (
    DomainState,
    TopLevelState,
    generate_domain_templates,
    generate_top_level_templates,
)


class TestSpaces(unittest.TestCase):
    def test_top_state_vector_size(self) -> None:
        domains = ["tech", "energy", "health"]
        state = TopLevelState(
            domain_momentum={"tech": 0.2, "energy": -0.1, "health": 0.05},
            domain_volatility={"tech": 0.3, "energy": 0.5, "health": 0.2},
            domain_liquidity={"tech": 0.9, "energy": 0.7, "health": 0.8},
            current_allocation={"tech": 0.4, "energy": 0.2, "health": 0.4},
            remaining_horizon_steps=12,
            step_index=3,
            cash_ratio=0.05,
        )
        vec = state.to_vector(domains)
        self.assertEqual(len(vec), TopLevelState.vector_size(len(domains)))

    def test_domain_state_vector_size(self) -> None:
        stocks = ["A", "B", "C"]
        state = DomainState(
            domain_name="tech",
            stock_momentum={"A": 0.1, "B": -0.2, "C": 0.05},
            stock_volatility={"A": 0.3, "B": 0.4, "C": 0.25},
            stock_liquidity={"A": 0.8, "B": 0.6, "C": 0.9},
            current_stock_allocation={"A": 0.4, "B": 0.3, "C": 0.3},
            remaining_domain_steps=5,
            step_index=2,
        )
        vec = state.to_vector(stocks)
        self.assertEqual(len(vec), DomainState.vector_size(len(stocks)))

    def test_templates_are_normalized(self) -> None:
        top_templates = generate_top_level_templates(["tech", "energy"], max_hold_steps=3)
        self.assertTrue(top_templates)
        for action in top_templates:
            self.assertAlmostEqual(sum(action.domain_weights.values()), 1.0, places=6)
            self.assertGreaterEqual(action.hold_steps, 1)
            self.assertLessEqual(action.hold_steps, 3)

        domain_templates = generate_domain_templates(
            domain_name="tech",
            stock_names=["AAPL", "MSFT", "NVDA"],
            max_hold_steps=4,
        )
        self.assertTrue(domain_templates)
        for action in domain_templates:
            self.assertAlmostEqual(sum(action.stock_weights.values()), 1.0, places=6)
            self.assertGreaterEqual(action.hold_steps, 1)
            self.assertLessEqual(action.hold_steps, 4)


if __name__ == "__main__":
    unittest.main()

