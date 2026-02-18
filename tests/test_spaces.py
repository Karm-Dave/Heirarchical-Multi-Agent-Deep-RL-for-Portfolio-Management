import unittest

from hmadrl.spaces import (
    DomainState,
    TopLevelState,
    generate_domain_action_templates,
    generate_top_level_action_templates,
)


class TestStateSpaces(unittest.TestCase):
    def test_top_level_state_vector_size(self) -> None:
        domains = ["tech", "finance", "healthcare"]
        state = TopLevelState(
            domain_momentum={"tech": 0.2, "finance": 0.1, "healthcare": -0.05},
            domain_volatility={"tech": 0.3, "finance": 0.2, "healthcare": 0.15},
            current_allocation={"tech": 0.3, "finance": 0.4, "healthcare": 0.3},
            remaining_horizon_steps=20,
            elapsed_steps=5,
            cash_buffer=0.1,
        )
        vector = state.to_vector(domains)
        self.assertEqual(len(vector), TopLevelState.vector_size(len(domains)))

    def test_domain_state_vector_size(self) -> None:
        stocks = ["A", "B", "C", "D"]
        state = DomainState(
            domain_name="tech",
            stock_momentum={s: 0.1 for s in stocks},
            stock_volatility={s: 0.2 for s in stocks},
            current_stock_allocation={s: 0.25 for s in stocks},
            remaining_domain_horizon_steps=8,
            elapsed_steps=2,
        )
        vector = state.to_vector(stocks)
        self.assertEqual(len(vector), DomainState.vector_size(len(stocks)))


class TestActionTemplates(unittest.TestCase):
    def test_top_templates_are_normalized(self) -> None:
        templates = generate_top_level_action_templates(
            domain_names=["tech", "energy"],
            max_hold_steps=3,
        )
        self.assertTrue(templates)
        for action in templates:
            self.assertAlmostEqual(sum(action.domain_weights.values()), 1.0, places=8)
            self.assertGreaterEqual(action.hold_steps, 1)
            self.assertLessEqual(action.hold_steps, 3)

    def test_domain_templates_are_normalized(self) -> None:
        templates = generate_domain_action_templates(
            domain_name="tech",
            stock_names=["AAPL", "MSFT", "NVDA"],
            max_hold_steps=4,
        )
        self.assertTrue(templates)
        for action in templates:
            self.assertAlmostEqual(sum(action.stock_weights.values()), 1.0, places=8)
            self.assertGreaterEqual(action.hold_steps, 1)
            self.assertLessEqual(action.hold_steps, 4)


if __name__ == "__main__":
    unittest.main()

