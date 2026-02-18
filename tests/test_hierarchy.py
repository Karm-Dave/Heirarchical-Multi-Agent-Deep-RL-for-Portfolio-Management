import unittest

from hmadrl.factory import build_hierarchical_agent
from hmadrl.spaces import DomainState, TopLevelState


class TestHierarchy(unittest.TestCase):
    def test_decision_respects_hold_hierarchy(self) -> None:
        domain_to_stocks = {
            "tech": ["AAPL", "MSFT"],
            "finance": ["JPM", "BAC"],
        }
        agent = build_hierarchical_agent(
            mode="moe_router",
            domain_to_stocks=domain_to_stocks,
            max_domain_hold_steps=6,
            max_stock_hold_steps=4,
            seed=5,
        )

        top_state = TopLevelState(
            domain_momentum={"tech": 0.3, "finance": 0.1},
            domain_volatility={"tech": 0.4, "finance": 0.2},
            domain_liquidity={"tech": 0.9, "finance": 0.8},
            current_allocation={"tech": 0.5, "finance": 0.5},
            remaining_horizon_steps=10,
            step_index=1,
            cash_ratio=0.0,
        )
        domain_states = {
            "tech": DomainState(
                domain_name="tech",
                stock_momentum={"AAPL": 0.2, "MSFT": 0.1},
                stock_volatility={"AAPL": 0.3, "MSFT": 0.25},
                stock_liquidity={"AAPL": 0.8, "MSFT": 0.85},
                current_stock_allocation={"AAPL": 0.5, "MSFT": 0.5},
                remaining_domain_steps=10,
                step_index=1,
            ),
            "finance": DomainState(
                domain_name="finance",
                stock_momentum={"JPM": 0.12, "BAC": 0.09},
                stock_volatility={"JPM": 0.22, "BAC": 0.26},
                stock_liquidity={"JPM": 0.7, "BAC": 0.75},
                current_stock_allocation={"JPM": 0.4, "BAC": 0.6},
                remaining_domain_steps=10,
                step_index=1,
            ),
        }

        decision = agent.decide(top_state=top_state, domain_states=domain_states, stochastic=False)
        self.assertTrue(decision.final_stock_weights)
        self.assertAlmostEqual(sum(decision.final_stock_weights.values()), 1.0, places=6)

        for domain, domain_action in decision.domain_actions.items():
            self.assertLessEqual(domain_action.hold_steps, decision.top_action.hold_steps)
            self.assertIn(domain, decision.stock_rebalance_after_steps)


if __name__ == "__main__":
    unittest.main()

