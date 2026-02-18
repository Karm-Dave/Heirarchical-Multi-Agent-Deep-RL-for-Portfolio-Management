import unittest

from hmadrl.factory import build_top_manager
from hmadrl.top_manager import MoERouterTopManager, RLTopManager


class TestFactory(unittest.TestCase):
    def test_rl_mode(self) -> None:
        manager = build_top_manager(
            mode="rl",
            domain_names=["tech", "energy"],
            max_hold_steps=5,
            seed=7,
        )
        self.assertIsInstance(manager, RLTopManager)

    def test_moe_router_mode(self) -> None:
        manager = build_top_manager(
            mode="moe_router",
            domain_names=["tech", "energy"],
            max_hold_steps=5,
            seed=7,
        )
        self.assertIsInstance(manager, MoERouterTopManager)


if __name__ == "__main__":
    unittest.main()

