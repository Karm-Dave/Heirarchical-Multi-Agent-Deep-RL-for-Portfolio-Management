import unittest

import numpy as np

from hmadrl.rl_core import DQNPolicy


class TestRLCore(unittest.TestCase):
    def test_dqn_train_step_runs(self) -> None:
        policy = DQNPolicy(
            state_dim=6,
            action_dim=4,
            hidden_dims=(32, 32),
            batch_size=8,
            replay_size=128,
            seed=11,
        )

        rng = np.random.default_rng(11)
        for _ in range(50):
            state = rng.normal(size=6).astype(np.float32)
            next_state = state + rng.normal(scale=0.1, size=6).astype(np.float32)
            action = int(rng.integers(0, 4))
            reward = float(rng.normal())
            done = bool(rng.integers(0, 2))
            policy.store(state, action, reward, next_state, done)

        loss = policy.train_step()
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()

