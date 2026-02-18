import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from hmadrl.config import load_config
from hmadrl.data_api import StooqProvider, build_feature_bundle, build_states_from_features


class FakeProvider:
    def fetch(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        base = 100 + hash(symbol) % 10
        close = [base + i * 0.2 for i in range(len(idx))]
        volume = [1_000_000 + i * 1_000 for i in range(len(idx))]
        return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


class TestConfigAndData(unittest.TestCase):
    def test_load_config(self) -> None:
        payload = {
            "top_mode": "rl",
            "domain_to_stocks": {"tech": ["AAPL", "MSFT"]},
            "rl": {
                "max_domain_hold_steps": 5,
                "max_stock_hold_steps": 3,
                "random_seed": 1,
                "training_steps": 10,
                "stochastic_actions": True,
            },
            "data": {
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "interval": "1d",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.top_mode, "rl")
        self.assertEqual(config.rl.training_steps, 10)
        self.assertIn("tech", config.domain_to_stocks)
        self.assertEqual(config.data.provider, "auto")
        self.assertEqual(config.data.lookback, 20)
        self.assertEqual(config.data.train_split, 0.7)
        self.assertEqual(config.stochastic_control.risk_aversion, 3.0)
        self.assertEqual(config.experiments.results_dir, "results")

    def test_build_features_and_states(self) -> None:
        domain_to_stocks = {
            "tech": ["AAPL", "MSFT"],
            "finance": ["JPM", "BAC"],
        }
        features = build_feature_bundle(
            provider=FakeProvider(),
            domain_to_stocks=domain_to_stocks,
            start_date="2023-01-01",
            end_date="2024-01-01",
            interval="1d",
            lookback=10,
        )
        self.assertEqual(set(features.domain_features.keys()), {"tech", "finance"})
        self.assertIn("AAPL", features.stock_features["tech"])

        top_state, domain_states = build_states_from_features(
            domain_to_stocks=domain_to_stocks,
            features=features,
            remaining_horizon_steps=20,
            step_index=0,
        )
        self.assertEqual(top_state.remaining_horizon_steps, 20)
        self.assertEqual(set(domain_states.keys()), {"tech", "finance"})

    @patch("hmadrl.data_api.requests.get")
    def test_stooq_provider_parsing(self, mock_get: Mock) -> None:
        csv_text = (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,100,101,99,100.5,1200000\n"
            "2024-01-03,100.5,102,100,101.5,1300000\n"
        )
        response = Mock()
        response.text = csv_text
        response.raise_for_status = Mock()
        mock_get.return_value = response

        with tempfile.TemporaryDirectory() as tmp:
            provider = StooqProvider(cache_dir=tmp)
            frame = provider.fetch(
                symbol="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-31",
                interval="1d",
            )
        self.assertFalse(frame.empty)
        self.assertIn("Adj Close", frame.columns)
        self.assertIn("Volume", frame.columns)


if __name__ == "__main__":
    unittest.main()
