import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hmadrl.config import load_config
from hmadrl.pipeline import (
    PreparedData,
    _classify_regime,
    _method_from_name,
    _normalize_frame_by_window,
    _resolve_window_domain_map,
)


def _build_config() -> object:
    payload = {
        "top_mode": "rl",
        "domain_to_stocks": {
            "d1": ["A", "B"],
            "d2": ["C", "D"],
        },
        "rl": {
            "max_domain_hold_steps": 4,
            "max_stock_hold_steps": 3,
            "random_seed": 7,
            "training_steps": 10,
            "stochastic_actions": True,
        },
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "interval": "1d",
            "use_learned_clusters": True,
            "learned_cluster_count": 2,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(path)


class TestTemporalClean(unittest.TestCase):
    def test_regime_labels_do_not_depend_on_future_samples(self) -> None:
        dates = pd.date_range("2020-01-01", periods=120, freq="D")
        ret = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
        vol = pd.Series(np.concatenate([np.full(90, 0.2), np.full(30, 4.0)]), index=dates)

        full = _classify_regime(ret, vol)
        cut = _classify_regime(ret.iloc[:95], vol.iloc[:95])
        check_date = dates[85]
        self.assertEqual(str(full.loc[check_date]), str(cut.loc[check_date]))

    def test_window_clustering_uses_train_slice_only(self) -> None:
        cfg = _build_config()
        dates = pd.date_range("2021-01-01", periods=120, freq="D")
        rng = np.random.default_rng(11)

        a1 = rng.normal(0.0, 0.01, size=60)
        b1 = a1 + rng.normal(0.0, 0.0005, size=60)
        c1 = rng.normal(0.0, 0.02, size=60)

        a2 = rng.normal(0.0, 0.01, size=60)
        c2 = a2 + rng.normal(0.0, 0.0005, size=60)
        b2 = -a2 + rng.normal(0.0, 0.0005, size=60)

        a = np.concatenate([a1, a2])
        b = np.concatenate([b1, b2])
        c = np.concatenate([c1, c2])
        d = rng.normal(0.0, 0.02, size=120)
        ret = pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=dates)

        ones = pd.DataFrame(1.0, index=dates, columns=["A", "B", "C", "D"])
        global_df = pd.DataFrame({"market_vol": np.linspace(0.1, 0.3, len(dates))}, index=dates)
        regimes = pd.Series("sideways", index=dates)
        prepared = PreparedData(
            domain_to_stocks={"d1": ["A", "B"], "d2": ["C", "D"]},
            momentum_df=ones,
            volatility_df=ones,
            liquidity_df=ones,
            stock_feature_frames={"f": ones},
            stock_feature_names=["f"],
            next_returns_df=ret.shift(-1).fillna(0.0),
            global_feature_df=global_df,
            domain_factor_frames={"d1": global_df.copy(), "d2": global_df.copy()},
            valid_dates=list(dates),
            regimes=regimes,
            close_df=(1.0 + ret).cumprod(),
            returns_df=ret,
            factor_returns_df=pd.DataFrame({"SPY": ret["A"]}, index=dates),
            provider_label="test",
        )

        first_map = _resolve_window_domain_map(cfg, prepared, list(dates[:60]), use_learned_clusters=True)
        second_map = _resolve_window_domain_map(cfg, prepared, list(dates[60:]), use_learned_clusters=True)

        def same_cluster(mapping: dict[str, list[str]], left: str, right: str) -> bool:
            for symbols in mapping.values():
                if left in symbols and right in symbols:
                    return True
            return False

        self.assertTrue(same_cluster(first_map, "A", "B"))
        self.assertTrue(same_cluster(second_map, "A", "C"))

    def test_normalization_fits_train_window_only(self) -> None:
        dates = pd.date_range("2022-01-01", periods=80, freq="D")
        values = np.concatenate([np.linspace(-1.0, 1.0, 40), np.linspace(40.0, 60.0, 40)])
        frame = pd.DataFrame({"x": values}, index=dates)
        regimes = pd.Series("sideways", index=dates)
        train_dates = list(dates[:40])
        test_dates = list(dates[40:])

        norm = _normalize_frame_by_window(frame, regimes, train_dates)
        train_mean = float(norm.loc[train_dates, "x"].mean())
        test_mean = float(norm.loc[test_dates, "x"].mean())
        self.assertAlmostEqual(train_mean, 0.0, places=6)
        self.assertGreater(test_mean, 10.0)

    def test_method_registry_contains_baselines_and_ablation(self) -> None:
        cfg = _build_config()
        self.assertEqual(_method_from_name("equal_weight", cfg).family, "baseline")
        self.assertEqual(_method_from_name("flat_sac", cfg).family, "baseline")
        ab = _method_from_name("ablation_no_global_macro", cfg)
        self.assertEqual(ab.family, "hierarchical")
        self.assertFalse(ab.use_global_macro_features)


if __name__ == "__main__":
    unittest.main()
