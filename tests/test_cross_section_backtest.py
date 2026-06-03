import numpy as np
import pandas as pd

from utils.backtest_tool.cross_section import (
    backtest_cross_section_strategy,
    build_market_neutral_weights,
)


def _sample_panel():
    times = pd.date_range("2025-01-01", periods=8, freq="h")
    symbols = ["A", "B", "C", "D"]
    rows = []
    for s_idx, symbol in enumerate(symbols):
        for t_idx, timestamp in enumerate(times):
            rows.append(
                {
                    "datetime": timestamp,
                    "symbol": symbol,
                    "open": 100 + s_idx * 5 + t_idx * (s_idx + 1),
                    "close": 100 + s_idx * 5 + t_idx * (s_idx + 1),
                }
            )
    return pd.DataFrame(rows).set_index(["datetime", "symbol"]).sort_index()


def test_market_neutral_weights_are_balanced():
    panel = _sample_panel()
    factor = panel["open"]
    weights = build_market_neutral_weights(factor, quantile=0.25)

    np.testing.assert_allclose(weights.sum(axis=1), 0.0)
    np.testing.assert_allclose(weights.abs().sum(axis=1), 1.0)


def test_cross_section_backtest_outputs_metrics():
    panel = _sample_panel()
    factor = panel["open"]
    result = backtest_cross_section_strategy(panel, factor, quantile=0.25)

    assert {"equity", "returns", "weights", "metrics"} <= set(result)
    assert "with_fee" in result["equity"].columns
    assert "sharpe" in result["metrics"].columns
    assert result["equity"]["with_fee"].iloc[-1] > 0
