import numpy as np
import pandas as pd

from QuantGplearn.gpu_transformer import GpuSymbolicTransformer
from QuantGplearn.tensor_data import TensorPanelData


def _panel():
    rng = np.random.default_rng(7)
    times = pd.date_range("2024-01-01", periods=32, freq="h")
    symbols = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([times, symbols], names=["datetime", "symbol"])
    df = pd.DataFrame(index=idx)
    df["close"] = 100 + rng.normal(size=len(idx)).cumsum()
    df["volume"] = rng.lognormal(mean=0, sigma=1, size=len(idx))
    ret = df.groupby(level="symbol")["close"].pct_change().shift(-1)
    df["target"] = ret.fillna(0.0)
    return df


def test_tensor_panel_data_shape_cpu():
    df = _panel()
    data = TensorPanelData.from_panel_df(
        df,
        feature_names=["close", "volume"],
        target_col="target",
        device="cpu",
    )
    assert data.values.shape == (32, 4, 2)
    assert data.target.shape == (32, 4)
    assert data.mask.shape == (32, 4)


def test_gpu_symbolic_transformer_cpu_smoke():
    df = _panel()
    model = GpuSymbolicTransformer(
        population_size=12,
        generations=2,
        hall_of_fame=6,
        n_components=2,
        tournament_size=3,
        init_depth=(1, 2),
        function_set=["add", "sub", "mul", "div", "cs_rank", "ts_delta", "ts_mean"],
        feature_names=["close", "volume"],
        device="cpu",
        objective="ic",
        stopping_criteria=999,
        random_state=1,
        verbose=0,
    )
    model.fit_panel(df, target_col="target")
    factors = model.transform_panel(output="dataframe")
    expr = model.get_factor_expressions()
    assert factors.shape == (32 * 4, 2)
    assert expr.shape[0] == 2
    assert "expression" in expr.columns
