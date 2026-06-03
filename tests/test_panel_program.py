import numpy as np
import pandas as pd

from QuantGplearn import functions
from QuantGplearn._program import _Program
from QuantGplearn.fitness import _Fitness


def _dummy_metric(y, y_pred, sample_weight):
    return float(np.nanmean(np.nan_to_num(y_pred)))


def test_panel_program_combines_time_series_and_section_functions():
    times = pd.date_range("2025-01-01", periods=30, freq="h")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for time_index, timestamp in enumerate(times):
            rows.append([timestamp, symbol, symbol_index + time_index])
    df = pd.DataFrame(rows, columns=["datetime", "symbol", "feature"]).set_index(["datetime", "symbol"])
    security_codes = pd.Categorical(df.index.get_level_values("symbol")).codes.astype(float)
    time_codes = pd.Categorical(df.index.get_level_values("datetime")).codes.astype(float)
    X = np.column_stack([np.zeros(len(df)), df[["feature"]].values, security_codes, time_codes])

    program = _Program(
        function_dict={"number": [functions._function_map["cs_rank"], functions._function_map["ts_delta"]], "category": []},
        arities={1: [functions._function_map["cs_rank"]], 2: [functions._function_map["ts_delta"]]},
        init_depth=(2, 3),
        init_method="half and half",
        n_features=1,
        const_range=(-1, 1),
        metric=_Fitness(_dummy_metric, greater_is_better=True),
        p_point_replace=0.1,
        parsimony_coefficient=0.0,
        random_state=np.random.RandomState(1),
        data_type="panel",
        n_cat_features=0,
        feature_names=["feature"],
        program=[functions._function_map["cs_rank"], functions._function_map["ts_delta"], "1", 24],
    )
    out = program.execute(X)

    assert out.shape == (len(df),)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    assert finite.min() >= 0
    assert finite.max() <= 1
