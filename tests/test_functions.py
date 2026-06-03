import numpy as np
import pandas as pd

from QuantGplearn import functions


def test_time_series_operator_fixes():
    x = np.arange(30, dtype=float)

    shifted = functions._function_map["ts_shift"](x, 24)
    assert np.isnan(shifted[:24]).all()
    assert np.allclose(shifted[24:], x[:-24])

    assert functions._function_map["ts_argmin"].name == "ts_argmin"

    kurt = functions._function_map["ts_kurt"](x, 24)
    expected = pd.Series(x).rolling(24).kurt().values
    np.testing.assert_allclose(kurt, expected, equal_nan=True)


def test_section_operators_grouped_by_time():
    labels = np.array([1, 1, 1, 2, 2, 2])
    values = np.array([3.0, 1.0, 2.0, 30.0, 10.0, 20.0])

    ranked = functions._groupby(labels, functions._function_map["cs_rank"], values)
    np.testing.assert_allclose(ranked, [1.0, 1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3])

    demeaned = functions._groupby(labels, functions._function_map["cs_demean"], values)
    np.testing.assert_allclose(demeaned[:3], [1.0, -1.0, 0.0])
    np.testing.assert_allclose(demeaned[3:], [10.0, -10.0, 0.0])
