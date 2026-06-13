# Optimized Time-Series Operators Integration

This document summarizes the integration of the uploaded optimized `functions.py` operator set into the GPU/Torch QuantGplearn framework.

## Main correctness fixes

The uploaded operator file was useful as a reference for the intended time-series operator coverage, but several implementation details were corrected before integration:

1. `ts_shift` in the uploaded file assigned `res[d:] = res[:-d]`, which propagates NaN rather than shifting the input. The integrated implementation uses `res[d:] = x[:-d]`.
2. `rolling_argmax`, `rolling_argmin`, `rolling_argsort`, and `rolling_freq` used `int32` result arrays while assigning `np.nan`. The integrated implementation returns `float64`, so missing warm-up windows are represented correctly.
3. `ts_rank` in the uploaded file returned the position of the maximum value in the rolling window, not the percentile rank of the latest value. The integrated implementation computes the latest value's rolling percentile rank with average tie handling on CPU.
4. `ts_cdlbodym`, `ts_bar_bs`, `ts_bopr`, and similar ratio operators now use protected division to avoid `inf`/`-inf` values in GP trees.
5. `ts_one_ols_k` now uses the closed-form rolling-sum formula instead of per-window polynomial fitting.
6. `ts_one_ols_resid` now returns the current-point residual from rolling OLS rather than a rolling-sum residual.
7. `ts_kurt` remains an actual rolling kurtosis operator; the uploaded implementation called rolling skew by mistake.
8. `ts_hedge` retains the regression-residual hedge form because the uploaded sorted top/bottom variant treats `d2` as a ratio, while the GP metadata supplies `d2` as an integer window from `para_list`.

## CPU backend changes

`QuantGplearn/functions.py` now includes optional Numba-accelerated helpers for:

- `ts_argmax`
- `ts_argmin`
- `ts_rank`
- `ts_freq`

The code falls back to pure NumPy/Pandas implementations if Numba is unavailable.

The following CPU time-series operators were updated or added to the default GP function set:

- `ts_freq`
- `ts_ema`
- `ts_bband`
- optimized `ts_one_ols_k`
- corrected `ts_one_ols_resid`
- corrected `ts_xs_ratio`
- rolling-MA variant of `ts_macd`
- range-expansion variant of `ts_atr`

## GPU backend changes

`QuantGplearn/torch_functions.py` now has Torch implementations for the expanded time-series operator set:

- `ts_shift`, `ts_delta`, `ts_mom`
- `ts_min`, `ts_max`, `ts_argmax`, `ts_argmin`, `ts_rank`, `ts_freq`
- `ts_sum`, `ts_mean`, `ts_std`, `ts_corr`, `ts_zscore`
- `ts_cdlbodym`, `ts_bar_bs`, `ts_aroon`, `ts_adx`, `ts_bopr`
- `ts_cmo`, `ts_ema`, `ts_macd`, `ts_rsi`, `ts_stochf`
- `ts_xs_ratio`, `ts_one_ols_k`, `ts_one_ols_resid`
- `ts_skew`, `ts_kurt`, `ts_atr`, `ts_hedge`, `ts_bband`

These functions are registered automatically when `QuantGplearn.torch_functions` is imported.

## Notes on semantic compatibility

The GPU backend is optimized for dense `[T, N]` panel tensors. It prioritizes numerical stability and GPU throughput. For a small number of operators, especially ranking and exponential smoothing, results may differ slightly from Pandas in tie or NaN edge cases. The intended training target is robust factor discovery rather than exact byte-for-byte Pandas reproduction.
