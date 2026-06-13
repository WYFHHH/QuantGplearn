# QuantGplearn

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey)](https://github.com/WYFHHH/QuantGplearn)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GP](https://img.shields.io/badge/genetic%20programming-factor%20mining-purple)](QuantGplearn/genetic.py)

QuantGplearn is a genetic-programming framework for quantitative factor
research. It provides a legacy NumPy/Pandas execution path and a Torch backend
for GPU-accelerated panel factor evaluation.

```mermaid
flowchart LR
    A[Panel Features] --> B[GP Expression Trees]
    B --> C[NumPy/Pandas or Torch Execution]
    C --> D[IC, RankIC, ICIR, or Sharpe Objective]
    D --> E[Factor Selection and Correlation Filtering]
```

## Highlights

- `gplearn`-style estimators: `SymbolicRegressor`, `SymbolicClassifier`, and
  `SymbolicTransformer`.
- `GpuSymbolicTransformer` for Torch-based panel factor mining.
- Finance-oriented rolling time-series and cross-section operators.
- Dense panel representation with shape `[time, security, feature]`.
- GPU objectives including IC, RankIC, ICIR, RankICIR, and long-short Sharpe.
- Expression caching, daily normalization, and factor-correlation filtering.

## Installation

QuantGplearn targets Python 3.11+ on Linux.

```bash
git clone https://github.com/WYFHHH/QuantGplearn.git
cd QuantGplearn
python -m pip install -e .
```

Runtime dependencies are declared in `setup.py`. GPU execution requires a
Torch build compatible with the installed CUDA runtime.

For development tests:

```bash
python -m pip install pytest
pytest -q
```

## CPU API

The original symbolic estimators remain available:

```python
from QuantGplearn.genetic import SymbolicTransformer

model = SymbolicTransformer(
    population_size=1000,
    generations=20,
    hall_of_fame=100,
    n_components=20,
    function_set=["add", "sub", "mul", "div", "ts_delta", "ts_mean"],
    random_state=2025,
)
```

The CPU path supports raw, time-series, cross-section, and mixed panel
expressions.

## GPU/Torch Panel API

Input data should be a `pandas.DataFrame` indexed by `datetime` and `symbol`,
with feature columns and a target column:

```python
from QuantGplearn.gpu_transformer import GpuSymbolicTransformer
from QuantGplearn.torch_functions import GPU_SAFE_PANEL_FUNCTIONS

feature_names = ["open", "high", "low", "close", "volume"]

model = GpuSymbolicTransformer(
    population_size=512,
    generations=20,
    hall_of_fame=100,
    n_components=20,
    tournament_size=64,
    function_set=GPU_SAFE_PANEL_FUNCTIONS,
    objective="icir",
    feature_names=feature_names,
    time_series_index="datetime",
    security_index="symbol",
    device="cuda:0",
    random_state=2025,
    verbose=1,
)

model.fit_panel(panel_df, target_col="target")
factor_df = model.transform_panel(output="dataframe")
expression_df = model.get_factor_expressions()
```

Internally the panel is converted to:

```text
values: [T, N, F]
target: [T, N]
mask:   [T, N]
```

When CUDA is unavailable, the GPU transformer can run on CPU tensors for
functional validation.

## Function Sets

Core primitives:

```text
add, sub, mul, div, sqrt, log, abs, neg, inv, max, min, sin, cos, tan, sig
```

Time-series operators include:

```text
ts_shift, ts_delta, ts_mom, ts_min, ts_max, ts_argmax, ts_argmin,
ts_rank, ts_sum, ts_std, ts_corr, ts_mean, ts_zscore, ts_freq,
ts_cdlbodym, ts_bar_bs, ts_adx, ts_aroon, ts_bopr, ts_cmo, ts_ema,
ts_macd, ts_rsi, ts_stochf, ts_xs_ratio, ts_one_ols_k,
ts_one_ols_resid, ts_skew, ts_kurt, ts_atr, ts_hedge, ts_bband
```

Cross-section operators:

```text
cs_rank, cs_zscore, cs_demean, cs_scale, cs_winsorize
```

Use `functions.all_function` for legacy raw and time-series workflows,
`functions.section_function` for cross-section expressions, and
`functions.panel_function` for expressions that mix both dimensions.

## Public Repository Scope

This public repository contains the reusable framework:

```text
QuantGplearn/
  genetic.py
  _program.py
  functions.py
  gpu_transformer.py
  tensor_data.py
  torch_functions.py
  tensor_fitness.py
  evaluator.py
  alpha_pool.py
docs/
tests/
```

Private research data, strategy implementations, training launchers, mined
expressions, logs, and backtest outputs are intentionally excluded from the
public repository.

## Documentation

- `docs/GPU_FACTOR_MINING.md`: GPU architecture and usage.
- `docs/OPTIMIZED_OPERATORS.md`: optimized CPU and Torch operators.
- `docs/REFACTOR_SUMMARY.md`: CPU/GPU compatibility and refactor overview.

## Acknowledgements

QuantGplearn is inspired by and adapted from:

- [`gplearn`](https://github.com/trevorstephens/gplearn)
- [`gplearnplus`](https://github.com/ACEACEjasonhuang/gplearnplus)

## License

This project is released under the MIT License. See `LICENSE` for details.
