# QuantGplearn

QuantGplearn is a genetic programming toolkit for mining quantitative trading
factors. It builds on the `gplearn` API style and adds operators, constraints,
and example workflows that are useful when searching for time-series factors in
systematic investment strategies.

The repository includes the core package, a BTCUSDT factor-mining example, a
small backtest utility, sample data, and generated example results.

## Highlights

- `gplearn`-style estimators: `SymbolicRegressor`, `SymbolicClassifier`, and
  `SymbolicTransformer`.
- Finance-oriented time-series operators such as `ts_delta`, `ts_rank`,
  `ts_corr`, `ts_zscore`, `ts_macd`, `ts_atr`, and `ts_hedge`.
- Fixed economic window choices for integer parameters, including 1-day, 3-day,
  7-day, 14-day, 21-day, and 30-day windows on hourly data.
- Custom fitness functions that can score generated factors with domain logic,
  including backtest metrics such as Calmar ratio and Sharpe ratio.
- Formula-length controls and parsimony penalties to limit expression bloat
  during evolution.
- Parallel evolution with `pathos.multiprocessing` for faster population
  evaluation.

## Installation

QuantGplearn currently targets Python 3.11+ on Linux.

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/WYFHHH/QuantGplearn.git
cd QuantGplearn
python -m pip install -e .
```

The package metadata does not yet declare runtime dependencies. Install the
libraries used by the core package and the example workflow before running the
example:

```bash
python -m pip install numpy pandas scipy scikit-learn joblib pathos numba
python -m pip install requests tqdm dill matplotlib seaborn pyyaml tables
```

## Quick Start

Run the end-to-end BTCUSDT example:

```bash
python example/get_factors.py
```

The script:

1. Loads BTCUSDT hourly kline data from `data/`.
2. Downloads Binance perpetual futures data if the local HDF5 file is missing.
3. Builds base features such as momentum, moving averages, and rolling z-scores.
4. Evolves symbolic factors with `SymbolicTransformer`.
5. Saves selected programs under `example/details/factors/`.
6. Runs a simple backtest and writes results under `backtest_results/`.

The included sample run can produce factors such as:

```text
ts_min(ts_atr(zscore_7d, high, mom_3d, 24, 24), 72)

ts_macd(
    ts_hedge(
        ts_bar_bs(ts_zscore(ts_cdlbodym(zscore_7d, zscore_1d, 24), 72),
                  ts_cmo(abs(mom_1d), 72),
                  24),
        ts_kurt(mul(ts_mom(ma_7d, 72), ts_delta(ma_1d, 72)), 72),
        72,
        24),
    24,
    24,
    72)
```

## Basic API

The main user-facing entry points are exposed from `QuantGplearn.genetic`:

```python
from QuantGplearn import fitness, functions
from QuantGplearn.genetic import SymbolicTransformer


def score_func(y, y_pred, sample_weight):
    # Return a higher score for better factors.
    return 0.0


metric = fitness.make_fitness(
    function=score_func,
    greater_is_better=True,
    wrap=False,
)

model = SymbolicTransformer(
    population_size=100,
    hall_of_fame=20,
    n_components=5,
    generations=5,
    tournament_size=20,
    function_set=functions.all_function,
    metric=metric,
    const_range=(-1, 1),
    feature_names=["open", "high", "low", "close", "vwap"],
    n_jobs=4,
    random_state=2025,
)

model.fit(X, y, max_length=20)
factors = model.transform(X)
```

For a domain-specific fitness function that needs market data, bind the extra
inputs with `functools.partial`, as shown in `example/get_factors.py`.

## Function Set

`functions.all_function` combines protected arithmetic primitives with
finance-oriented time-series operators.

Core primitives:

```text
add, sub, mul, div, sqrt, log, abs, neg, inv, max, min, sig
```

Time-series operators:

```text
ts_shift, ts_delta, ts_mom, ts_min, ts_max, ts_argmax, ts_argmin,
ts_rank, ts_sum, ts_std, ts_corr, ts_mean, ts_zscore, ts_cdlbodym,
ts_bar_bs, ts_adx, ts_aroon, ts_bopr, ts_cmo, ts_macd, ts_rsi,
ts_stochf, ts_xs_ratio, ts_one_ols_k, ts_one_ols_resid, ts_skew,
ts_kurt, ts_atr, ts_hedge
```

The default time windows are configured for hourly bars:

```text
24, 72, 168, 336, 504, 720
```

## Project Layout

```text
QuantGplearn/
  genetic.py       Genetic programming estimators and evolution loop
  functions.py     Protected primitives and quantitative time-series operators
  fitness.py       Fitness factory and built-in metrics
  _program.py      Symbolic program representation
example/
  get_factors.py   End-to-end factor mining and backtest example
utils/
  backtest_tool/   PnL and performance helpers used by the example
config/
  config.yaml      Backtest output path configuration
data/              Sample HDF5 market data
backtest_results/  Example output artifacts
```

## Notes

- The example is intentionally small: `population_size=10` and `generations=2`
  keep runtime manageable for demonstration.
- Increase the population, generations, and tournament size for research runs.
- `fit` requires a `max_length` argument; programs above this length are kept
  away from crossover, subtree mutation, point mutation, and point replacement.
- `const_range=None` disables scalar constants. Time-series operators with
  integer window parameters use the fixed window list defined in
  `QuantGplearn/functions.py`.

## Acknowledgements

QuantGplearn is inspired by and adapted from:

- [`gplearn`](https://github.com/trevorstephens/gplearn), which provides the
  foundational symbolic genetic programming design.
- [`gplearnplus`](https://github.com/ACEACEjasonhuang/gplearnplus), which
  served as a reference for extending genetic programming functionality.

## License

This project is released under the MIT License. See `LICENSE` for details.
