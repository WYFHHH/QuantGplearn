import argparse
from functools import partial
import os
from pathlib import Path
import sys

import dill
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QuantGplearn import fitness, functions
from QuantGplearn.genetic import SymbolicTransformer
from utils.backtest_tool.cross_section import (
    backtest_cross_section_strategy,
    plot_cross_section_performance,
)
from utils.data.binance_futures import DEFAULT_CACHE_PATH, load_cached_panel


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "quote_volume",
    "mom_1d",
    "mom_3d",
    "mom_7d",
    "ma_ratio_1d",
    "ma_ratio_3d",
    "zscore_1d",
    "zscore_3d",
    "volume_zscore_1d",
]


def build_panel_features(panel):
    panel = panel.sort_index().copy()
    by_symbol = panel.groupby(level="symbol", group_keys=False)
    panel["mom_1d"] = by_symbol["vwap"].pct_change(24)
    panel["mom_3d"] = by_symbol["vwap"].pct_change(24 * 3)
    panel["mom_7d"] = by_symbol["vwap"].pct_change(24 * 7)
    panel["ma_ratio_1d"] = panel["vwap"] / by_symbol["vwap"].transform(lambda s: s.rolling(24).mean()) - 1
    panel["ma_ratio_3d"] = panel["vwap"] / by_symbol["vwap"].transform(lambda s: s.rolling(24 * 3).mean()) - 1
    panel["zscore_1d"] = by_symbol["vwap"].transform(
        lambda s: (s - s.rolling(24).mean()) / s.rolling(24).std(ddof=1)
    )
    panel["zscore_3d"] = by_symbol["vwap"].transform(
        lambda s: (s - s.rolling(24 * 3).mean()) / s.rolling(24 * 3).std(ddof=1)
    )
    panel["volume_zscore_1d"] = by_symbol["volume"].transform(
        lambda s: (s - s.rolling(24).mean()) / s.rolling(24).std(ddof=1)
    )
    panel["target"] = by_symbol["open"].shift(-2) / by_symbol["open"].shift(-1) - 1
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.dropna(subset=FEATURE_COLUMNS + ["target"])


def score_cross_section_factor(y, y_pred, sample_weight, panel_data):
    if len(y_pred) != len(panel_data):
        return 0.0
    try:
        factor = pd.Series(y_pred, index=panel_data.index, name="gp_factor")
        result = backtest_cross_section_strategy(panel_data, factor)
        metrics = result["metrics"].loc["with_fee"]
        if result["weights"].abs().sum(axis=1).mean() <= 0:
            return 0.0
        score = (metrics["sharpe"] + metrics["calmar"]) / 2
        return float(score) if np.isfinite(score) else 0.0
    except Exception as exc:
        print(exc)
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Mine a market-neutral cross-section strategy with GP.")
    parser.add_argument("--data", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--output-dir", default="backtest_results/cross_section")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        raw_panel = load_cached_panel(args.data)
    except FileNotFoundError as exc:
        print(exc)
        print("Download data first:")
        print("  python example/download_binance_panel.py")
        return 0

    panel = build_panel_features(raw_panel)
    datetimes = panel.index.get_level_values("datetime")
    split_time = datetimes.min() + (datetimes.max() - datetimes.min()) * 0.7
    train_panel = panel.loc[datetimes <= split_time]
    full_panel = panel.copy()

    metric = fitness.make_fitness(
        function=partial(score_cross_section_factor, panel_data=train_panel),
        greater_is_better=True,
        wrap=False,
    )
    model = SymbolicTransformer(
        population_size=args.population_size,
        hall_of_fame=min(args.population_size, max(args.components * 2, args.components)),
        n_components=args.components,
        generations=args.generations,
        tournament_size=min(args.population_size, 20),
        function_set=functions.panel_function,
        metric=metric,
        init_depth=(2, 4),
        const_range=(-1, 1),
        p_crossover=0.2,
        p_subtree_mutation=0.2,
        p_hoist_mutation=0.2,
        p_point_mutation=0.2,
        p_point_replace=0.2,
        tolerable_corr=0.7,
        data_type="panel",
        time_series_index="datetime",
        security_index="symbol",
        feature_names=FEATURE_COLUMNS,
        n_jobs=1,
        verbose=2,
        random_state=2025,
        first_train=True,
    )
    model.fit(train_panel[FEATURE_COLUMNS], train_panel["target"], max_length=args.max_length)
    factor_values = pd.DataFrame(
        model.transform(full_panel[FEATURE_COLUMNS]),
        index=full_panel.index,
        columns=[f"factor_{i}" for i in range(len(model))],
    )
    selected_factor = factor_values.iloc[:, 0].rename("gp_factor")
    result = backtest_cross_section_strategy(full_panel, selected_factor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_values.to_hdf(output_dir / "cross_section_gp_factors.h5", key="data", mode="w")
    result["equity"].to_hdf(output_dir / "cross_section_gp_equity.h5", key="data", mode="w")
    result["metrics"].to_csv(output_dir / "cross_section_gp_metrics.csv")
    with open(output_dir / "cross_section_gp_model.pickle", "wb") as file:
        dill.dump(model, file)
    plot_cross_section_performance(
        result,
        output_dir / "cross_section_gp_performance.png",
        title="QuantGplearn Cross-Section GP Strategy",
    )

    print(model)
    print(result["metrics"].to_string())
    print(f"Artifacts saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
