from pathlib import Path

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 24 * 365


def _to_wide_series(values, name=None):
    if isinstance(values, pd.DataFrame):
        if not isinstance(values.index, pd.MultiIndex):
            return values.sort_index()
        if name is None:
            if values.shape[1] != 1:
                raise ValueError("DataFrame input requires a column name")
            name = values.columns[0]
        values = values[name]
    if not isinstance(values.index, pd.MultiIndex):
        raise ValueError("values must have a MultiIndex of datetime and symbol")
    return values.unstack("symbol").sort_index()


def build_market_neutral_weights(
    factor,
    quantile=0.3,
    long_total=0.5,
    short_total=-0.5,
):
    factor_wide = _to_wide_series(factor)
    valid_count = factor_wide.notna().sum(axis=1)
    leg_size = np.floor(valid_count * quantile).astype(int).clip(lower=1)
    leg_size = pd.concat([leg_size, (valid_count // 2).astype(int)], axis=1).min(axis=1)
    tradable = leg_size >= 1

    rank_asc = factor_wide.rank(axis=1, method="first", ascending=True)
    rank_desc = factor_wide.rank(axis=1, method="first", ascending=False)
    long_mask = rank_desc.le(leg_size, axis=0) & tradable.to_numpy()[:, None]
    short_mask = rank_asc.le(leg_size, axis=0) & tradable.to_numpy()[:, None]

    weights = pd.DataFrame(0.0, index=factor_wide.index, columns=factor_wide.columns)
    weights = weights.mask(long_mask, long_total).mask(short_mask, short_total)
    weights = weights.div(leg_size.replace(0, np.nan), axis=0).fillna(0.0)
    return weights


def _drawdown(equity):
    peak = equity.cummax()
    return equity / peak - 1.0


def compute_performance_metrics(strategy_returns, equity, turnover=None):
    returns = strategy_returns.dropna()
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "avg_turnover": 0.0,
        }
    years = max(len(returns) / HOURS_PER_YEAR, 1 / HOURS_PER_YEAR)
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else total_return / years
    vol = returns.std(ddof=1)
    sharpe = float(returns.mean() / vol * np.sqrt(HOURS_PER_YEAR)) if vol and np.isfinite(vol) else 0.0
    max_drawdown = float(_drawdown(equity).min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    avg_turnover = float(turnover.mean()) if turnover is not None and len(turnover) else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "avg_turnover": avg_turnover,
    }


def backtest_cross_section_strategy(
    panel_data,
    factor,
    price_col="open",
    fee_rate=3 / 10000,
    quantile=0.3,
):
    if not isinstance(panel_data.index, pd.MultiIndex):
        raise ValueError("panel_data must be indexed by datetime and symbol")
    price = panel_data[price_col].unstack("symbol").sort_index()
    factor_wide = _to_wide_series(factor).reindex_like(price)
    factor_series = factor_wide.stack()
    weights = build_market_neutral_weights(factor_series, quantile=quantile)
    weights = weights.reindex_like(price).fillna(0.0)

    forward_returns = price.shift(-2) / price.shift(-1) - 1.0
    gross_returns = (weights * forward_returns).sum(axis=1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    fee = turnover * fee_rate
    net_returns = gross_returns - fee
    benchmark_returns = forward_returns.mean(axis=1).fillna(0.0)

    equity = pd.DataFrame(
        {
            "with_fee": (1.0 + net_returns).cumprod(),
            "without_fee": (1.0 + gross_returns).cumprod(),
            "equal_weight": (1.0 + benchmark_returns).cumprod(),
        }
    )
    returns = pd.DataFrame(
        {
            "gross": gross_returns,
            "net": net_returns,
            "fee": fee,
            "benchmark": benchmark_returns,
            "turnover": turnover,
        }
    )
    metrics = compute_performance_metrics(net_returns, equity["with_fee"], turnover)
    metrics_without_fee = compute_performance_metrics(gross_returns, equity["without_fee"], turnover)
    metrics_table = pd.DataFrame(
        [metrics, metrics_without_fee],
        index=["with_fee", "without_fee"],
    )
    return {
        "equity": equity,
        "returns": returns,
        "weights": weights,
        "metrics": metrics_table,
    }


def plot_cross_section_performance(result, output_path, title="Cross-Section Strategy"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    equity = result["equity"]
    returns = result["returns"]
    metrics = result["metrics"]
    drawdown = _drawdown(equity["with_fee"])
    rolling_sharpe = (
        returns["net"].rolling(24 * 30).mean() /
        returns["net"].rolling(24 * 30).std(ddof=1) *
        np.sqrt(HOURS_PER_YEAR)
    )
    monthly = returns["net"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    heatmap = monthly.to_frame("return")
    heatmap["year"] = heatmap.index.year
    heatmap["month"] = heatmap.index.month
    heatmap = heatmap.pivot(index="year", columns="month", values="return")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(title, fontsize=18, fontweight="bold")

    equity.plot(ax=axes[0, 0], linewidth=1.8)
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].grid(alpha=0.25)

    drawdown.plot(ax=axes[0, 1], color="#c0392b")
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color="#c0392b", alpha=0.2)
    axes[0, 1].set_title("Drawdown")
    axes[0, 1].grid(alpha=0.25)

    rolling_sharpe.plot(ax=axes[1, 0], color="#2c7fb8")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("30D Rolling Sharpe")
    axes[1, 0].grid(alpha=0.25)

    returns["turnover"].rolling(24).mean().plot(ax=axes[1, 1], color="#7b3294")
    axes[1, 1].set_title("24H Average Turnover")
    axes[1, 1].grid(alpha=0.25)

    sns.heatmap(heatmap, ax=axes[2, 0], cmap="RdYlGn", center=0, annot=True, fmt=".1%")
    axes[2, 0].set_title("Monthly Net Returns")

    axes[2, 1].axis("off")
    rows = []
    for name, row in metrics.iterrows():
        rows.append(
            f"{name}\n"
            f"Total: {row['total_return']:.2%}\n"
            f"Annual: {row['annual_return']:.2%}\n"
            f"Sharpe: {row['sharpe']:.2f}\n"
            f"Max DD: {row['max_drawdown']:.2%}\n"
            f"Calmar: {row['calmar']:.2f}\n"
            f"Avg Turnover: {row['avg_turnover']:.2f}"
        )
    axes[2, 1].text(0.02, 0.95, "\n\n".join(rows), va="top", ha="left", fontsize=12, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
