import datetime as dt
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
DEFAULT_CACHE_PATH = Path("data/cache/binance_um_perp_1h_panel.h5")


class BinanceAccessError(RuntimeError):
    """Raised when Binance blocks or rejects the market data request."""


def recent_complete_hour_window(days=365, now=None):
    now = now or dt.datetime.utcnow()
    end_hour = now.replace(minute=0, second=0, microsecond=0)
    end_time = end_hour - dt.timedelta(milliseconds=1)
    start_time = end_hour - dt.timedelta(days=days)
    return start_time, end_time


def _to_millis(value):
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime.combine(value, dt.time())
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return int(value.timestamp() * 1000)


def _request_klines(session, base_url, params, timeout):
    response = session.get(
        f"{base_url}/fapi/v1/continuousKlines",
        params=params,
        timeout=timeout,
    )
    if response.status_code == 451:
        raise BinanceAccessError(
            "Binance Futures API returned HTTP 451 for this location. "
            "Run the downloader from an eligible network or set HTTP_PROXY/"
            "HTTPS_PROXY to a proxy that can access Binance official APIs."
        )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "code" in payload:
        raise BinanceAccessError(f"Binance API error: {payload}")
    return payload


def normalize_klines(symbol, rows):
    if not rows:
        return pd.DataFrame(columns=["symbol", *KLINE_COLUMNS])
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["symbol"] = symbol
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype("Int64")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["close_datetime"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["vwap"] = (df["quote_volume"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
    keep_cols = [
        "datetime",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "vwap",
        "close_datetime",
    ]
    return df[keep_cols]


def download_symbol_klines(
    symbol,
    start_time,
    end_time,
    interval="1h",
    base_url=BINANCE_FAPI_BASE_URL,
    session=None,
    request_sleep=0.15,
    timeout=20,
):
    session = session or requests.Session()
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)
    cursor = start_ms
    rows = []

    while cursor <= end_ms:
        params = {
            "pair": symbol,
            "contractType": "PERPETUAL",
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        batch = _request_klines(session, base_url, params, timeout)
        if not batch:
            break
        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(request_sleep)

    df = normalize_klines(symbol, rows)
    df = df.drop_duplicates(["datetime", "symbol"]).sort_values(["datetime", "symbol"])
    return df


def download_panel_klines(
    symbols=None,
    days=365,
    start_time=None,
    end_time=None,
    output_path=DEFAULT_CACHE_PATH,
    interval="1h",
    base_url=BINANCE_FAPI_BASE_URL,
    request_sleep=0.15,
    save=True,
):
    symbols = symbols or DEFAULT_SYMBOLS
    if start_time is None or end_time is None:
        default_start, default_end = recent_complete_hour_window(days=days)
        start_time = start_time or default_start
        end_time = end_time or default_end

    session = requests.Session()
    frames = [
        download_symbol_klines(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            base_url=base_url,
            session=session,
            request_sleep=request_sleep,
        )
        for symbol in symbols
    ]
    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel.set_index(["datetime", "symbol"]).sort_index()
    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_hdf(output_path, key="data", mode="w")
    return panel


def load_cached_panel(path=DEFAULT_CACHE_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run example/download_binance_panel.py first "
            "from a network that can access Binance Futures APIs."
        )
    return pd.read_hdf(path, key="data")
