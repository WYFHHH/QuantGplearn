import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.binance_futures import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SYMBOLS,
    BinanceAccessError,
    download_panel_klines,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Binance USD-M perpetual 1h panel data.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        panel = download_panel_klines(
            symbols=args.symbols,
            days=args.days,
            output_path=args.output,
            request_sleep=args.sleep,
        )
    except BinanceAccessError as exc:
        print(exc)
        return 1
    print(f"Saved {panel.shape[0]} rows x {panel.shape[1]} columns to {args.output}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Datetime range: {panel.index.get_level_values('datetime').min()} -> "
          f"{panel.index.get_level_values('datetime').max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
