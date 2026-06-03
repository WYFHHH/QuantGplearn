import pytest

from utils.data.binance_futures import BinanceAccessError, normalize_klines


def test_normalize_klines_builds_panel_fields():
    rows = [[
        1735689600000,
        "100",
        "110",
        "90",
        "105",
        "2",
        1735693199999,
        "210",
        12,
        "1",
        "105",
        "0",
    ]]
    df = normalize_klines("BTCUSDT", rows)

    assert list(df["symbol"]) == ["BTCUSDT"]
    assert df.loc[0, "vwap"] == 105
    assert str(df.loc[0, "datetime"]) == "2025-01-01 00:00:00"


def test_binance_access_error_message_mentions_proxy():
    with pytest.raises(BinanceAccessError) as exc_info:
        raise BinanceAccessError("set HTTP_PROXY/HTTPS_PROXY")
    assert "HTTP_PROXY" in str(exc_info.value)
