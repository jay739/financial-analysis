import numpy as np
import pandas as pd
import pytest

from analytics import (
    analyze_sentiment_dict,
    clean_prices,
    parse_news_item,
    portfolio_stats,
    simulate_gbm,
    value_at_risk,
)


def _download(tickers, days=30):
    """Frame shaped like a current yfinance multi-ticker download."""
    idx = pd.date_range("2020-02-03", periods=days, freq="B")
    cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
    rng = np.random.default_rng(0)
    data = 100 + rng.standard_normal((days, len(cols))).cumsum(axis=0)
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.mark.parametrize(
    "text,label",
    [
        ("Stocks surge on strong earnings beat", "POSITIVE"),
        ("Markets crash as recession fears grow", "NEGATIVE"),
        ("Company holds annual meeting", "NEUTRAL"),
    ],
)
def test_sentiment_labels(text, label):
    assert analyze_sentiment_dict(text)["label"] == label


def test_sentiment_score_is_clamped():
    res = analyze_sentiment_dict(" ".join(["surge"] * 20))
    assert res["score"] == pytest.approx(1.0)
    assert analyze_sentiment_dict("nothing here")["score"] == 0.5


def test_clean_prices_reads_close_without_adj_close():
    raw = _download(["SPY", "AAPL"])
    assert "Adj Close" not in raw.columns.get_level_values(0)
    prices = clean_prices(raw, ["SPY", "AAPL"])
    assert list(prices.columns) == ["SPY", "AAPL"]
    assert not prices.isna().any().any()


def test_clean_prices_drops_tickers_with_no_data():
    raw = _download(["SPY", "BOGUS"])
    raw[("Close", "BOGUS")] = np.nan
    assert list(clean_prices(raw, ["SPY", "BOGUS"]).columns) == ["SPY"]


def test_clean_prices_single_ticker_series():
    idx = pd.date_range("2020-02-03", periods=5, freq="B")
    raw = pd.DataFrame({"Close": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=idx)
    prices = clean_prices(raw, ["SPY"])
    assert list(prices.columns) == ["SPY"]
    # A day with no price at all is dropped rather than invented.
    assert prices["SPY"].tolist() == [1.0, 2.0, 4.0, 5.0]


def test_clean_prices_fills_gaps_between_tickers():
    raw = _download(["SPY", "AAPL"], days=5)
    raw.iloc[2, raw.columns.get_loc(("Close", "AAPL"))] = np.nan
    prices = clean_prices(raw, ["SPY", "AAPL"])
    assert len(prices) == 5
    assert prices["AAPL"].iloc[2] == prices["AAPL"].iloc[1]  # forward-filled


def test_portfolio_stats_equal_weights_match_manual():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.02]})
    stats = portfolio_stats(returns, risk_free_rate=0.0)
    w = np.array([0.5, 0.5])
    assert stats["mean"] == pytest.approx(float(returns.mean() @ w))
    assert stats["var"] == pytest.approx(float(w @ returns.cov().values @ w))
    assert stats["ann_return"] == pytest.approx(stats["mean"] * 252)
    assert stats["sharpe"] == pytest.approx(stats["ann_return"] / stats["ann_vol"])


def test_portfolio_stats_zero_volatility_sharpe():
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0]})
    assert portfolio_stats(returns)["sharpe"] == 0.0


def test_simulate_gbm_shape_and_start():
    paths = simulate_gbm(
        0.0005, 0.0001, 0.01, 50, 200, 10_000, np.random.default_rng(1)
    )
    assert paths.shape == (50, 200)
    assert (paths > 0).all()
    # One step from 10,000 with 1% daily vol stays close to the start.
    assert np.abs(paths[0] / 10_000 - 1).max() < 0.1


def test_simulate_gbm_zero_vol_is_deterministic():
    paths = simulate_gbm(0.001, 0.0, 0.0, 10, 3, 100)
    expected = 100 * np.exp(0.001 * np.arange(1, 11))
    assert np.allclose(paths[:, 0], expected)


def test_value_at_risk():
    final = np.linspace(9_000, 11_000, 101)
    var = value_at_risk(final, 10_000, percentile=5)
    assert var["threshold"] == pytest.approx(9_100)
    assert var["dollar"] == pytest.approx(900)
    assert var["percent"] == pytest.approx(9.0)


def test_parse_news_item_old_flat_shape():
    item = {"title": "T", "publisher": "Reuters", "link": "https://x"}
    assert parse_news_item(item) == {
        "title": "T",
        "publisher": "Reuters",
        "link": "https://x",
    }


def test_parse_news_item_nested_content_shape():
    item = {
        "id": "1",
        "content": {
            "title": "Nested",
            "provider": {"displayName": "Yahoo Finance"},
            "canonicalUrl": {"url": "https://y"},
        },
    }
    assert parse_news_item(item) == {
        "title": "Nested",
        "publisher": "Yahoo Finance",
        "link": "https://y",
    }


def test_parse_news_item_missing_fields():
    assert parse_news_item({}) == {"title": "", "publisher": "", "link": "#"}
