"""Run the whole Streamlit script headlessly with yfinance stubbed out, so the
app is exercised end to end without network access."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance as yf
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def fake_market(monkeypatch):
    def download(tickers, start=None, end=None, **kwargs):
        idx = pd.date_range(start, periods=60, freq="B")
        cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(
            0.01 * rng.standard_normal((60, len(cols))).cumsum(axis=0)
        )
        return pd.DataFrame(prices, index=idx, columns=cols)

    class FakeTicker:
        def __init__(self, symbol):
            self.news = [
                {
                    "content": {
                        "title": f"{symbol} shares surge after strong results",
                        "provider": {"displayName": "Wire"},
                        "canonicalUrl": {"url": "https://example.test/a"},
                    }
                }
            ]

    monkeypatch.setattr(yf, "download", download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)


def test_app_renders_metrics_and_news(fake_market):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    page = " ".join(m.value for m in at.markdown)
    assert "Annualized Return" in page
    assert "shares surge after strong results" in page
    assert "POSITIVE" in page


def test_monte_carlo_button_produces_var(fake_market):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Value at Risk" in page
