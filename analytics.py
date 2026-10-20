"""Pure calculation helpers for the dashboard, kept free of Streamlit so they
can be tested directly."""

import numpy as np
import pandas as pd

POS_WORDS = {
    "surge",
    "gain",
    "growth",
    "positive",
    "bullish",
    "profit",
    "upgrade",
    "climb",
    "rise",
    "beat",
    "outperform",
    "strong",
    "optimistic",
}
NEG_WORDS = {
    "drop",
    "loss",
    "crash",
    "negative",
    "bearish",
    "inflation",
    "recession",
    "crisis",
    "fall",
    "slump",
    "down",
    "deficit",
    "warn",
    "debt",
    "risk",
}

TRADING_DAYS = 252


def analyze_sentiment_dict(text):
    """Dictionary-based financial sentiment, used when FinBERT is off or fails."""
    words = text.lower().replace(".", "").replace(",", "").split()
    score = 0.0
    for w in words:
        if w in POS_WORDS:
            score += 0.2
        elif w in NEG_WORDS:
            score -= 0.2
    score = max(min(score, 1.0), -1.0)
    label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    return {"label": label, "score": abs(score) if score != 0 else 0.5}


def clean_prices(raw, tickers):
    """Turn a yfinance download into a tidy price frame, one column per ticker.

    yfinance 0.2.51+ adjusts prices by default and no longer returns an
    'Adj Close' column, so the adjusted series is read from 'Close'.
    """
    data = raw["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.dropna(axis=1, how="all").dropna(how="all")
    return data.ffill().bfill()


def portfolio_stats(returns, risk_free_rate=0.04):
    """Equal-weighted portfolio statistics from a frame of daily returns."""
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    cov_matrix = returns.cov()
    port_mean = float(np.sum(returns.mean() * weights))
    port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    port_std = float(np.sqrt(port_var))
    ann_return = port_mean * TRADING_DAYS
    ann_vol = port_std * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    return {
        "mean": port_mean,
        "var": port_var,
        "std": port_std,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


def simulate_gbm(port_mean, port_var, port_std, horizon, n_sims, initial, rng=None):
    """Geometric Brownian Motion paths, shape (horizon, n_sims)."""
    rng = rng or np.random.default_rng()
    drift = port_mean - 0.5 * port_var
    noise = rng.standard_normal((horizon, n_sims))
    return initial * np.cumprod(np.exp(drift + port_std * noise), axis=0)


def value_at_risk(final_values, initial, percentile=5):
    """Loss at the given percentile of final values, in dollars and percent."""
    threshold = float(np.percentile(final_values, percentile))
    var_dollar = initial - threshold
    return {
        "threshold": threshold,
        "dollar": var_dollar,
        "percent": var_dollar / initial * 100,
    }


def parse_news_item(item):
    """Read a yfinance news item in either the old flat shape or the newer
    one nested under 'content'."""
    content = item.get("content") or {}
    title = item.get("title") or content.get("title") or ""
    publisher = item.get("publisher") or (content.get("provider") or {}).get(
        "displayName", ""
    )
    link = (
        item.get("link")
        or (content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url")
        or "#"
    )
    return {"title": title, "publisher": publisher, "link": link}
