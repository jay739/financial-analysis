# Financial Crisis Analysis & Portfolio Risk Tool

This repository contains an interactive financial dashboard built using **Streamlit** and **Plotly**. It integrates real-time stock market data loading, **Value at Risk (VaR)** calculations, **Monte Carlo portfolio simulations**, and **financial news sentiment analysis** using Hugging Face's FinBERT model.

## Features

1. **Crisis Context Backtesting**: Allows users to select and analyze portfolio performance during major historical crisis windows, such as the 2008 Great Financial Crisis (GFC) and the 2020 COVID-19 stock market crash.
2. **Portfolio Ratios**: Calculates key risk-adjusted metrics like Annualized Returns, Annualized Volatility, and the Sharpe Ratio during volatile regimes.
3. **Monte Carlo Simulations**: Runs geometric Brownian motion (GBM) pathway simulations over user-defined horizons to forecast portfolio values.
4. **Value at Risk (VaR)**: Determines the 95% confidence Value at Risk in both absolute dollar and percentage terms, demonstrating expected tail losses.
5. **Sentiment Analysis Layer**: Fetches current news articles for tickers using the Yahoo Finance API and performs sentiment scoring (Positive, Neutral, Negative) utilizing a local analyzer or downloading the pre-trained **FinBERT** transformer model.

---

## Directory Structure

```text
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation
```

---

## Installation & Setup

### 1. Requirements
Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the local Streamlit development server:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Technical Details

### Monte Carlo Simulation (GBM)
The price paths are simulated using the formula:
$$S_t = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma Z \sqrt{dt} \right)$$
Where:
- $S_t$ is the portfolio value at day $t$
- $\mu$ is the portfolio mean return (drift)
- $\sigma$ is the portfolio standard deviation (volatility)
- $Z$ is a random standard normal variable
- $dt$ is the time step ($1$ trading day)
