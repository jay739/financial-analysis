import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from analytics import (
    analyze_sentiment_dict,
    clean_prices,
    parse_news_item,
    portfolio_stats,
    simulate_gbm,
    value_at_risk,
)

# Page config
st.set_page_config(
    page_title="Financial Crisis Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style for Rich Aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3139;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Try importing FinBERT model (optional fallback)
@st.cache_resource
def load_finbert():
    try:
        from transformers import pipeline
        # Use a lightweight pipeline or cached model
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return classifier
    except Exception:
        return None

# App Title
st.title("📊 Financial Crisis Analysis & Portfolio Risk Tool")
st.markdown("""
This interactive tool allows you to perform **Monte Carlo portfolio simulations**, analyze **Value at Risk (VaR)**, and review **news sentiment analysis** across S&P 500 sectors and historical crisis periods.
""")

# Sidebar settings
st.sidebar.header("📁 Portfolio Settings")
ticker_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "SPY, AAPL, MSFT, JPM, XOM")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Simulation Settings
st.sidebar.header("⚙️ Monte Carlo Settings")
num_simulations = st.sidebar.slider("Number of Simulations:", 100, 1000, 500)
time_horizon = st.sidebar.slider("Time Horizon (Trading Days):", 30, 252, 126)
initial_investment = st.sidebar.number_input("Initial Investment ($):", min_value=1000, value=10000)

# Historical Crisis Period Selector
st.sidebar.header("⏳ Historical Crisis Context")
crisis_period = st.sidebar.selectbox(
    "Select Crisis Period for Backtest:",
    ["Great Financial Crisis (2007-2009)", "COVID-19 Sell-off (2020)", "Custom Range"]
)

if crisis_period == "Great Financial Crisis (2007-2009)":
    start_date = datetime(2007, 10, 1)
    end_date = datetime(2009, 3, 31)
elif crisis_period == "COVID-19 Sell-off (2020)":
    start_date = datetime(2020, 2, 1)
    end_date = datetime(2020, 5, 1)
else:
    start_date = st.sidebar.date_input("Start Date:", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date:", datetime.now())

# Load FinBERT toggle
use_finbert = st.sidebar.checkbox("Use FinBERT Transformer Model (Downloads ~400MB)", value=False)

# Fetch Market Data
@st.cache_data(show_spinner="Fetching historical stock data...")
def get_stock_data(tickers, start, end):
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        return clean_prices(raw, tickers)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load Data
stock_data = get_stock_data(tickers, start_date, end_date)

if stock_data is not None and not stock_data.empty:
    st.header(f"📈 Crisis Performance: {crisis_period}")
    st.markdown(f"Historical date range analyzed: **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**.")

    # Normalized returns plot
    normalized_data = stock_data / stock_data.iloc[0] * 100
    fig_historical = go.Figure()
    for col in normalized_data.columns:
        fig_historical.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], mode='lines', name=col))
    fig_historical.update_layout(
        title="Asset Performance during Crisis (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price ($)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_historical, use_container_width=True)

    # Core Calculations
    # Equal weighted across the tickers that actually returned data, so an
    # invalid ticker no longer breaks the weights/covariance shapes.
    returns = stock_data.pct_change().dropna()
    stats = portfolio_stats(returns)
    port_mean, port_var, port_std = stats["mean"], stats["var"], stats["std"]
    ann_return, ann_vol, sharpe_ratio = stats["ann_return"], stats["ann_vol"], stats["sharpe"]

    st.markdown("### 📊 Calculated Portfolio Ratios (Crisis Period)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Annualized Return</h4><h2>{ann_return*100:.2f}%</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Annualized Volatility</h4><h2>{ann_vol*100:.2f}%</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Sharpe Ratio (R_f = 4%)</h4><h2>{sharpe_ratio:.2f}</h2></div>", unsafe_allow_html=True)

    # Monte Carlo Simulation
    st.header("🎲 Monte Carlo Portfolio Simulation")
    st.markdown(f"Running **{num_simulations}** iterations of Geometric Brownian Motion over a **{time_horizon}** trading-day horizon.")

    if st.button("🚀 Run Monte Carlo Simulation"):
        sim_results = simulate_gbm(
            port_mean, port_var, port_std, time_horizon, num_simulations, initial_investment
        )

        # Plot Simulation Pathways
        fig_sim = go.Figure()
        # Plot first 100 paths to prevent slowing down browser
        paths_to_plot = min(num_simulations, 100)
        for i in range(paths_to_plot):
            fig_sim.add_trace(go.Scatter(y=sim_results[:, i], mode='lines', line=dict(width=0.8), opacity=0.4, showlegend=False))
        
        # Add average path
        mean_path = np.mean(sim_results, axis=1)
        fig_sim.add_trace(go.Scatter(y=mean_path, mode='lines', name="Average Path", line=dict(color='yellow', width=2.5)))
        
        fig_sim.update_layout(
            title=f"First {paths_to_plot} Monte Carlo Simulation Pathways",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Value at Risk (VaR) Calculation
        final_values = sim_results[-1, :]
        var = value_at_risk(final_values, initial_investment, percentile=5)
        percentile_5, var_dollar, var_percentage = var["threshold"], var["dollar"], var["percent"]

        st.subheader("⚠️ Risk Assessment (Value at Risk)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"<div class='metric-card'><h4>5% Value at Risk (VaR) in Dollars</h4><h2>${var_dollar:,.2f}</h2><p>Maximum expected loss with 95% confidence over the time horizon</p></div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='metric-card'><h4>5% VaR as % of Portfolio</h4><h2>{var_percentage:.2f}%</h2><p>Portion of your initial investment at risk</p></div>", unsafe_allow_html=True)

        # Histogram of final values
        fig_hist = px.histogram(
            x=final_values,
            nbins=40,
            title="Distribution of Portfolio Values at Horizon End",
            labels={'x': 'Portfolio Value ($)'},
            template="plotly_dark"
        )
        fig_hist.add_vline(x=percentile_5, line_width=3, line_dash="dash", line_color="red", annotation_text="5% VaR Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Sentiment Analysis Layer
    st.header("📰 FinBERT & News Sentiment Analysis")
    st.markdown("Fetches current financial news articles from Yahoo Finance and outputs sentiment scores.")

    news_data = []
    # Grab latest news for primary ticker
    primary_ticker = tickers[0]
    with st.spinner(f"Fetching latest news for {primary_ticker}..."):
        try:
            ticker_obj = yf.Ticker(primary_ticker)
            ticker_news = ticker_obj.news[:5]  # Take top 5 news
            news_data = [n for n in map(parse_news_item, ticker_news) if n["title"]]
        except Exception as e:
            st.warning(f"Could not load live news: {e}. Showing mock sector news.")
            news_data = [
                {"title": f"{primary_ticker} Stock Braced for Inflation Report as Volatility Rises", "publisher": "Bloomberg", "link": "#"},
                {"title": f"Earnings Beat: {primary_ticker} Outperforms Sector Estimates in Q1", "publisher": "Reuters", "link": "#"},
                {"title": f"Market Correction Risks Loom Amid Interest Rate Guidance", "publisher": "Financial Times", "link": "#"},
                {"title": f"Investors Optimistic as {primary_ticker} Unveils AI Cloud Services", "publisher": "TechCrunch", "link": "#"},
            ]

    if news_data:
        # Load Model if selected
        classifier = None
        if use_finbert:
            classifier = load_finbert()
            if classifier is None:
                st.warning("Could not download FinBERT transformer model. Falling back to local dictionary-based scoring.")

        sentiments = []
        for news in news_data:
            title = news["title"]
            if classifier is not None:
                try:
                    res = classifier(title)[0]
                    label = res["label"]
                    score = res["score"]
                except Exception:
                    # fallback
                    res = analyze_sentiment_dict(title)
                    label = res["label"]
                    score = res["score"]
            else:
                res = analyze_sentiment_dict(title)
                label = res["label"]
                score = res["score"]

            sentiments.append({"title": title, "publisher": news["publisher"], "link": news["link"], "label": label, "score": score})

        # Display News Cards
        for sent in sentiments:
            label_color = "#28a745" if sent["label"] == "POSITIVE" else "#dc3545" if sent["label"] == "NEGATIVE" else "#ffc107"
            text_color = "#ffffff" if sent["label"] != "NEUTRAL" else "#000000"
            st.markdown(f"""
            <div style='background-color: #1a1e24; padding: 15px; border-radius: 8px; border-left: 5px solid {label_color}; margin-bottom: 12px;'>
                <a href='{sent["link"]}' target='_blank' style='font-size: 16px; font-weight: bold; color: #ffffff; text-decoration: none;'>{sent["title"]}</a>
                <p style='color: #888888; font-size: 12px; margin: 4px 0 0 0;'>Published by {sent["publisher"]} | Sentiment: 
                <span style='background-color: {label_color}; color: {text_color}; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;'>{sent["label"]} ({sent["score"]:.2%})</span></p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Please enter valid tickers and make sure you have internet access to download stock prices.")
