---
title: Bitcoin Trading Signals
emoji: 📈
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# Bitcoin Trading Signals Dashboard

Real-time Bitcoin trading signals using a **pre-trained Random Forest model** from TP Bitcoin Trading (Machine Learning A - EMLV).

## Features
- Real-time BTC price from Yahoo Finance
- Buy/Sell signals with confidence percentage
- Interactive candlestick charts
- Feature importance visualization

## Model
Pre-trained Random Forest trained on historical Bitcoin data with technical indicators:
- EMA (10, 30)
- RSI (10, 30)
- Stochastic Oscillator (%K, %D)
- Momentum & ROC

**Disclaimer**: Educational project only. Not financial advice.
