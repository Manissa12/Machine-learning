"""
Bitcoin Trading Signal Predictor - Streamlit Dashboard
Bonus project for TP Bitcoin Trading (Machine Learning A - EMLV)

Uses a pre-trained Random Forest model from the TP notebook.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Bitcoin Trading Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #f7931a, #ffcd00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff1744, #ff8a80);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Technical Indicators Functions (matching notebook exactly)
# =============================================================================

def EMA(df, n, col='Close'):
    return df[col].ewm(span=n, min_periods=n).mean()

def ROC(series, n):
    M = series.diff(n - 1)
    N = series.shift(n - 1)
    return ((M / N) * 100)

def MOM(series, n):
    return series.diff(n)

def RSI(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def STOK(close, low, high, n):
    return ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100

def STOD(close, low, high, n):
    return STOK(close, low, high, n).rolling(3).mean()

def MA(series, n):
    return series.rolling(n, min_periods=n).mean()

def compute_features(df):
    """Compute all technical indicators matching the notebook exactly"""
    df = df.copy()
    
    # Moving Averages for signal
    df['short_mavg'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['long_mavg'] = df['Close'].rolling(window=60, min_periods=1).mean()
    df['signal'] = np.where(df['short_mavg'] > df['long_mavg'], 1.0, 0.0)
    
    # Weighted_Price (approximate with typical price for Yahoo Finance data)
    df['Weighted_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # EMAs (10, 30, 200)
    df['EMA10'] = EMA(df, 10)
    df['EMA30'] = EMA(df, 30)
    df['EMA200'] = EMA(df, 200)
    
    # ROC (10, 30)
    df['ROC10'] = ROC(df['Close'], 10)
    df['ROC30'] = ROC(df['Close'], 30)
    
    # Momentum (10, 30)
    df['MOM10'] = MOM(df['Close'], 10)
    df['MOM30'] = MOM(df['Close'], 30)
    
    # RSI (10, 30, 200)
    df['RSI10'] = RSI(df['Close'], 10)
    df['RSI30'] = RSI(df['Close'], 30)
    df['RSI200'] = RSI(df['Close'], 200)
    
    # Stochastic (10, 30, 200)
    df['%K10'] = STOK(df['Close'], df['Low'], df['High'], 10)
    df['%D10'] = STOD(df['Close'], df['Low'], df['High'], 10)
    df['%K30'] = STOK(df['Close'], df['Low'], df['High'], 30)
    df['%D30'] = STOD(df['Close'], df['Low'], df['High'], 30)
    df['%K200'] = STOK(df['Close'], df['Low'], df['High'], 200)
    df['%D200'] = STOD(df['Close'], df['Low'], df['High'], 200)
    
    # Simple MAs (21, 63, 252)
    df['MA21'] = MA(df['Close'], 21)
    df['MA63'] = MA(df['Close'], 63)
    df['MA252'] = MA(df['Close'], 252)
    
    return df

# Feature columns matching the notebook EXACTLY (22 features)
FEATURE_COLS = [
    'Close', 'Volume_(BTC)', 'Weighted_Price',
    'EMA10', 'EMA30', 'EMA200',
    'ROC10', 'ROC30',
    'MOM10', 'MOM30',
    'RSI10', 'RSI30', 'RSI200',
    '%K10', '%D10', '%K30', '%D30', '%K200', '%D200',
    'MA21', 'MA63', 'MA252'
]

# =============================================================================
# Load Pre-trained Model
# =============================================================================

@st.cache_resource
def load_model():
    model_path = "bitcoin_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    st.error(f"Model file not found: {model_path}")
    return None

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=300)
def load_bitcoin_data(period="1y", interval="1h"):
    """Load Bitcoin data - need 1y for MA252"""
    try:
        btc = yf.download("BTC-USD", period=period, interval=interval, progress=False)
        if btc.empty:
            return None
        btc = btc.reset_index()
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [col[0] if col[1] == '' else col[0] for col in btc.columns]
        return btc
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=100)
st.sidebar.title("Settings")

st.sidebar.markdown("---")
st.sidebar.info("**Model**: Pre-trained Random Forest from TP Bitcoin Trading notebook (22 features)")

# =============================================================================
# Main Content
# =============================================================================

st.markdown('<h1 class="main-header">Bitcoin Trading Signals</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Real-time predictions using pre-trained ML model</p>", unsafe_allow_html=True)

# Load model
model = load_model()
if model is None:
    st.stop()

# Load data (need 1y for MA252)
with st.spinner("Fetching Bitcoin data..."):
    df = load_bitcoin_data(period="1y", interval="1h")

if df is None or df.empty:
    st.error("Could not load Bitcoin data. Please try again later.")
    st.stop()

# Rename Volume column to match notebook
if 'Volume' in df.columns:
    df['Volume_(BTC)'] = df['Volume']

# Compute features
df_features = compute_features(df)

# Check we have all features
missing_features = [f for f in FEATURE_COLS if f not in df_features.columns]
if missing_features:
    st.error(f"Missing features: {missing_features}")
    st.stop()

# Get latest prediction
latest_data = df_features[FEATURE_COLS].dropna().iloc[-1:]

if not latest_data.empty:
    # Scale features (fit on recent data)
    scaler = StandardScaler()
    recent_data = df_features[FEATURE_COLS].dropna().tail(1000)
    scaler.fit(recent_data)
    latest_scaled = scaler.transform(latest_data)
    
    prediction = model.predict(latest_scaled)[0]
    prediction_proba = model.predict_proba(latest_scaled)[0]
else:
    prediction = df_features['signal'].iloc[-1] if 'signal' in df_features.columns else 0
    prediction_proba = [0.5, 0.5]
    st.warning("Using rule-based signal (insufficient data)")

# =============================================================================
# Metrics Row
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = df['Close'].iloc[-1]
    price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")

with col2:
    st.metric("24h High", f"${df['High'].tail(24).max():,.2f}")

with col3:
    st.metric("24h Low", f"${df['Low'].tail(24).min():,.2f}")

with col4:
    st.metric("Model", "Random Forest")

# =============================================================================
# Signal Display
# =============================================================================

st.markdown("---")
col_signal, col_proba = st.columns([1, 1])

with col_signal:
    if prediction == 1:
        st.markdown('<div class="signal-buy">BUY SIGNAL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-sell">SELL SIGNAL</div>', unsafe_allow_html=True)

with col_proba:
    st.markdown("### Signal Confidence")
    fig_conf = go.Figure(data=[
        go.Bar(
            x=['Sell', 'Buy'],
            y=prediction_proba,
            marker_color=['#ff1744', '#00c853'],
            text=[f'{p*100:.1f}%' for p in prediction_proba],
            textposition='auto'
        )
    ])
    fig_conf.update_layout(
        height=200, template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Probability", showlegend=False
    )
    st.plotly_chart(fig_conf, use_container_width=True)

# =============================================================================
# Price Chart
# =============================================================================

st.markdown("---")
st.subheader("Price Chart with Signals")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Bitcoin Price & Moving Averages", "RSI", "Signal")
)

fig.add_trace(go.Candlestick(
    x=df_features.index, open=df_features['Open'], high=df_features['High'],
    low=df_features['Low'], close=df_features['Close'], name='BTC-USD'
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_features.index, y=df_features['short_mavg'], 
    name='MA10', line=dict(color='#00c853', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_features.index, y=df_features['long_mavg'], 
    name='MA60', line=dict(color='#ff1744', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=df_features.index, y=df_features['RSI10'], 
    name='RSI(10)', line=dict(color='#9c27b0', width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

colors = ['#ff1744' if s == 0 else '#00c853' for s in df_features['signal'].dropna()]
fig.add_trace(go.Bar(x=df_features.index, y=df_features['signal'], 
    name='Signal', marker_color=colors), row=3, col=1)

fig.update_layout(
    height=800, template='plotly_dark', showlegend=True,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Feature Importance
# =============================================================================

st.markdown("---")
st.subheader("Feature Importance")

if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance_df['Importance'], y=importance_df['Feature'],
        orientation='h', marker_color='#f7931a'
    ))
    fig_imp.update_layout(
        height=500, template='plotly_dark',
        title="Which indicators matter most?",
        xaxis_title="Importance", yaxis_title=""
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Disclaimer:</strong> Educational project only. Not financial advice.</p>
    <p>Made for EMLV Machine Learning A - TP Bitcoin Trading Bonus</p>
    <p>Data: Yahoo Finance | Model: Pre-trained Random Forest (22 features)</p>
</div>
""", unsafe_allow_html=True)
