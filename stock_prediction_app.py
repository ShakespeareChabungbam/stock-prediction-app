# ==========================================================
# 🎯 Streamlit App: Stock Market Prediction using Random Forest
# 📘 Author: Shakespeare Chabungbam
# 🏫 MCA Minor Project
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Reliance Stock Prediction", page_icon="📈", layout="centered")

st.title("📈 Reliance Stock Market Prediction using Random Forest")
st.markdown("### Author: **Shakespeare Chabungbam (MCA Minor Project)**")

st.write("""
This web application uses a **Random Forest Regressor** to predict future **Reliance Industries Ltd. (RELIANCE.NS)** stock prices  
based on historical market data from Yahoo Finance.
""")

# ----------------------------------------------------------
# User Input Section
# ----------------------------------------------------------
ticker = st.text_input("🔹 Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS, INFY.NS):", "RELIANCE.NS")

period = st.selectbox("📅 Select Data Period:", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.selectbox("⏱ Select Data Interval:", ["1d", "1wk", "1mo"], index=0)

# ----------------------------------------------------------
# Download Stock Data
# ----------------------------------------------------------
st.write("📥 **Fetching stock data...**")

try:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)

    if len(data) < 60:
        st.warning("⚠️ Not enough data for reliable prediction. Try a longer period.")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to download data: {e}")
    st.stop()

st.success(f"✅ Data downloaded successfully for {ticker}! ({len(data)} records)")
st.line_chart(data["Close"])

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------
st.write("🧠 **Preparing features for model training...**")

data["Prev_Close"] = data["Close"].shift(1)
data["Price_Change"] = data["Close"] - data["Prev_Close"]
data["3day_MA"] = data["Close"].rolling(window=3).mean()
data["7day_MA"] = data["Close"].rolling(window=7).mean()
data["Volatility"] = data["Close"].rolling(window=5).std()
data.dropna(inplace=True)

features = ["Open", "High", "Low", "Volume", "Prev_Close", "3day_MA", "7day_MA", "Price_Change", "Volatility"]
target = "Close"

X = data[features]
y = data[target]

# ----------------------------------------------------------
# Train/Test Split
# ----------------------------------------------------------
split_ratio = st.slider("📊 Train/Test Split (%)", min_value=60, max_value=90, value=80, step=5)
test_size = (100 - split_ratio) / 100

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# ----------------------------------------------------------
# Model Training
# ----------------------------------------------------------
st.write("🚀 **Training Random Forest Model...**")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# Prediction and Evaluation
# ----------------------------------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance (Test Data)")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("R² Score", f"{r2:.3f}")

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
st.write("### 📈 Actual vs Predicted Closing Prices")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label='Actual Price', color='blue')
ax.plot(y_pred, label='Predicted Price', color='orange')
ax.set_xlabel("Days")
ax.set_ylabel("Price (₹)")
ax.set_title(f"{ticker} - Random Forest Stock Price Prediction")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ----------------------------------------------------------
# Next Day Prediction
# ----------------------------------------------------------
st.write("### 🔮 Next-Day Price Prediction")

last_row = data.iloc[-1][features].values.reshape(1, -1)
next_day_price = model.predict(last_row)[0]
last_close = data["Close"].iloc[-1]
change = next_day_price - last_close
direction = "🔺 Increase" if change > 0 else "🔻 Decrease"

st.markdown(f"**Predicted Next-Day Closing Price:** ₹{next_day_price:.2f}")
st.markdown(f"**Expected Movement:** {direction} ({change:+.2f} ₹)")

# ----------------------------------------------------------
# Data Download Option
# ----------------------------------------------------------
st.download_button(
    label="📊 Download Processed Data (CSV)",
    data=data.to_csv().encode('utf-8'),
    file_name=f"{ticker}_processed_data.csv",
    mime="text/csv"
)

st.caption("© 2025 | MCA Minor Project by Shakespeare Chabungbam | Model: Random Forest (Reliance Stock)")
