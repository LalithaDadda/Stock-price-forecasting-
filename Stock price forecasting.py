import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Download stock data
stock = input("Enter stock symbol (e.g., TCS.NS): ")

data = yf.download(stock, period="2y", progress=False)

if data.empty:
    print("Invalid stock symbol or data unavailable.")
    raise SystemExit

# Prepare data
df = data[["Close"]].dropna().copy()

# Create previous-day price features
for i in range(1, 6):
    df[f"Day_{i}"] = df["Close"].shift(i)

df.dropna(inplace=True)

X = df[[f"Day_{i}" for i in range(1, 6)]]
y = df["Close"]

# Split data chronologically
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2))

# Forecast next trading day's price
latest = df["Close"].iloc[-5:].to_numpy()[::-1].reshape(1, -1)
forecast = model.predict(latest)[0]

print("Latest closing price:", round(df["Close"].iloc[-1], 2))
print("Forecast next trading day:", round(forecast, 2))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price")
plt.plot(y_test.index, predictions, label="Predicted Price")
plt.title("Stock Price Forecasting")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.tight_layout()
plt.show()
