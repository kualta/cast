import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Fetch BTCUSDT data from public endpoint for the last 10 days
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 1000
}
response = requests.get(url, params=params)
data = response.json()

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 2. Use Prophet python library to create a demand forecasting model
prophet_df = df[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})
prophet_df["y"] = pd.to_numeric(prophet_df["y"])

model = Prophet()
model.fit(prophet_df)

# Make future predictions
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

# 3. Plot the price data and the forecasted price on the same plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prophet_df["ds"], prophet_df["y"], label="Actual Price")
ax.plot(forecast["ds"], forecast["yhat"], label="Forecasted Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.show()
