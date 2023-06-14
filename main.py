import requests
import pandas as pd
from prophet import Prophet
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool

# 1. Fetch BTCUSDT data from public endpoint for the last 10 days
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 100
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

# 3. Plot the price data and the forecasted price on the same plot using Bokeh
output_notebook()

source_actual = ColumnDataSource(prophet_df)
source_forecast = ColumnDataSource(forecast)

hover = HoverTool(tooltips=[("Date", "@ds{%F}"), ("Price", "@y")], formatters={"@ds": "datetime"})

p = figure(title="LTCUSDT Price Forecast", x_axis_label="Date", x_axis_type="datetime", y_axis_label="Price", tools=[hover])
p.line(x="ds", y="y", source=source_actual, legend_label="Actual Price", color="blue")
p.line(x="ds", y="yhat", source=source_forecast, legend_label="Forecasted Price", color="red")
p.legend.location = "top_left"

show(p)
