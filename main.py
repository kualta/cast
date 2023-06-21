"""
Demand forecasting using Prophet (c) kualta 2023
Author: kualta <contact@kualta.dev>
License: MIT
"""

from math import sqrt
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxZoomTool,
    WheelZoomTool,
    ResetTool,
    PanTool,
    SaveTool,
    BoxSelectTool,
    TapTool,
    CrosshairTool,
)
from bokeh.models.widgets import Select, TextInput
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.models.widgets import Div
from sklearn.metrics import mean_squared_error, mean_absolute_error

URL = "https://api.binance.com/api/v3/klines"
source_actual = ColumnDataSource()
source_prophet_forecast = ColumnDataSource()

hover_tool = HoverTool(
    tooltips=[
        ("Date", "@ds{%F}"),
        ("Actual Price", "@y"),
        ("Forecasted Price", "@yhat"),
    ],
    formatters={"@ds": "datetime"},
)
symbol_select = TextInput(value="BTCUSDT", title="Symbol:")
interval_select = Select(
    title="Interval", options=["1m", "1h", "1d", "3d", "1w"], value="3d"
)
predict_period_input = TextInput(value="30", title="Prediction Periods:")
seasonality_mode_select = Select(
    title="Seasonality Mode", options=["additive", "multiplicative"], value="additive"
)
changepoint_prior_scale_input = TextInput(
    value="0.05", title="Changepoint Prior Scale:"
)
mae_div = Div(text="")
mse_div = Div(text="")
rmse_div = Div(text="")
mape_div = Div(text="")


def get_data(symbol, interval, limit):
    """Parse price data from Binance"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(URL, params=params, timeout=100)
    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_prophet_forecast(df, periods):
    """Fit the prophet model and precit future periods"""
    prophet_df = df[["timestamp", "close"]].rename(
        columns={"timestamp": "ds", "close": "y"}
    )
    prophet_df["y"] = pd.to_numeric(prophet_df["y"])
    model = Prophet(
        seasonality_mode=seasonality_mode_select.value,
        changepoint_prior_scale=float(changepoint_prior_scale_input.value),
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(df, forecast):
    """Calculate error metrics for the forecast"""
    mae = mean_absolute_error(df["close"], forecast["yhat"][: len(df)])
    mse = mean_squared_error(df["close"], forecast["yhat"][: len(df)])
    rmse = sqrt(mse)
    mape = (
        np.mean(np.abs((df["close"] - forecast["yhat"][: len(df)]) / df["close"])) * 100
    )
    return mae, mse, rmse, mape


def update_data(attr, old, new):
    """Callback for UI widgets"""
    symbol = symbol_select.value
    interval = interval_select.value
    periods = int(predict_period_input.value)
    df = get_data(symbol, interval, 1000)
    df["close"] = pd.to_numeric(df["close"])

    prophet_forecast = get_prophet_forecast(df, periods)

    source_actual.data = df[["timestamp", "close"]]
    source_prophet_forecast.data = prophet_forecast[["ds", "yhat"]]

    # Update error metric widgets
    mae, mse, rmse, mape = evaluate_forecast(df, prophet_forecast)
    mae_div.text = f"Mean Absolute Error: {mae}"
    mse_div.text = f"Mean Squared Error: {mse}"
    rmse_div.text = f"Root Mean Squared Error: {rmse}"
    mape_div.text = f"Mean Absolute Percentage Error: {mape}%"


# Plot
p = figure(
    title=f"{symbol_select.value} Price Forecast",
    x_axis_label="Date",
    x_axis_type="datetime",
    width=1600,
    tools=[
        hover_tool,
        BoxZoomTool(),
        WheelZoomTool(),
        ResetTool(),
        PanTool(),
        SaveTool(),
        BoxSelectTool(),
        TapTool(),
        CrosshairTool(),
    ],
)
p.line(
    source=source_actual,
    x="timestamp",
    y="close",
    color="blue",
    legend_label="Actual Price",
)
p.line(
    source=source_prophet_forecast,
    x="ds",
    y="yhat",
    color="red",
    legend_label="Prophet Forecast",
)
p.add_tools(hover_tool)

symbol_select.on_change("value", update_data)
interval_select.on_change("value", update_data)
predict_period_input.on_change("value", update_data)
seasonality_mode_select.on_change("value", update_data)
changepoint_prior_scale_input.on_change("value", update_data)

# Layout
layout = column(
    row(
        symbol_select,
        interval_select,
        predict_period_input,
        seasonality_mode_select,
        changepoint_prior_scale_input,
    ),
    p,
    mae_div,
    mse_div,
    rmse_div,
    mape_div,
)
curdoc().add_root(layout)
