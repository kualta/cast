
from prophet.diagnostics import cross_validation

# Define the number of days to forecast
forecast_horizon = 30

# Define the frequency of the data
freq = 'D'

# Define the range of values to test for changepoint_prior_scale
param_grid = {'changepoint_prior_scale': [0.001, 0.01, 0.1, 1]}

# Run cross-validation to evaluate the performance of the model
df_cv = cross_validation(model, horizon=forecast_horizon, period=freq, param_grid=param_grid)