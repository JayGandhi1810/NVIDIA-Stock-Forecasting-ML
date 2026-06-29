import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv("NVDA_Feature_Engineered.csv")

# Use Close Price
series = df['Close']

# 85% Train, 15% Test
train_size = int(len(series) * 0.85)

train = series[:train_size]
test = series[train_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# ARIMA Model
model = ARIMA(
    train,
    order=(5,1,0)
)

model_fit = model.fit()

print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(
    steps=len(test)
)

# Metrics
mae = mean_absolute_error(
    test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)

print("\nARIMA Results")
print("MAE =", round(mae,4))
print("RMSE =", round(rmse,4))

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    test.values,
    label="Actual"
)

plt.plot(
    forecast,
    label="Forecast"
)

plt.title(
    "ARIMA Forecast vs Actual"
)

plt.xlabel("Time")

plt.ylabel("Close Price")

plt.legend()

plt.tight_layout()

plt.show()