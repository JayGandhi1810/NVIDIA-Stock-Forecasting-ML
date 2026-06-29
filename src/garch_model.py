import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model

# Load Dataset
df = pd.read_csv("NVDA_Feature_Engineered.csv")

# Daily Returns
returns = df['Daily_Return'].dropna() * 100

# Train-Test Split
train_size = int(len(returns) * 0.85)

train = returns[:train_size]
test = returns[train_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# GARCH(1,1)
model = arch_model(
    train,
    vol='Garch',
    p=1,
    q=1,
    mean='Zero'
)

garch_fit = model.fit(disp='off')

print(garch_fit.summary())

# Forecast Volatility
forecast = garch_fit.forecast(
    horizon=len(test)
)

# Extract variance forecasts
variance_forecast = forecast.variance.iloc[-1]

volatility_forecast = np.sqrt(
    variance_forecast
)

print("\nForecasted Volatility:")
print(volatility_forecast.head())

# Step 2: Create Figure 4.9
plt.figure(figsize=(12,6))

plt.plot(
    test.reset_index(drop=True),
    label='Actual Returns'
)

plt.title(
    'Actual NVIDIA Returns'
)

plt.xlabel('Time')

plt.ylabel('Return (%)')

plt.legend()

plt.tight_layout()

plt.show()

# For Better Dissertation Figure
rolling_vol = (
    returns
    .rolling(window=20)
    .std()
)

plt.figure(figsize=(12,6))

plt.plot(
    rolling_vol,
    label='Rolling Volatility'
)

plt.title(
    'NVIDIA Rolling Volatility'
)

plt.xlabel('Time')

plt.ylabel('Volatility')

plt.legend()

plt.tight_layout()

plt.show()