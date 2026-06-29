import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

df = pd.read_csv("NVDA_EDA_Processed.csv")

df['Date'] = pd.to_datetime(df['Date'])

# Moving Averages
df['MA5'] = df['Close'].rolling(window=5).mean()

df['MA20'] = df['Close'].rolling(window=20).mean()

# Rolling Volatility
df['Volatility'] = (
    df['Daily_Return']
    .rolling(window=20)
    .std()
)

# RSI
rsi = RSIIndicator(
    close=df['Close'],
    window=14
)

df['RSI'] = rsi.rsi()

# MACD
macd = MACD(close=df['Close'])

df['MACD'] = macd.macd()

# Remove missing values created by indicators
df.dropna(inplace=True)

print(df.head())

print(df.columns)

df.to_csv(
    "NVDA_Feature_Engineered.csv",
    index=False
)

# Figure 4.5 — Moving Average Analysis
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(df['Date'], df['Close'], label='Close')

plt.plot(df['Date'], df['MA5'], label='MA5')

plt.plot(df['Date'], df['MA20'], label='MA20')

plt.legend()

plt.title('NVIDIA Closing Price with Moving Averages')

plt.show()

# Figure 4.6 — RSI Indicator
plt.figure(figsize=(12,6))

plt.plot(df['Date'], df['RSI'])

plt.axhline(70)

plt.axhline(30)

plt.title('Relative Strength Index (RSI)')

plt.show()

# Figure 4.7 — MACD Indicator
plt.figure(figsize=(12,6))

plt.plot(df['Date'], df['MACD'])

plt.title('MACD Indicator')

plt.show()

#
print(df.columns)
#
print(df.shape)