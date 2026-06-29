import yfinance as yf
import pandas as pd

# NVIDIA historical data
nvda = yf.download(
    "NVDA",
    start="2015-01-01",
    end="2025-12-31",
    auto_adjust=False
)

# Required data into coloumns.
nvda.reset_index(inplace=True)

# Select and reorder columns
nvda = nvda[
    ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
]

# Saving into CSV file format.
nvda.to_csv('NVDA_Historical_Data.csv', index=False)

print(nvda.head())