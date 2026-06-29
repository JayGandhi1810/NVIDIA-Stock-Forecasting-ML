import pandas as pd

# Load the dataset
df = pd.read_csv("NVDA_Historical_Data.csv")

print(df.head())
print(df.info())
print(df.describe())

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

# Check Missing Values
print(df.isnull().sum())

# Figure 4.1 — NVDA Closing Price Trend
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])

plt.title('NVIDIA Closing Price (2015–2025)')
plt.xlabel('Year')
plt.ylabel('Closing Price ($)')

plt.tight_layout()
plt.show()

# Figure 4.2 — Trading Volume Trend
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Volume'])

plt.title('NVIDIA Trading Volume')
plt.xlabel('Year')
plt.ylabel('Volume')

plt.tight_layout()
plt.show()

# Step 6: Create Daily Returns
df['Daily_Return'] = df['Close'].pct_change()

# Step 7: Figure 4.3 — Return Distribution
import seaborn as sns

plt.figure(figsize=(10,6))

sns.histplot(
    df['Daily_Return'].dropna(),
    bins=50,
    kde=True
)

plt.title('Distribution of Daily Returns')

plt.tight_layout()
plt.show()

# Step 8: Figure 4.4 — Correlation Heatmap
corr = df[
    ['Open','High','Low',
     'Close','Adj Close',
     'Volume']
].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

#Step 9: Save Clean Dataset
df.to_csv(
    "NVDA_EDA_Processed.csv",
    index=False
)

#
print(df.describe())
#
print(df.isnull().sum())

# Feature Engineering

