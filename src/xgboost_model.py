# Step 2 Load Dataset
import pandas as pd
import numpy as np

df = pd.read_csv("NVDA_Feature_Engineered.csv")

# Target Variable
df['Target'] = (
    df['Daily_Return'].shift(-1) > 0
).astype(int)

df.dropna(inplace=True)

# Step 3: Select Features
features = [
    'MA5',
    'MA20',
    'Volatility',
    'RSI',
    'MACD',
    'Volume'
]

X = df[features]

y = df['Target']

#Step 4: Chronological Train-Test Split
train_size = int(len(df) * 0.85)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Step 5: Train XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

# Step 6: Predictions
predictions = xgb.predict(X_test)

# Step 7: Performance Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("\nXGBoost Results")

print("Accuracy =", round(accuracy,4))
print("Precision =", round(precision,4))
print("Recall =", round(recall,4))
print("F1 Score =", round(f1,4))

# Figure 4.12: XGBoost Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title(
    'XGBoost Confusion Matrix'
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.tight_layout()

plt.show()

# Figure 4.13: XGBoost Feature Importance
importance = xgb.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title(
    'XGBoost Feature Importance'
)

plt.tight_layout()

plt.show()