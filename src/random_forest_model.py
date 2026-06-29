import pandas as pd
import numpy as np

df = pd.read_csv("NVDA_Feature_Engineered.csv")

# Next-day return direction

df['Target'] = (
    df['Daily_Return'].shift(-1) > 0
).astype(int)

df.dropna(inplace=True)

# Step 2: Selection Features
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

#Step 3: Train-Test Split
train_size = int(len(df) * 0.85)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

#Step 4: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

# Step 5: Predictions
predictions = rf.predict(X_test)


# Step 6: Performance Metrics
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

print("\nRandom Forest Results")

print("Accuracy =", round(accuracy,4))
print("Precision =", round(precision,4))
print("Recall =", round(recall,4))
print("F1 Score =", round(f1,4))

# Step 7: Figure 4.10 — Confusion Matrix
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
    cmap='Blues'
)

plt.title(
    'Random Forest Confusion Matrix'
)

plt.xlabel(
    'Predicted'
)

plt.ylabel(
    'Actual'
)

plt.tight_layout()

plt.show()

# Step 8: Feature Importance Figure
importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = (
    feature_importance
    .sort_values(
        by='Importance',
        ascending=False
    )
)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title(
    'Random Forest Feature Importance'
)

plt.tight_layout()

plt.show()