import pandas as pd
import numpy as np

df = pd.read_csv("NVDA_Feature_Engineered.csv")

#Step 3: Create Target Variable
df['Target'] = (
    df['Daily_Return'].shift(-1) > 0
).astype(int)

df.dropna(inplace=True)

#Step 4: Select Features
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

#Step 5: Scale Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

#Step 6: Create Sequences
sequence_length = 10

X_seq = []
y_seq = []

for i in range(sequence_length, len(X_scaled)):
    
    X_seq.append(
        X_scaled[
            i-sequence_length:i
        ]
    )
    
    y_seq.append(
        y.iloc[i]
    )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(X_seq.shape)

#Step 7: Train-Test Split
train_size = int(
    len(X_seq) * 0.85
)

X_train = X_seq[:train_size]
X_test = X_seq[train_size:]

y_train = y_seq[:train_size]
y_test = y_seq[train_size:]

#Step 8: Build LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(32)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

#Step 9: Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

#Step 10: Generate Predictions
raw_predictions = model.predict(X_test)

print(raw_predictions[:20])
print("Min:", raw_predictions.min())
print("Max:", raw_predictions.max())
print("Mean:", raw_predictions.mean())

predictions = (
    raw_predictions > 0.27
).astype(int)


#Step 11: Performance Metrics
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

print("\nLSTM Results")

print("Accuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1 =", f1)

# Figure 4.14: Training Loss
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.legend()

plt.title(
    'LSTM Training Loss'
)

plt.tight_layout()

plt.show()

#Figure 4.15: LSTM Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.title('LSTM Confusion Matrix')
plt.colorbar()

plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i, j],
                 ha='center',
                 va='center')

plt.tight_layout()
plt.show()

# python
import numpy as np

print(np.unique(predictions, return_counts=True))

# checking 
print(df['Target'].value_counts())
print(predictions[:20])

print(history.history.keys())

print("Final Training Accuracy:",
      history.history['accuracy'][-1])

print("Final Validation Accuracy:",
      history.history['val_accuracy'][-1])

print("Final Training Loss:",
      history.history['loss'][-1])

print("Final Validation Loss:",
      history.history['val_loss'][-1])