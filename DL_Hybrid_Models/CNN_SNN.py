# DL_Hybrid_CNN_SNN.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                           accuracy_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
import joblib

def aqi_to_category(aqi_values):
    categories = []
    for aqi in aqi_values:
        if aqi <= 50: categories.append("Good")
        elif aqi <= 100: categories.append("Satisfactory")
        elif aqi <= 200: categories.append("Moderate")
        elif aqi <= 300: categories.append("Poor")
        elif aqi <= 400: categories.append("Very Poor")
        else: categories.append("Severe")
    return categories

# Load data
df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

df = df[features + [target]].dropna()
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tolerance = 20
categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Reshape for CNN
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print("="*70)
print("DEEP LEARNING HYBRID: CNN + SNN (SlideNN)")
print("="*70)

# Train CNN Model
cnn_model = Sequential([
    Input(shape=(1, X_train.shape[1])),
    Conv1D(64, kernel_size=1, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_seq, y_train, epochs=20, batch_size=32, verbose=0)
cnn_pred = cnn_model.predict(X_test_seq).flatten()

# Train SNN (SlideNN) Model
snn_model = Sequential([Input(shape=(X_train.shape[1],))])
for units in [128, 64, 32]:
    snn_model.add(Dense(units, activation='relu'))
snn_model.add(Dense(1))
snn_model.compile(optimizer='adam', loss='mse')
snn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
snn_pred = snn_model.predict(X_test).flatten()

# Hybrid combination - Simple Average
hybrid_pred = (cnn_pred + snn_pred) / 2

# Calculate all metrics
r2 = r2_score(y_test, hybrid_pred)
mae = mean_absolute_error(y_test, hybrid_pred)
rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
accuracy_tolerance = np.mean(np.abs(hybrid_pred - y_test) <= tolerance) * 100
true_categories = aqi_to_category(y_test)
predicted_categories = aqi_to_category(hybrid_pred)
accuracy_category = accuracy_score(true_categories, predicted_categories) * 100

# Print results
print(f"\n📊 PERFORMANCE METRICS:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Accuracy (±{tolerance} points): {accuracy_tolerance:.2f}%")
print(f"Category Accuracy: {accuracy_category:.2f}%")

# Confusion Matrix
print(f"\n🎯 CONFUSION MATRIX:")
cm = confusion_matrix(true_categories, predicted_categories, labels=categories)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - CNN + SNN\nAccuracy: {:.2f}%'.format(accuracy_category))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_dl_cnn_snn.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)

joblib.dump({'cnn': cnn_model, 'snn': snn_model},
           'hybrid_dl_cnn_snn.joblib')