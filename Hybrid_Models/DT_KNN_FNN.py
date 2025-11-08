# hybrid_fnn_dt_knn_visual.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                           accuracy_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

print("--- Hybrid Model (FNN + Decision Tree + KNN Regression) ---")
print("=== WITH VISUAL CONFUSION MATRIX ===\n")

# 1. Load dataset
try:
    df = pd.read_csv('../AQI_complete_imputed_2014_2025.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Dataset not found.")
    exit()

# 2. Define AQI category mapping
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

# 3. Prepare Data
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

df = df[features + [target]].dropna()
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for FNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Individual Models
print("\n--- Training Individual Models ---")

# FNN Model
print("Training FNN...")
fnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

fnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
fnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
fnn_pred = fnn_model.predict(X_test_scaled).flatten()

# Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# KNN Regression
print("Training KNN Regression...")
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# 5. Hybrid Model (Weighted Average)
print("\n--- Hybrid Model (Weighted Average) ---")

# Calculate weights based on individual model R² scores
fnn_r2 = r2_score(y_test, fnn_pred)
dt_r2 = r2_score(y_test, dt_pred)
knn_r2 = r2_score(y_test, knn_pred)

total_r2 = fnn_r2 + dt_r2 + knn_r2
fnn_weight = fnn_r2 / total_r2
dt_weight = dt_r2 / total_r2
knn_weight = knn_r2 / total_r2

print(f"FNN Weight: {fnn_weight:.4f} (R²: {fnn_r2:.4f})")
print(f"Decision Tree Weight: {dt_weight:.4f} (R²: {dt_r2:.4f})")
print(f"KNN Regression Weight: {knn_weight:.4f} (R²: {knn_r2:.4f})")

# Hybrid prediction
hybrid_pred = (fnn_pred * fnn_weight) + (dt_pred * dt_weight) + (knn_pred * knn_weight)

# 6. Metrics for Hybrid Model
print("\n### Hybrid Model Metrics ###")

# Regression Metrics
r2 = r2_score(y_test, hybrid_pred)
mae = mean_absolute_error(y_test, hybrid_pred)
rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))

print(f"🎯 R-squared (R²): {r2:.4f}")
print(f"📏 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📐 Root Mean Squared Error (RMSE): {rmse:.2f}")

# Accuracy Metrics
tolerance = 20
is_accurate_tolerance = np.abs(hybrid_pred - y_test) <= tolerance
accuracy_tolerance = np.mean(is_accurate_tolerance) * 100
print(f"✅ Accuracy within ±{tolerance} point tolerance: {accuracy_tolerance:.2f}%")

true_categories = aqi_to_category(y_test)
predicted_categories = aqi_to_category(hybrid_pred)
accuracy_category = accuracy_score(true_categories, predicted_categories) * 100
print(f"✅ Accuracy based on AQI categories: {accuracy_category:.2f}%")

# 7. VISUAL CONFUSION MATRIX
print("\n" + "="*60)
print("VISUAL CONFUSION MATRIX")
print("="*60)

categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
cm = confusion_matrix(true_categories, predicted_categories, labels=categories)

# Create a beautiful visual confusion matrix
plt.figure(figsize=(12, 10))

# Create heatmap
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=categories, yticklabels=categories,
                 cbar_kws={'label': 'Number of Predictions'})

# Customize the plot
plt.title('Confusion Matrix - Hybrid FNN+DT+KNN Model\n', fontsize=16, fontweight='bold')
plt.xlabel('Predicted AQI Category', fontsize=12, fontweight='bold')
plt.ylabel('Actual AQI Category', fontsize=12, fontweight='bold')

# Add accuracy to the title
plt.suptitle(f'Overall Accuracy: {accuracy_category:.2f}%',
             y=0.92, fontsize=14, color='green')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add grid lines
ax.set_xticks(np.arange(len(categories)) + 0.5, minor=False)
ax.set_yticks(np.arange(len(categories)) + 0.5, minor=False)
ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Tight layout and save
plt.tight_layout()
plt.savefig('hybrid_model_confusion_matrix_visual.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Save Hybrid Model
hybrid_model = {
    'fnn_model': fnn_model,
    'dt_model': dt_model,
    'knn_model': knn_model,
    'scaler': scaler,
    'weights': {'fnn': fnn_weight, 'dt': dt_weight, 'knn': knn_weight}
}

# Save TensorFlow model separately
fnn_model.save('fnn_model.h5')
joblib.dump(hybrid_model, 'hybrid_fnn_dt_knn_model.joblib')
print("\n💾 Hybrid model saved as 'hybrid_fnn_dt_knn_model.joblib'")
print("💾 FNN model saved as 'fnn_model.h5'")

print("\n✅ Hybrid model analysis complete! Visual confusion matrix saved.")