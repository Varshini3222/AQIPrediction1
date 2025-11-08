# hybrid_dt_knn.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import joblib

print("--- Hybrid Model (Decision Tree + KNN Regression) ---")

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

# 4. Train Individual Models
print("\n--- Training Individual Models ---")

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# KNN Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# 5. Hybrid Model (Weighted Average)
print("\n--- Hybrid Model (Weighted Average) ---")

# Calculate weights based on individual model R² scores
dt_r2 = r2_score(y_test, dt_pred)
knn_r2 = r2_score(y_test, knn_pred)

total_r2 = dt_r2 + knn_r2
dt_weight = dt_r2 / total_r2
knn_weight = knn_r2 / total_r2

print(f"Decision Tree Weight: {dt_weight:.4f} (R²: {dt_r2:.4f})")
print(f"KNN Regression Weight: {knn_weight:.4f} (R²: {knn_r2:.4f})")

# Hybrid prediction
hybrid_pred = (dt_pred * dt_weight) + (knn_pred * knn_weight)

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

# 7. Compare with Individual Models
print("\n### Model Comparison ###")
models = {
    'Decision Tree': dt_pred,
    'KNN Regression': knn_pred,
    'Hybrid Model': hybrid_pred
}

for name, pred in models.items():
    r2_val = r2_score(y_test, pred)
    category_acc = accuracy_score(aqi_to_category(y_test), aqi_to_category(pred)) * 100
    print(f"{name:20} | R²: {r2_val:.4f} | Category Acc: {category_acc:.2f}%")

# 8. Save Hybrid Model
hybrid_model = {
    'dt_model': dt_model,
    'knn_model': knn_model,
    'weights': {'dt': dt_weight, 'knn': knn_weight}
}

joblib.dump(hybrid_model, 'hybrid_dt_knn_model.joblib')
print("\n💾 Hybrid model saved as 'hybrid_dt_knn_model.joblib'")