import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import joblib

print("--- Lasso Regression (New Dataset: 2014–2025) ---")

# 1. Load dataset
try:
    df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'AQI_complete_imputed_2014_2025.csv' not found.")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Lasso Regression
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# 5. Predictions
predictions = model.predict(X_test)

# ================= Regression Metrics =================
print("\n### Regression Metrics ###")
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"🎯 R-squared (R²): {r2:.4f}")
print(f"📏 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📐 Root Mean Squared Error (RMSE): {rmse:.2f}")

# ================= Accuracy Metrics =================
print("\n### Accuracy Metrics ###")
tolerance = 20
is_accurate_tolerance = np.abs(predictions - y_test) <= tolerance
accuracy_tolerance = np.mean(is_accurate_tolerance) * 100
print(f"✅ Accuracy within ±{tolerance} point tolerance: {accuracy_tolerance:.2f}%")

true_categories = aqi_to_category(y_test)
predicted_categories = aqi_to_category(predictions)
accuracy_category = accuracy_score(true_categories, predicted_categories) * 100
print(f"✅ Accuracy based on AQI categories: {accuracy_category:.2f}%")

# 6. Save Model
joblib.dump(model, 'aqi_lasso_model.joblib')
print("💾 Model saved as 'aqi_lasso_model.joblib'")
