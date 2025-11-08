import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import joblib

print("--- Decision Tree Regressor (New Dataset: 2014–2025) ---")

# 1. Load the new dataset
try:
    df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'AQI_complete_imputed_2014_2025.csv' not found.")
    print("Please ensure the file is in the same directory.")
    exit()

# 2. Prepare Data for Training
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

# Keep only required columns and drop missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🔍 Data prepared. Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 3. Train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# 4. Predictions
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

# Accuracy within a tolerance threshold
tolerance = 20
is_accurate_tolerance = np.abs(predictions - y_test) <= tolerance
accuracy_tolerance = np.mean(is_accurate_tolerance) * 100
print(f"✅ Accuracy within ±{tolerance} point tolerance: {accuracy_tolerance:.2f}%")

# Accuracy by AQI Category
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

true_categories = aqi_to_category(y_test)
predicted_categories = aqi_to_category(predictions)
accuracy_category = accuracy_score(true_categories, predicted_categories) * 100
print(f"✅ Accuracy based on AQI categories: {accuracy_category:.2f}%")

# 5. Save the trained model
joblib.dump(model, 'aqi_decision_tree_model.joblib')
print("\n💾 Model saved as 'aqi_decision_tree_model.joblib'")

print("\n✅ Process complete! Decision Tree AQI model is ready to use.")
