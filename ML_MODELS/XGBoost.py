import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report
)
from sklearn.preprocessing import LabelEncoder

print("--- XGBoost AQI Prediction ---")

# Load dataset
df = pd.read_csv("AQI_complete_imputed_2014_2025.csv")

# Select features and target
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'
df = df[features + [target]].dropna()  # remove missing values

# Function to map AQI into categories
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

# ================= Regression =================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = XGBRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

print("\n### Regression Metrics ###")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Accuracy within tolerance
tolerance = 20
acc_tolerance = np.mean(np.abs(y_pred - y_test) <= tolerance) * 100
print(f"Accuracy within ±{tolerance}: {acc_tolerance:.2f}%")

# Category accuracy from regression outputs
true_categories = aqi_to_category(y_test)
pred_categories = aqi_to_category(y_pred)
print(f"Category Accuracy (from regression predictions): {accuracy_score(true_categories, pred_categories)*100:.2f}%")

# ================= Classification =================
# Convert AQI to categories
y_class = aqi_to_category(y)

# Encode categories to numbers
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42
)

clf_model = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)

print("\n### Classification Metrics ###")
print(f"Category Accuracy: {accuracy_score(y_test_c, y_pred_c)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=le.classes_))
