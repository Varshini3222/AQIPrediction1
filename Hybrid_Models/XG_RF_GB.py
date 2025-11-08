# hybrid_ml_model_with_confusion_matrix.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                           accuracy_score, confusion_matrix, classification_report)
import joblib

print("--- Hybrid ML Model (XGBoost + Random Forest + Gradient Boosting) ---")

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

# XGBoost
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# 5. Hybrid Model (Weighted Average)
print("\n--- Hybrid Model (Weighted Average) ---")

# Calculate weights based on individual model R² scores
xgb_r2 = r2_score(y_test, xgb_pred)
rf_r2 = r2_score(y_test, rf_pred)
gb_r2 = r2_score(y_test, gb_pred)

total_r2 = xgb_r2 + rf_r2 + gb_r2
xgb_weight = xgb_r2 / total_r2
rf_weight = rf_r2 / total_r2
gb_weight = gb_r2 / total_r2

print(f"XGBoost Weight: {xgb_weight:.4f} (R²: {xgb_r2:.4f})")
print(f"Random Forest Weight: {rf_weight:.4f} (R²: {rf_r2:.4f})")
print(f"Gradient Boosting Weight: {gb_weight:.4f} (R²: {gb_r2:.4f})")

# Hybrid prediction
hybrid_pred = (xgb_pred * xgb_weight) + (rf_pred * rf_weight) + (gb_pred * gb_weight)

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
    'XGBoost': xgb_pred,
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'Hybrid Model': hybrid_pred
}

for name, pred in models.items():
    r2_val = r2_score(y_test, pred)
    category_acc = accuracy_score(aqi_to_category(y_test), aqi_to_category(pred)) * 100
    print(f"{name:20} | R²: {r2_val:.4f} | Category Acc: {category_acc:.2f}%")

# 8. CONFUSION MATRIX ANALYSIS
print("\n" + "="*60)
print("CONFUSION MATRIX ANALYSIS")
print("="*60)

# Get all possible categories for consistent ordering
categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Create confusion matrix
cm = confusion_matrix(true_categories, predicted_categories, labels=categories)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(true_categories, predicted_categories, labels=categories))

# 9. VISUALIZE CONFUSION MATRIX
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.title(f'Confusion Matrix - XG+RF+GB\nAccuracy: {accuracy_category:.2f}%')
plt.xlabel('Predicted AQI Category')
plt.ylabel('Actual AQI Category')
plt.tight_layout()
plt.savefig('hybrid_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. DETAILED CATEGORY ANALYSIS
print("\n" + "="*60)
print("DETAILED CATEGORY PERFORMANCE ANALYSIS")
print("="*60)

# Calculate precision, recall, F1-score for each category
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    true_categories, predicted_categories, labels=categories, zero_division=0
)

category_performance = pd.DataFrame({
    'Category': categories,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support  # Number of actual occurrences
})

print(category_performance.to_string(index=False))

# 11. Save Hybrid Model
hybrid_model = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'gb_model': gb_model,
    'weights': {'xgb': xgb_weight, 'rf': rf_weight, 'gb': gb_weight}
}

joblib.dump(hybrid_model, 'hybrid_ml_model.joblib')
print("\n💾 Hybrid model saved as 'hybrid_ml_model.joblib'")

print("\n✅ Hybrid model training and evaluation complete!")