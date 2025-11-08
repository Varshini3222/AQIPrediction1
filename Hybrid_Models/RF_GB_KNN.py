# hybrid_rf_gb_knn.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                           accuracy_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
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
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tolerance = 20
categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# hybrid_rf_gb_knn.py
print("="*70)
print("HYBRID MODEL: Random Forest + Gradient Boosting + KNN")
print("="*70)

# Train individual models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
knn_model = KNeighborsRegressor(n_neighbors=5)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Hybrid combination
rf_r2 = r2_score(y_test, rf_pred)
gb_r2 = r2_score(y_test, gb_pred)
knn_r2 = r2_score(y_test, knn_pred)

total_r2 = rf_r2 + gb_r2 + knn_r2
rf_weight = rf_r2 / total_r2
gb_weight = gb_r2 / total_r2
knn_weight = knn_r2 / total_r2

hybrid_pred = (rf_pred * rf_weight) + (gb_pred * gb_weight) + (knn_pred * knn_weight)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - RF+GB+KNN\nAccuracy: {:.2f}%'.format(accuracy_category))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_rf_gb_knn.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)

joblib.dump({'rf': rf_model, 'gb': gb_model, 'knn': knn_model,
            'weights': [rf_weight, gb_weight, knn_weight]},
           'hybrid_rf_gb_knn.joblib')