# Hybrid_Combo3_RF_DT_with_XGB_Meta.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib


def essential_preprocessing():
    """Only essential preprocessing"""
    df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    target = 'AQI'

    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def aqi_to_category(aqi_values):
    categories = []
    for aqi in aqi_values:
        if aqi <= 50:
            categories.append("Good")
        elif aqi <= 100:
            categories.append("Satisfactory")
        elif aqi <= 200:
            categories.append("Moderate")
        elif aqi <= 300:
            categories.append("Poor")
        elif aqi <= 400:
            categories.append("Very Poor")
        else:
            categories.append("Severe")
    return categories


def calculate_all_metrics(y_true, y_pred, model_name=""):
    """Calculate all required metrics"""
    # Regression metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Accuracy metrics
    tolerance = 20
    accuracy_tolerance = np.mean(np.abs(y_pred - y_true) <= tolerance) * 100

    # Category accuracy
    true_categories = aqi_to_category(y_true)
    pred_categories = aqi_to_category(y_pred)
    category_accuracy = accuracy_score(true_categories, pred_categories) * 100

    # Print results
    print(f"\n📊 {model_name.upper()} METRICS:")
    print(f"R² Score:          {r2:.4f}")
    print(f"MAE:               {mae:.2f} AQI points")
    print(f"RMSE:              {rmse:.2f} AQI points")
    print(f"Accuracy (±{tolerance}): {accuracy_tolerance:.2f}%")
    print(f"Category Accuracy: {category_accuracy:.2f}%")

    return {
        'r2': r2, 'mae': mae, 'rmse': rmse,
        'accuracy_tolerance': accuracy_tolerance,
        'category_accuracy': category_accuracy
    }


print("🚀 HYBRID COMBO 3: (Random Forest + Decision Tree) → XGBoost as Meta")
print("=" * 70)

# Step 1: Data Preparation
X_train, X_test, y_train, y_test = essential_preprocessing()
print(f"✅ Data ready: {X_train.shape[0]} training samples")

# Step 2: Train Base Models
print("🤖 Training base models...")
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
dt = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train, y_train)

# Step 3: Get Base Predictions
rf_pred_train = rf.predict(X_train)
dt_pred_train = dt.predict(X_train)
rf_pred_test = rf.predict(X_test)
dt_pred_test = dt.predict(X_test)

# Step 4: Create Meta Features (combined predictions)
meta_features_train = np.column_stack([rf_pred_train, dt_pred_train])
meta_features_test = np.column_stack([rf_pred_test, dt_pred_test])

print(f"Meta features shape: {meta_features_train.shape}")

# Step 5: Train Meta Model (XGBoost)
print("🎯 Training meta model (XGBoost)...")
meta_model = XGBRegressor(n_estimators=100, random_state=42)
meta_model.fit(meta_features_train, y_train)

# Step 6: Final Prediction
final_pred = meta_model.predict(meta_features_test)

# Step 7: Evaluate All Metrics
combo3_metrics = calculate_all_metrics(y_test, final_pred, "Hybrid Combo 3")

# Compare with individual models
print("\n🔍 PERFORMANCE COMPARISON:")
print("=" * 50)
print(f"{'Model':<25} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Cat Acc':<12}")
print("-" * 65)

# Individual model metrics
rf_metrics = calculate_all_metrics(y_test, rf_pred_test, "Random Forest Alone")
dt_metrics = calculate_all_metrics(y_test, dt_pred_test, "Decision Tree Alone")
xgb_alone = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb_pred_alone = xgb_alone.predict(X_test)
xgb_metrics = calculate_all_metrics(y_test, xgb_pred_alone, "XGBoost Alone")

# Print comparison table
models_comparison = {
    'Random Forest Alone': rf_metrics,
    'Decision Tree Alone': dt_metrics,
    'XGBoost Alone': xgb_metrics,
    'Hybrid Combo 3': combo3_metrics
}

for name, metrics in models_comparison.items():
    print(
        f"{name:<25} {metrics['r2']:.4f}    {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} {metrics['category_accuracy']:<8.2f}%")

# Step 8: Save
joblib.dump({
    'base_models': {'rf': rf, 'dt': dt},
    'meta_model': meta_model,
    'combo_type': 'RF_DT_with_XGB_Meta',
    'metrics': combo3_metrics
}, 'hybrid_combo3_rf_dt_xgb_meta.joblib')

print("\n💾 Combo 3 saved as 'hybrid_combo3_rf_dt_xgb_meta.joblib'")