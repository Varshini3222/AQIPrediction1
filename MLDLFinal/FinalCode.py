import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.optimizers import Adam
import joblib


def essential_preprocessing():
    """Only essential preprocessing"""
    df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    target = 'AQI'

    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target]

    # Scale features for DL models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


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


# Model Building Functions
def build_fnn_model(input_dim, name_suffix=""):
    """Feed Forward Neural Network"""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, name=f'fnn_output_{name_suffix}'))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_gru_model(input_shape, name_suffix=""):
    """Gated Recurrent Unit for sequential data"""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, name=f'gru_output_{name_suffix}'))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_slide_nn_model(input_dim, name_suffix=""):
    """Slide Neural Network with multiple dense layers"""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, name=f'slidenn_output_{name_suffix}'))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


print("🚀  HYBRID MODEL: (FNN+GRU→SlideNN) + (RF+DT→XGB) → FINAL META")
print("=" * 80)

# Step 1: Data Preparation
X_train, X_test, y_train, y_test, scaler = essential_preprocessing()
X_train_gru = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_gru = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# For ML models, use original data (no scaling needed for tree-based models)
df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'
df_clean = df[features + [target]].dropna()
X_ml = df_clean[features]
y_ml = df_clean[target]
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

print(f"✅ Data ready: {X_train.shape[0]} training samples")
print(f"✅ Test Data ready: {X_test.shape[0]} testing samples")

# ============================================================================
# STEP 2: TRAIN DL HYBRID (FNN + GRU → SlideNN)
# ============================================================================
print("\n" + "=" * 50)
print("🤖 TRAINING DL HYBRID: (FNN + GRU) → SlideNN")
print("=" * 50)

# Train DL Base Models
print("Training DL base models...")
fnn_base = build_fnn_model(X_train.shape[1], "base")
gru_base = build_gru_model((X_train_gru.shape[1], X_train_gru.shape[2]), "base")

fnn_base.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
gru_base.fit(X_train_gru, y_train, epochs=25, batch_size=32, verbose=0)

# Get DL Base Predictions
fnn_pred_train = fnn_base.predict(X_train, verbose=0).flatten()
gru_pred_train = gru_base.predict(X_train_gru, verbose=0).flatten()
fnn_pred_test = fnn_base.predict(X_test, verbose=0).flatten()
gru_pred_test = gru_base.predict(X_test_gru, verbose=0).flatten()

# Create DL Meta Features
dl_meta_features_train = np.column_stack([fnn_pred_train, gru_pred_train])
dl_meta_features_test = np.column_stack([fnn_pred_test, gru_pred_test])

# Train DL Meta Model
print("Training SlideNN Meta Model...")
slidenn_meta = build_slide_nn_model(dl_meta_features_train.shape[1], "meta")
slidenn_meta.fit(dl_meta_features_train, y_train, epochs=30, batch_size=32, verbose=0)

# Get DL Final Predictions
dl_final_pred_train = slidenn_meta.predict(dl_meta_features_train, verbose=0).flatten()
dl_final_pred_test = slidenn_meta.predict(dl_meta_features_test, verbose=0).flatten()

print("✅ DL Hybrid trained successfully!")

# ============================================================================
# STEP 3: TRAIN ML HYBRID (RF + DT → XGBoost)
# ============================================================================
print("\n" + "=" * 50)
print("🤖 TRAINING ML HYBRID: (RF + DT) → XGBoost")
print("=" * 50)

# Train ML Base Models
print("Training ML base models...")
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_ml, y_train_ml)
dt = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train_ml, y_train_ml)

# Get ML Base Predictions
rf_pred_train = rf.predict(X_train_ml)
dt_pred_train = dt.predict(X_train_ml)
rf_pred_test = rf.predict(X_test_ml)
dt_pred_test = dt.predict(X_test_ml)

# Create ML Meta Features
ml_meta_features_train = np.column_stack([rf_pred_train, dt_pred_train])
ml_meta_features_test = np.column_stack([rf_pred_test, dt_pred_test])

# Train ML Meta Model
print("Training XGBoost Meta Model...")
xgb_meta = XGBRegressor(n_estimators=100, random_state=42)
xgb_meta.fit(ml_meta_features_train, y_train_ml)

# Get ML Final Predictions
ml_final_pred_train = xgb_meta.predict(ml_meta_features_train)
ml_final_pred_test = xgb_meta.predict(ml_meta_features_test)

print("✅ ML Hybrid trained successfully!")

# ============================================================================
# STEP 4: CREATE SUPER HYBRID (DL + ML → Final Meta)
# ============================================================================
print("\n" + "=" * 50)
print("🎯 CREATING HYBRID: DL + ML → Final Meta")
print("=" * 50)

# Create Super Meta Features (combine DL and ML predictions)
super_meta_features_train = np.column_stack([dl_final_pred_train, ml_final_pred_train])
super_meta_features_test = np.column_stack([dl_final_pred_test, ml_final_pred_test])

print(f"Meta features shape: {super_meta_features_train.shape}")

# Train Final Super Meta Model (Linear Regression for optimal weighting)
print("Training Final Meta Model...")
super_meta = LinearRegression()
super_meta.fit(super_meta_features_train, y_train)

# Get Super Final Predictions
super_final_pred = super_meta.predict(super_meta_features_test)

# ============================================================================
# STEP 5: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n" + "=" * 50)
print("📊 COMPREHENSIVE PERFORMANCE EVALUATION")
print("=" * 50)

# Evaluate Super Hybrid
super_metrics = calculate_all_metrics(y_test, super_final_pred, "HYBRID: DL+ML→Final")

# Evaluate Individual Hybrids
dl_hybrid_metrics = calculate_all_metrics(y_test, dl_final_pred_test, "DL Hybrid: FNN+GRU→SlideNN")
ml_hybrid_metrics = calculate_all_metrics(y_test_ml, ml_final_pred_test, "ML Hybrid: RF+DT→XGB")

# Evaluate Individual Models
fnn_metrics = calculate_all_metrics(y_test, fnn_pred_test, "FNN Alone")
gru_metrics = calculate_all_metrics(y_test, gru_pred_test, "GRU Alone")
rf_metrics = calculate_all_metrics(y_test_ml, rf_pred_test, "Random Forest Alone")
dt_metrics = calculate_all_metrics(y_test_ml, dt_pred_test, "Decision Tree Alone")

# Evaluate SlideNN and XGBoost alone
slidenn_alone = build_slide_nn_model(X_train.shape[1])
slidenn_alone.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
slidenn_pred = slidenn_alone.predict(X_test, verbose=0).flatten()
slidenn_metrics = calculate_all_metrics(y_test, slidenn_pred, "SlideNN Alone")

xgb_alone = XGBRegressor(n_estimators=100, random_state=42).fit(X_train_ml, y_train_ml)
xgb_pred_alone = xgb_alone.predict(X_test_ml)
xgb_metrics = calculate_all_metrics(y_test_ml, xgb_pred_alone, "XGBoost Alone")

# ============================================================================
# STEP 6: PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("🔍 PERFORMANCE COMPARISON")
print("=" * 80)
print(f"{'MODEL':<40} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Cat Acc':<12}")
print("-" * 90)

models_comparison = {
    'INDIVIDUAL MODELS': '',
    '  FNN Alone': fnn_metrics,
    '  GRU Alone': gru_metrics,
    '  Random Forest Alone': rf_metrics,
    '  Decision Tree Alone': dt_metrics,
    '  SlideNN Alone': slidenn_metrics,
    '  XGBoost Alone': xgb_metrics,
    'HYBRID MODELS': '',
    '  DL Hybrid (FNN+GRU→SlideNN)': dl_hybrid_metrics,
    '  ML Hybrid (RF+DT→XGB)': ml_hybrid_metrics,
    'HYBRID': '',
    '  🚀 HYBRID (DL+ML→Final)': super_metrics
}

for name, metrics in models_comparison.items():
    if metrics == '':
        print(f"\n{name}")
        print("-" * 40)
    else:
        print(
            f"{name:<40} {metrics['r2']:.4f}    {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} {metrics['category_accuracy']:<8.2f}%")

# ============================================================================
# STEP 7: SAVE COMPLETE SUPER HYBRID MODEL
# ============================================================================
print("\n" + "=" * 50)
print("💾 SAVING HYBRID MODEL")
print("=" * 50)

# Save complete model package
joblib.dump({
    'dl_hybrid': {
        'base_models': {'fnn': fnn_base, 'gru': gru_base},
        'meta_model': slidenn_meta,
        'scaler': scaler
    },
    'ml_hybrid': {
        'base_models': {'rf': rf, 'dt': dt},
        'meta_model': xgb_meta
    },
    'super_meta': super_meta,
    'super_meta_coefficients': {
        'dl_weight': super_meta.coef_[0],
        'ml_weight': super_meta.coef_[1],
        'intercept': super_meta.intercept_
    },
    'performance_metrics': {
        'super_hybrid': super_metrics,
        'dl_hybrid': dl_hybrid_metrics,
        'ml_hybrid': ml_hybrid_metrics
    },
    'combo_type': 'SUPER_HYBRID_DL_ML_META'
}, 'super_hybrid_dl_ml_meta.joblib')

print("✅ Super Hybrid saved as 'super_hybrid_dl_ml_meta.joblib'")

# ============================================================================
# STEP 8: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎯 HYBRID MODEL SUMMARY")
print("=" * 80)
print("ARCHITECTURE: (FNN + GRU → SlideNN) + (RF + DT → XGBoost) → Linear Meta")
print("\nCOMPONENTS:")
print("  • DL Branch: FNN (spatial) + GRU (temporal) → SlideNN (hierarchical)")
print("  • ML Branch: RF (ensemble) + DT (rules) → XGBoost (boosting)")
print("  • Final Meta: Linear combination with optimal weights")
print(f"\nOPTIMAL WEIGHTS:")
print(f"  • DL Hybrid weight: {super_meta.coef_[0]:.3f}")
print(f"  • ML Hybrid weight: {super_meta.coef_[1]:.3f}")
print(f"  • Intercept: {super_meta.intercept_:.3f}")

# Performance improvement calculation
super_r2 = super_metrics['r2']
dl_r2 = dl_hybrid_metrics['r2']
ml_r2 = ml_hybrid_metrics['r2']
best_individual = max(dl_r2, ml_r2)
improvement = ((super_r2 - best_individual) / best_individual) * 100

print(f"\nPERFORMANCE IMPROVEMENT:")
print(f"  • Best Individual Hybrid: R² = {best_individual:.4f}")
print(f"  • Super Hybrid: R² = {super_r2:.4f}")
print(f"  • Improvement: +{improvement:.2f}% 🚀")

print("\n✅HYBRID MODEL COMPLETED SUCCESSFULLY!")
print("=" * 80)