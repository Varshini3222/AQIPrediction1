import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
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

    # Scale features
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


print("🚀 DL COMBO 2: (FNN + SlideNN) → GRU as Meta")
print("=" * 70)

# Step 1: Data Preparation
X_train, X_test, y_train, y_test, scaler = essential_preprocessing()
X_train_gru = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_gru = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"✅ Data ready: {X_train.shape[0]} training samples")

# Step 2: Train Base Models (FNN and SlideNN)
print("🤖 Training base models (FNN + SlideNN)...")
fnn_base = build_fnn_model(X_train.shape[1], "base")
slidenn_base = build_slide_nn_model(X_train.shape[1])

print("Training FNN base model...")
fnn_base.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
print("Training SlideNN base model...")
slidenn_base.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
print("✅ Base models trained successfully!")

# Step 3: Get Base Predictions
print("Generating base predictions...")
fnn_pred_train = fnn_base.predict(X_train, verbose=0).flatten()
slidenn_pred_train = slidenn_base.predict(X_train, verbose=0).flatten()
fnn_pred_test = fnn_base.predict(X_test, verbose=0).flatten()
slidenn_pred_test = slidenn_base.predict(X_test, verbose=0).flatten()

# Step 4: Create Meta Features for GRU
# Reshape predictions as sequences for GRU (2 timesteps, 1 feature each)
meta_features_train_gru = np.column_stack([fnn_pred_train, slidenn_pred_train]).reshape(-1, 2, 1)
meta_features_test_gru = np.column_stack([fnn_pred_test, slidenn_pred_test]).reshape(-1, 2, 1)

print(f"Meta features shape for GRU: {meta_features_train_gru.shape}")

# Step 5: Train GRU Meta Model
print("🎯 Training GRU Meta Model...")
gru_meta = build_gru_model((meta_features_train_gru.shape[1], meta_features_train_gru.shape[2]), "meta")
history = gru_meta.fit(meta_features_train_gru, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

# Step 6: Final Prediction
print("Making final predictions...")
final_pred = gru_meta.predict(meta_features_test_gru, verbose=0).flatten()

# Step 7: Evaluate All Metrics
combo2_metrics = calculate_all_metrics(y_test, final_pred, "Combo 2: (FNN+SlideNN)→GRU")

# Compare with individual models
print("\n🔍 PERFORMANCE COMPARISON:")
print("=" * 60)
print(f"{'Model':<30} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Cat Acc':<12}")
print("-" * 75)

# Individual model metrics
fnn_metrics = calculate_all_metrics(y_test, fnn_pred_test, "FNN Alone")
slidenn_metrics = calculate_all_metrics(y_test, slidenn_pred_test, "SlideNN Alone")

# GRU alone for comparison
print("Training GRU alone for comparison...")
gru_alone = build_gru_model((X_train_gru.shape[1], X_train_gru.shape[2]))
gru_alone.fit(X_train_gru, y_train, epochs=50, batch_size=32, verbose=0)
gru_pred_alone = gru_alone.predict(X_test_gru, verbose=0).flatten()
gru_metrics = calculate_all_metrics(y_test, gru_pred_alone, "GRU Alone")

models_comparison = {
    'FNN Alone': fnn_metrics,
    'SlideNN Alone': slidenn_metrics,
    'GRU Alone': gru_metrics,
    '(FNN+SlideNN)→GRU Meta': combo2_metrics
}

for name, metrics in models_comparison.items():
    print(
        f"{name:<30} {metrics['r2']:.4f}    {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} {metrics['category_accuracy']:<8.2f}%")

# Step 8: Save models
joblib.dump({
    'base_models': {'fnn': fnn_base, 'slidenn': slidenn_base},
    'meta_model': gru_meta,
    'combo_type': 'FNN_SlideNN_GRU_Meta',
    'metrics': combo2_metrics,
    'scaler': scaler
}, 'dl_combo2_fnn_slidenn_gru_meta.joblib')

print("\n💾 Combo 2 saved as 'dl_combo2_fnn_slidenn_gru_meta.joblib'")
print("✅ GRU AS META COMPLETED SUCCESSFULLY!")