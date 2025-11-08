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


print("🚀 DL COMBO 3: (GRU + SlideNN) → FNN as Meta")
print("=" * 70)

# Step 1: Data Preparation
X_train, X_test, y_train, y_test, scaler = essential_preprocessing()
X_train_gru = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_gru = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"✅ Data ready: {X_train.shape[0]} training samples")

# Step 2: Train Base Models (GRU and SlideNN)
print("🤖 Training base models (GRU + SlideNN)...")
gru_base = build_gru_model((X_train_gru.shape[1], X_train_gru.shape[2]), "base")
slidenn_base = build_slide_nn_model(X_train.shape[1])

print("Training GRU base model...")
gru_base.fit(X_train_gru, y_train, epochs=25, batch_size=32, verbose=0)
print("Training SlideNN base model...")
slidenn_base.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
print("✅ Base models trained successfully!")

# Step 3: Get Base Predictions
print("Generating base predictions...")
gru_pred_train = gru_base.predict(X_train_gru, verbose=0).flatten()
slidenn_pred_train = slidenn_base.predict(X_train, verbose=0).flatten()
gru_pred_test = gru_base.predict(X_test_gru, verbose=0).flatten()
slidenn_pred_test = slidenn_base.predict(X_test, verbose=0).flatten()

# Step 4: Create Meta Features for FNN
meta_features_train_fnn = np.column_stack([gru_pred_train, slidenn_pred_train])
meta_features_test_fnn = np.column_stack([gru_pred_test, slidenn_pred_test])

print(f"Meta features shape for FNN: {meta_features_train_fnn.shape}")

# Step 5: Train FNN Meta Model
print("🎯 Training FNN Meta Model...")
fnn_meta = build_fnn_model(meta_features_train_fnn.shape[1], "meta")
history = fnn_meta.fit(meta_features_train_fnn, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

# Step 6: Final Prediction
print("Making final predictions...")
final_pred = fnn_meta.predict(meta_features_test_fnn, verbose=0).flatten()

# Step 7: Evaluate All Metrics
combo3_metrics = calculate_all_metrics(y_test, final_pred, "Combo 3: (GRU+SlideNN)→FNN")

# Compare with individual models
print("\n🔍 PERFORMANCE COMPARISON:")
print("=" * 60)
print(f"{'Model':<30} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Cat Acc':<12}")
print("-" * 75)

# Individual model metrics
gru_metrics = calculate_all_metrics(y_test, gru_pred_test, "GRU Alone")
slidenn_metrics = calculate_all_metrics(y_test, slidenn_pred_test, "SlideNN Alone")

# FNN alone for comparison
print("Training FNN alone for comparison...")
fnn_alone = build_fnn_model(X_train.shape[1])
fnn_alone.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
fnn_pred_alone = fnn_alone.predict(X_test, verbose=0).flatten()
fnn_metrics = calculate_all_metrics(y_test, fnn_pred_alone, "FNN Alone")

models_comparison = {
    'GRU Alone': gru_metrics,
    'SlideNN Alone': slidenn_metrics,
    'FNN Alone': fnn_metrics,
    '(GRU+SlideNN)→FNN Meta': combo3_metrics
}

for name, metrics in models_comparison.items():
    print(
        f"{name:<30} {metrics['r2']:.4f}    {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} {metrics['category_accuracy']:<8.2f}%")

# Step 8: Save models
joblib.dump({
    'base_models': {'gru': gru_base, 'slidenn': slidenn_base},
    'meta_model': fnn_meta,
    'combo_type': 'GRU_SlideNN_FNN_Meta',
    'metrics': combo3_metrics,
    'scaler': scaler
}, 'dl_combo3_gru_slidenn_fnn_meta.joblib')

print("\n💾 Combo 3 saved as 'dl_combo3_gru_slidenn_fnn_meta.joblib'")
print("✅ FNN AS META COMPLETED SUCCESSFULLY!")