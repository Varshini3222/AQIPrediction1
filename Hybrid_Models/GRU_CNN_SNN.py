# hybrid_cnn_gru_slidenn_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D, GRU, LSTM,
                                     Flatten, Reshape, Input, concatenate,
                                     BatchNormalization, Dropout, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

print("--- Hybrid Model (CNN + GRU + slideNN) ---")
print("=== WITH CONFUSION MATRIX ANALYSIS ===\n")

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


# 3. Prepare Data
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

df = df[features + [target]].dropna()
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN and GRU (add time step dimension)
# Since we have only 6 features, we'll use them as a sequence of length 6
X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_3d = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print(f"Data shape - Original: {X_train_scaled.shape}, 3D: {X_train_3d.shape}")

# 4. Build Individual Deep Learning Models (FIXED ARCHITECTURES)
print("\n--- Building Individual DL Models ---")

# FIXED CNN Model - Adjusted for small sequence length (6)
print("Building CNN Model...")


def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),  # Reduced pool size to avoid dimension issues
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),  # Use global pooling instead of MaxPooling to avoid dimension issues
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


cnn_model = build_cnn_model((X_train_3d.shape[1], 1))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train with smaller batch size to avoid memory issues
cnn_history = cnn_model.fit(X_train_3d, y_train, epochs=30, batch_size=64,
                            verbose=0, validation_split=0.1, callbacks=[early_stop])
cnn_pred = cnn_model.predict(X_test_3d).flatten()

# FIXED GRU Model
print("Building GRU Model...")


def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        GRU(32, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


gru_model = build_gru_model((X_train_3d.shape[1], 1))
gru_history = gru_model.fit(X_train_3d, y_train, epochs=30, batch_size=64,
                            verbose=0, validation_split=0.1, callbacks=[early_stop])
gru_pred = gru_model.predict(X_test_3d).flatten()

# FIXED slideNN Model (Simplified for stability)
print("Building slideNN Model...")


def build_slidenn_model(input_dim):
    inputs = Input(shape=(input_dim,))

    # Simplified pathways to avoid overfitting
    pathway1 = Dense(32, activation='relu')(inputs)
    pathway1 = BatchNormalization()(pathway1)

    pathway2 = Dense(64, activation='relu')(inputs)
    pathway2 = BatchNormalization()(pathway2)
    pathway2 = Dense(32, activation='relu')(pathway2)

    # Concatenate pathways
    concatenated = concatenate([pathway1, pathway2])
    concatenated = Dropout(0.3)(concatenated)

    # Final layers
    x = Dense(64, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


slidenn_model = build_slidenn_model(X_train_scaled.shape[1])
slidenn_history = slidenn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=64,
                                    verbose=0, validation_split=0.1, callbacks=[early_stop])
slidenn_pred = slidenn_model.predict(X_test_scaled).flatten()

# 5. Hybrid Model (Weighted Average)
print("\n--- Hybrid Model (Weighted Average) ---")

# Calculate weights based on individual model R² scores
cnn_r2 = max(0.01, r2_score(y_test, cnn_pred))
gru_r2 = max(0.01, r2_score(y_test, gru_pred))
slidenn_r2 = max(0.01, r2_score(y_test, slidenn_pred))

total_r2 = cnn_r2 + gru_r2 + slidenn_r2
cnn_weight = cnn_r2 / total_r2
gru_weight = gru_r2 / total_r2
slidenn_weight = slidenn_r2 / total_r2

print(f"CNN Weight: {cnn_weight:.4f} (R²: {cnn_r2:.4f})")
print(f"GRU Weight: {gru_weight:.4f} (R²: {gru_r2:.4f})")
print(f"slideNN Weight: {slidenn_weight:.4f} (R²: {slidenn_r2:.4f})")

# Hybrid prediction
hybrid_pred = (cnn_pred * cnn_weight) + (gru_pred * gru_weight) + (slidenn_pred * slidenn_weight)

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
    'CNN': cnn_pred,
    'GRU': gru_pred,
    'slideNN': slidenn_pred,
    'Hybrid Model': hybrid_pred
}

for name, pred in models.items():
    r2_val = r2_score(y_test, pred)
    category_acc = accuracy_score(aqi_to_category(y_test), aqi_to_category(pred)) * 100
    print(f"{name:15} | R²: {r2_val:.4f} | Category Acc: {category_acc:.2f}%")

# 8. VISUAL CONFUSION MATRIX
print("\n" + "=" * 60)
print("VISUAL CONFUSION MATRIX")
print("=" * 60)

categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
cm = confusion_matrix(true_categories, predicted_categories, labels=categories)

# Create visual confusion matrix
plt.figure(figsize=(12, 10))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=categories, yticklabels=categories,
                 cbar_kws={'label': 'Number of Predictions'})

plt.title('Confusion Matrix - Hybrid CNN+GRU+slideNN Model\n', fontsize=16, fontweight='bold')
plt.xlabel('Predicted AQI Category', fontsize=12, fontweight='bold')
plt.ylabel('Actual AQI Category', fontsize=12, fontweight='bold')
plt.suptitle(f'Overall Accuracy: {accuracy_category:.2f}%', y=0.92, fontsize=14, color='green')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('cnn_gru_slidenn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Save Models
print("\n💾 Saving models...")

# Save individual models
cnn_model.save('cnn_model_fixed.h5')
gru_model.save('gru_model_fixed.h5')
slidenn_model.save('slidenn_model_fixed.h5')

# Save hybrid configuration
hybrid_config = {
    'models': {
        'cnn': 'cnn_model_fixed.h5',
        'gru': 'gru_model_fixed.h5',
        'slidenn': 'slidenn_model_fixed.h5'
    },
    'weights': {
        'cnn': float(cnn_weight),
        'gru': float(gru_weight),
        'slidenn': float(slidenn_weight)
    },
    'scaler': scaler,
    'input_shapes': {
        '3d': X_train_3d.shape[1:],
        '2d': X_train_scaled.shape[1]
    }
}

joblib.dump(hybrid_config, 'hybrid_cnn_gru_slidenn_config.joblib')

print("✅ Models saved successfully!")
print("   - cnn_model_fixed.h5")
print("   - gru_model_fixed.h5")
print("   - slidenn_model_fixed.h5")
print("   - hybrid_cnn_gru_slidenn_config.joblib")

print("\n✅ Hybrid CNN+GRU+slideNN model analysis complete!")