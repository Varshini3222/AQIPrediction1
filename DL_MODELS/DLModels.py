# =====================================================
# Air Quality Index Prediction - Deep Learning Models
# Models: FNN, LSTM, GRU, BiLSTM, CNN, SlideNN, LSTM+Attention
# Dataset: AQI 2015–2025
# =====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, Flatten, Input, Attention
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
print("📂 Loading dataset (2014–2025)...")
try:
    df = pd.read_csv("AQI_complete_imputed_2014_2025.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Dataset not found. Please check the path.")
    exit()

features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

df = df[features + [target]].dropna()

X = df[features].values
y = df[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Helper Functions
# -----------------------------
def aqi_to_category(aqi_values):
    """Convert AQI values into categorical labels."""
    categories = []
    for aqi in aqi_values:
        if aqi <= 50: categories.append("Good")
        elif aqi <= 100: categories.append("Satisfactory")
        elif aqi <= 200: categories.append("Moderate")
        elif aqi <= 300: categories.append("Poor")
        elif aqi <= 400: categories.append("Very Poor")
        else: categories.append("Severe")
    return categories

def evaluate_model(model, X_test, y_test):
    """Evaluate model with regression + AQI category accuracy."""
    predictions = model.predict(X_test).flatten()
    print("\n### Regression Metrics ###")
    print(f"R²: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")

    true_categories = aqi_to_category(y_test)
    predicted_categories = aqi_to_category(predictions)
    print(f"Accuracy (AQI categories): "
          f"{accuracy_score(true_categories, predicted_categories) * 100:.2f}%")
    return predictions

# =====================================================
# 1. Feedforward Neural Network (FNN)
# =====================================================
print("\n===== Feedforward Neural Network (FNN) =====")
fnn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
fnn.compile(optimizer='adam', loss='mse')
fnn.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(fnn, X_test, y_test)
fnn.save("aqi_fnn_model.h5")

# =====================================================
# 2. LSTM
# =====================================================
print("\n===== LSTM =====")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm = Sequential([
    Input(shape=(1, X_train.shape[1])),
    LSTM(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(lstm, X_test_lstm, y_test)
lstm.save("aqi_lstm_model.h5")

# =====================================================
# 3. GRU
# =====================================================
print("\n===== GRU =====")
gru = Sequential([
    Input(shape=(1, X_train.shape[1])),
    GRU(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(1)
])
gru.compile(optimizer='adam', loss='mse')
gru.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(gru, X_test_lstm, y_test)
gru.save("aqi_gru_model.h5")

# =====================================================
# 4. BiLSTM
# =====================================================
print("\n===== BiLSTM =====")
bilstm = Sequential([
    Input(shape=(1, X_train.shape[1])),
    Bidirectional(LSTM(64, activation='tanh')),
    Dense(32, activation='relu'),
    Dense(1)
])
bilstm.compile(optimizer='adam', loss='mse')
bilstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(bilstm, X_test_lstm, y_test)
bilstm.save("aqi_bilstm_model.h5")

# =====================================================
# 5. CNN
# =====================================================
print("\n===== CNN =====")
cnn = Sequential([
    Input(shape=(1, X_train.shape[1])),
    Conv1D(64, kernel_size=1, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
cnn.compile(optimizer='adam', loss='mse')
cnn.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(cnn, X_test_lstm, y_test)
cnn.save("aqi_cnn_model.h5")

# =====================================================
# 6. SlideNN (Variable-depth NN)
# =====================================================
print("\n===== SlideNN (Variable-depth NN) =====")
slidenn = Sequential([Input(shape=(X_train.shape[1],))])
for units in [128, 64, 32]:
    slidenn.add(Dense(units, activation='relu'))
slidenn.add(Dense(1))
slidenn.compile(optimizer='adam', loss='mse')
slidenn.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(slidenn, X_test, y_test)
slidenn.save("aqi_slidenn_model.h5")

# =====================================================
# 7. LSTM with Attention
# =====================================================
print("\n===== LSTM with Attention =====")
inputs = Input(shape=(1, X_train.shape[1]))
lstm_out = LSTM(64, return_sequences=True)(inputs)
attn_out = Attention()([lstm_out, lstm_out])
flat = tf.keras.layers.Flatten()(attn_out)
dense1 = Dense(32, activation='relu')(flat)
output = Dense(1)(dense1)

lstm_attn = Model(inputs, output)
lstm_attn.compile(optimizer='adam', loss='mse')
lstm_attn.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
evaluate_model(lstm_attn, X_test_lstm, y_test)
lstm_attn.save("aqi_lstm_attention_model.h5")

print("\n✅ All DL models trained and evaluated successfully!")
