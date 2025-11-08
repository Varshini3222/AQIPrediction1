# FNN_CNN.py - FNN + CNN Hybrid Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, Average
from tensorflow.keras.optimizers import Adam
import joblib

print("=" * 70)
print("DEEP LEARNING HYBRID: FNN + CNN")
print("=" * 70)

# Load data (same as above)
df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

df = df[features + [target]].dropna()
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


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


# Create FNN + CNN Hybrid
def create_fnn_cnn_hybrid():
    fnn_input = Input(shape=(X_train.shape[1],), name='fnn_input')
    fnn_branch = Dense(64, activation='relu')(fnn_input)
    fnn_branch = Dense(32, activation='relu')(fnn_branch)
    fnn_output = Dense(1, name='fnn_output')(fnn_branch)

    cnn_input = Input(shape=(1, X_train.shape[1]), name='cnn_input')
    cnn_branch = Conv1D(64, kernel_size=1, activation='relu')(cnn_input)
    cnn_branch = Flatten()(cnn_branch)
    cnn_branch = Dense(32, activation='relu')(cnn_branch)
    cnn_output = Dense(1, name='cnn_output')(cnn_branch)

    combined = Average()([fnn_output, cnn_output])

    model = Model(inputs=[fnn_input, cnn_input], outputs=combined)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Build and train model
model = create_fnn_cnn_hybrid()
print("🔄 Training FNN+CNN Hybrid...")
history = model.fit(
    [X_train, X_train_seq], y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
predictions = model.predict([X_test, X_test_seq]).flatten()

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

tolerance = 20
accuracy_tolerance = np.mean(np.abs(predictions - y_test) <= tolerance) * 100
true_categories = aqi_to_category(y_test)
predicted_categories = aqi_to_category(predictions)
accuracy_category = accuracy_score(true_categories, predicted_categories) * 100

print(f"\n📊 PERFORMANCE METRICS:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Accuracy (±{tolerance} points): {accuracy_tolerance:.2f}%")
print(f"Category Accuracy: {accuracy_category:.2f}%")

# Confusion Matrix
categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
cm = confusion_matrix(true_categories, predicted_categories, labels=categories)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - FNN+CNN Hybrid\nAccuracy: {:.2f}%'.format(accuracy_category))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_fnn_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
model.save('hybrid_fnn_cnn_model.h5')
joblib.dump({'history': history.history, 'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse}},
            'hybrid_fnn_cnn_info.joblib')

print("\n" + "=" * 70)
print("✅ FNN+CNN Hybrid Model Completed!")