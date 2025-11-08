import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("--- Gaussian Naive Bayes Classifier (New Dataset: 2014–2025) ---")

# 1. Load the new dataset
try:
    df = pd.read_csv('AQI_complete_imputed_2014_2025.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'AQI_complete_imputed_2014_2025.csv' not found.")
    print("Please ensure the file is in the same directory.")
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

# 3. Prepare Data for Training
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI_Category'

# Keep only required columns and drop missing values
df = df[features + ['AQI']].dropna()
df[target] = aqi_to_category(df['AQI'])

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🔍 Data prepared. Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 4. Train Gaussian Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)
print("✅ Model training complete.")

# 5. Predictions
predictions = model.predict(X_test)

# ================= Classification Metrics =================
print("\n### Classification Metrics ###")
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

print(f"🎯 Accuracy: {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📐 Recall: {recall:.4f}")
print(f"💡 F1-Score: {f1:.4f}")

# 6. Save the trained GaussianNB model
joblib.dump(model, 'aqi_gaussian_nb_model.joblib')
print("\n💾 Model saved as 'aqi_gaussian_nb_model.joblib'")
print("\n✅ Process complete! Gaussian Naive Bayes AQI classifier is ready to use.")
