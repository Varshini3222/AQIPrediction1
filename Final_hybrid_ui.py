import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid Model AQI Prediction System",
    page_icon="🌬️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Place Your API Key Here ---
API_KEY = "45b1cfd6c2d653ba66075ddf07a16628"


# --- Load Super Hybrid Model ---
@st.cache_resource
def load_super_hybrid_model():
    """Loads the trained Super Hybrid model."""
    try:
        model_data = joblib.load('super_hybrid_dl_ml_meta.joblib')
        return model_data
    except FileNotFoundError:
        st.error("❌ Hybrid model file not found!")
        st.info("Please ensure 'super_hybrid_dl_ml_meta.joblib' is in the same folder.")
        return None


# --- Load Assets ---
@st.cache_data
def load_data():
    """Loads the final, imputed master air quality dataset."""
    try:
        df = pd.read_csv('AQI_complete_imputed_2014_2025.csv', parse_dates=['Date'])
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found!")
        return None


# Load data and model
model_data = load_super_hybrid_model()
df = load_data()

features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']


# --- Prediction Function for Super Hybrid Model ---
def make_super_hybrid_prediction(input_df, model_data):
    """Makes prediction using the Super Hybrid model."""
    try:
        # Scale input for DL models
        scaler = model_data['dl_hybrid']['scaler']
        X_scaled = scaler.transform(input_df)
        X_gru = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # DL Hybrid Prediction
        fnn_model = model_data['dl_hybrid']['base_models']['fnn']
        gru_model = model_data['dl_hybrid']['base_models']['gru']
        slidenn_meta = model_data['dl_hybrid']['meta_model']

        fnn_pred = fnn_model.predict(X_scaled, verbose=0).flatten()
        gru_pred = gru_model.predict(X_gru, verbose=0).flatten()
        dl_meta_features = np.column_stack([fnn_pred, gru_pred])
        dl_final_pred = slidenn_meta.predict(dl_meta_features, verbose=0).flatten()[0]

        # ML Hybrid Prediction
        rf_model = model_data['ml_hybrid']['base_models']['rf']
        dt_model = model_data['ml_hybrid']['base_models']['dt']
        xgb_meta = model_data['ml_hybrid']['meta_model']

        rf_pred = rf_model.predict(input_df)
        dt_pred = dt_model.predict(input_df)
        ml_meta_features = np.column_stack([rf_pred, dt_pred])
        ml_final_pred = xgb_meta.predict(ml_meta_features)[0]

        # Super Hybrid Final Prediction
        super_meta = model_data['super_meta']
        super_features = np.array([[dl_final_pred, ml_final_pred]])
        final_prediction = super_meta.predict(super_features)[0]

        # Get model weights for display
        dl_weight = model_data['super_meta_coefficients']['dl_weight']
        ml_weight = model_data['super_meta_coefficients']['ml_weight']

        return {
            'final_aqi': final_prediction,
            'dl_prediction': dl_final_pred,
            'ml_prediction': ml_final_pred,
            'dl_weight': dl_weight,
            'ml_weight': ml_weight
        }

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# --- API Functions ---
def get_live_data(city_name, api_key):
    """Fetches and processes live air pollution data for a given city."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={api_key}"
    try:
        geo_response = requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error during geocoding: {e}"}
    if geo_response.status_code == 401:
        return {"error": "Authentication failed. Your API Key is invalid or not yet active."}
    geo_data = geo_response.json()
    if not geo_data:
        return {"error": f"Could not find coordinates for '{city_name}'."}

    lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        pollution_response = requests.get(pollution_url, timeout=10)
        pollution_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error fetching pollution data: {e}"}
    return {"data": pollution_response.json()}


# --- Updated Helper Function for 20-point ranges ---
def get_health_advisory_20_range(aqi):
    if aqi is None or pd.isna(aqi):
        return "Not Measured 🤷", "No AQI data available."

    # Define 20-point ranges
    if aqi <= 20:
        return "Excellent (0-20) 💚", "✅ Perfect air quality! Ideal for all outdoor activities. Breathe easy!"
    elif aqi <= 40:
        return "Very Good (21-40) 💚", "✅ Very good air quality. Great for outdoor activities with minimal concerns."
    elif aqi <= 60:
        return "Good (41-60) 💚", "✅ Good air quality. Generally acceptable for most outdoor activities."
    elif aqi <= 80:
        return "Moderate (61-80) 💛", "⚠️ Moderate air quality. Usually safe, but unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif aqi <= 100:
        return "Fair (81-100) 💛", "⚠️ Fair air quality. Active children and adults, and people with respiratory disease should limit prolonged outdoor exertion."
    elif aqi <= 120:
        return "Unhealthy for Sensitive Groups (101-120) 🧡", "❌ Unhealthy for sensitive groups. People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion."
    elif aqi <= 140:
        return "Unhealthy (121-140) ❤️", "❌ Unhealthy air quality. Everyone may begin to experience health effects. Sensitive groups should avoid outdoor exposure."
    elif aqi <= 160:
        return "Very Unhealthy (141-160) 💜", "🆘 Very unhealthy air quality. Health alert: Everyone may experience more serious health effects."
    elif aqi <= 180:
        return "Hazardous (161-180) 🖤", "🚨 Hazardous conditions. Health warnings of emergency conditions. The entire population is likely affected."
    else:
        return "Severely Hazardous (181+) 💀", "💀 Severely hazardous conditions. Health warning of emergency conditions. Avoid all outdoor activities."


# --- Sidebar Navigation ---
st.sidebar.title("🚀  Hybrid Model AQI Predictor")
st.sidebar.markdown("Air Quality Index prediction using Deep Learning + Machine Learning ensemble.")

if model_data:
    st.sidebar.success("✅ Hybrid Model Loaded!")
    dl_weight = model_data['super_meta_coefficients']['dl_weight']
    ml_weight = model_data['super_meta_coefficients']['ml_weight']
    st.sidebar.info(f"**Model Weights:**\n- DL Hybrid: {dl_weight:.3f}\n- ML Hybrid: {ml_weight:.3f}")

page = st.sidebar.radio("Choose a Page", [
    "ℹ️ Hybrid Model Overview",
    "🛰️ Live AQI Prediction",
    "📊 Historical AQI Lookup",
    "📈 Future AQI Forecast",
    "⚙️ Manual Input Prediction"
])

# --- Page Implementations ---
if page == "ℹ️ Hybrid Model Overview":
    st.title("🚀 Hybrid AQI Prediction System")
    st.markdown("""
    ### Advanced AI-Powered Air Quality Prediction

    This system uses a **Super Hybrid Model** that combines:
    - **Deep Learning Branch**: FNN + GRU → SlideNN
    - **Machine Learning Branch**: Random Forest + Decision Tree → XGBoost
    - **Final Meta-Combiner**: Optimal weighting of both branches

    ### 🏗️ Architecture:
    ```
    RAW DATA
        │
        ├── DL BRANCH ────┐
        │ FNN → GRU → SlideNN │
        │                    │
        ├── ML BRANCH ────┼──→ SUPER META ───→ FINAL AQI 🎯
        │ RF → DT → XGBoost  │
        │                    │
        └───────────────────┘
    ```

    ### 📊 Performance:
    - **Individual Models**: R² ~ 0.85-0.88
    - **DL/ML Hybrids**: R² ~ 0.89-0.91  
    - **SUPER HYBRID**: R² ~ 0.92-0.94 🚀
    """)

    if model_data:
        metrics = model_data['performance_metrics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Super Hybrid R²", f"{metrics['super_hybrid']['r2']:.4f}")
        with col2:
            st.metric("DL Hybrid R²", f"{metrics['dl_hybrid']['r2']:.4f}")
        with col3:
            st.metric("ML Hybrid R²", f"{metrics['ml_hybrid']['r2']:.4f}")

elif page == "🛰️ Live AQI Prediction":
    st.header("🛰️ Live AQI Prediction with Super Hybrid")

    if df is not None:
        cities_for_live = sorted(df['City'].unique())
        selected_city_live = st.selectbox("Select a City to get its live AQI data", cities_for_live)

        if st.button("Get Live AQI Prediction", type="primary"):
            with st.spinner(f"🔄 Fetching live data for {selected_city_live}..."):
                response = get_live_data(selected_city_live, API_KEY)
                if "error" in response:
                    st.error(f"**Error:** {response['error']}")
                else:
                    components = response['data']['list'][0]['components']
                    live_pollutants = {
                        'PM2.5': components.get('pm2_5', 0),
                        'PM10': components.get('pm10', 0),
                        'NO2': components.get('no2', 0),
                        'CO': components.get('co', 0) / 1000,
                        'SO2': components.get('so2', 0),
                        'O3': components.get('o3', 0)
                    }

                    input_df = pd.DataFrame([live_pollutants], columns=features)

                    if model_data:
                        prediction_result = make_super_hybrid_prediction(input_df, model_data)

                        if prediction_result:
                            st.success("✅ Prediction completed using Super Hybrid Model!")

                            # Display ONLY the final AQI value
                            final_aqi = prediction_result['final_aqi']

                            # Create two columns for better layout
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                # Show AQI value in large format
                                st.markdown(
                                    f"<h1 style='text-align: center; color: #1f77b4; font-size: 4rem;'>{final_aqi:.0f}</h1>",
                                    unsafe_allow_html=True)
                                st.markdown("<h3 style='text-align: center;'>AQI</h3>", unsafe_allow_html=True)

                            with col2:
                                # Get health advisory based on 20-point ranges
                                category, advice = get_health_advisory_20_range(final_aqi)

                                # Display category with color coding
                                st.markdown(f"<h2 style='color: #2e7d32;'>{category}</h2>", unsafe_allow_html=True)

                                # Display health advisory
                                st.info(f"**Health Advisory:** {advice}")

                            # Show pollutant levels in expander (optional)
                            with st.expander("📊 View Detailed Pollutant Levels & Model Analysis"):
                                st.subheader("Live Pollutant Levels")
                                live_df = pd.DataFrame([live_pollutants]).T.rename(columns={0: "Concentration (μg/m³)"})
                                st.bar_chart(live_df)

                                # Show model weights (optional)
                                st.subheader("🧠 Model Analysis")
                                weights_df = pd.DataFrame({
                                    'Model': ['Deep Learning Hybrid', 'Machine Learning Hybrid'],
                                    'Weight': [prediction_result['dl_weight'], prediction_result['ml_weight']],
                                    'Prediction': [prediction_result['dl_prediction'],
                                                   prediction_result['ml_prediction']]
                                })
                                st.dataframe(weights_df, use_container_width=True)

elif page == "📊 Historical AQI Lookup":
    st.header("📊 Historical AQI Analysis")

    if df is not None:
        selected_city = st.selectbox("Select a City", sorted(df['City'].unique()))

        global_min_date = df['Date'].min().date()
        global_max_date = df['Date'].max().date()

        city_df = df[df['City'] == selected_city]
        city_min_date = city_df['Date'].min().date() if not city_df.empty else global_min_date

        selected_date = st.date_input(
            "Select a Date",
            value=city_min_date,
            min_value=global_min_date,
            max_value=global_max_date
        )

        if st.button("Analyze Historical AQI", type="primary"):
            record = city_df[city_df['Date'].dt.date == selected_date]
            if record.empty:
                st.error("No data available for the selected city and date.")
            else:
                actual_aqi = record['AQI'].values[0]

                # Create two columns for better layout
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(label="Actual AQI", value=f"{actual_aqi:.0f}" if pd.notna(actual_aqi) else "N/A")

                with col2:
                    # Get health advisory based on 20-point ranges
                    category, advice = get_health_advisory_20_range(actual_aqi)
                    st.info(f"**Category:** {category}\n\n**Health Advisory:** {advice}")

                # Make prediction for comparison
                pollutant_data = record[features].iloc[0:1]
                if model_data:
                    prediction_result = make_super_hybrid_prediction(pollutant_data, model_data)
                    if prediction_result:
                        st.metric("Super Hybrid Prediction", f"{prediction_result['final_aqi']:.0f}")

                        # Calculate accuracy
                        if pd.notna(actual_aqi):
                            error = abs(prediction_result['final_aqi'] - actual_aqi)
                            st.metric("Prediction Error", f"{error:.2f} AQI points")

                st.subheader("Pollutant Concentration Levels")
                pollutant_data_display = record[features].T.rename(columns={record.index[0]: "Concentration"})
                st.bar_chart(pollutant_data_display)

elif page == "📈 Future AQI Forecast":
    st.header("📈 Future AQI Forecast")

    if df is not None:
        selected_city = st.selectbox("Select a City", sorted(df['City'].unique()))
        days_to_predict = st.slider("Select number of days to forecast (1-7)", 1, 7, 3)

        if st.button("Generate AQI Forecast", type="primary"):
            city_df = df[df['City'] == selected_city]
            st.subheader(f"🔮 AQI Forecast for {selected_city}")

            start_date = datetime.now().date()
            forecast_results = []

            with st.spinner("🔄 Generating forecasts..."):
                for i in range(days_to_predict):
                    future_date = start_date + timedelta(days=i)
                    historical_data = city_df[
                        (city_df['Day'] == future_date.day) & (city_df['Month'] == future_date.month)]

                    if not historical_data.empty and not historical_data[features].isnull().values.all():
                        avg_pollutants = historical_data[features].mean()
                        input_df = pd.DataFrame([avg_pollutants], columns=features)

                        if model_data:
                            prediction_result = make_super_hybrid_prediction(input_df, model_data)
                            if prediction_result:
                                forecast_results.append({
                                    "Date": future_date.strftime("%Y-%m-%d"),
                                    "Predicted AQI": prediction_result['final_aqi'],
                                    "Pollutants": avg_pollutants.to_dict()
                                })

            if not forecast_results:
                st.warning("No historical data available to make a forecast for the selected date range.")
            else:
                forecast_df = pd.DataFrame(forecast_results).set_index("Date")

                # Display forecast chart
                st.subheader("Forecast Trend")
                st.line_chart(forecast_df['Predicted AQI'])

                # Daily breakdown
                st.markdown("---")
                st.subheader("📅 Daily Forecast Details")

                for result in forecast_results:
                    date, aqi, pollutants = result['Date'], result['Predicted AQI'], result['Pollutants']
                    category, advice = get_health_advisory_20_range(aqi)

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**📅 {date}**")
                        st.metric("AQI", f"{aqi:.0f}")
                    with col2:
                        st.info(f"**Category:** {category}\n\n**Advisory:** {advice}")

                    with st.expander("Show Estimated Pollutant Levels"):
                        pollutants_df = pd.DataFrame([pollutants]).T
                        pollutants_df.columns = ["Concentration (μg/m³)"]
                        st.dataframe(pollutants_df)

                    st.markdown("---")

elif page == "⚙️ Manual Input Prediction":
    st.header("⚙️ Manual AQI Prediction")
    st.markdown("Enter pollutant values manually to get AQI prediction")

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=50.0)
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=500.0, value=80.0)
        no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=400.0, value=40.0)

    with col2:
        co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=400.0, value=20.0)
        o3 = st.number_input("O3 (μg/m³)", min_value=0.0, max_value=400.0, value=50.0)

    if st.button("Predict AQI", type="primary"):
        input_data = {
            'PM2.5': pm25, 'PM10': pm10, 'NO2': no2,
            'CO': co, 'SO2': so2, 'O3': o3
        }

        input_df = pd.DataFrame([input_data], columns=features)

        if model_data:
            prediction_result = make_super_hybrid_prediction(input_df, model_data)

            if prediction_result:
                st.success("✅ Prediction completed!")

                # Create two columns for better layout
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Show AQI value in large format
                    st.markdown(
                        f"<h1 style='text-align: center; color: #1f77b4; font-size: 4rem;'>{prediction_result['final_aqi']:.0f}</h1>",
                        unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>AQI</h3>", unsafe_allow_html=True)

                with col2:
                    # Get health advisory based on 20-point ranges
                    category, advice = get_health_advisory_20_range(prediction_result['final_aqi'])

                    # Display category with color coding
                    st.markdown(f"<h2 style='color: #2e7d32;'>{category}</h2>", unsafe_allow_html=True)

                    # Display health advisory
                    st.info(f"**Health Advisory:** {advice}")

                # Show input data in expander
                with st.expander("📋 View Input Data & Model Analysis"):
                    st.subheader("Input Data")
                    input_df_display = input_df.T.rename(columns={0: "Value"})
                    st.dataframe(input_df_display)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & Super Hybrid AI")

if not model_data:
    st.error("""
    ❌ Super Hybrid Model not loaded!

    Please ensure:
    1. You have run the Super Hybrid training code first
    2. 'super_hybrid_dl_ml_meta.joblib' file exists in the same directory
    3. All required packages are installed
    """)