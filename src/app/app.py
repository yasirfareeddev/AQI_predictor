import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

# Page config - Professional favicon
st.set_page_config(
    page_title="AQI Forecast System",
    page_icon="https://img.icons8.com/color/96/air-quality.png",  # favicon
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #34495e;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        border-left: 4px solid;
    }
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_mongo_client():
    """Connect to MongoDB"""
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    return MongoClient(uri)

@st.cache_resource
def load_model():
    """Load trained ensemble model"""
    try:
        models = joblib.load('data/all_models.pkl')
        weights = joblib.load('data/ensemble_weights.pkl')
        return models, weights
    except FileNotFoundError:
        return None, None

def get_latest_features(city="Islamabad"):
    """Fetch latest features from MongoDB"""
    db = get_mongo_client()["aqi_db"]
    latest = db.features.find_one(
        {"city": city},
        sort=[("timestamp", -1)]
    )
    return latest

def get_historical_data(city="Islamabad", days=30):
    """Fetch historical AQI data for visualization"""
    db = get_mongo_client()["aqi_db"]
    cursor = db.features.find(
        {"city": city}
    ).sort("timestamp", -1).limit(days * 24)
    
    df = pd.DataFrame(list(cursor))
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    return df

def predict_next_3_days(latest_features, models, weights):
    """Generate predictions for next 3 days (72 hours)"""
    if models is None:
        return None
    
    last_timestamp = pd.to_datetime(latest_features['timestamp'])
    future_timestamps = [last_timestamp + timedelta(hours=h) for h in range(1, 73)]
    
    predictions = []
    current_features = latest_features.copy()
    
    for i, ts in enumerate(future_timestamps):
        current_features['hour'] = ts.hour
        current_features['day_of_week'] = ts.dayofweek
        current_features['month'] = ts.month
        current_features['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
        current_features['timestamp'] = ts
        
        for key in current_features:
            if key not in ['_id', 'timestamp', 'city', 'target_aqi']:
                if isinstance(current_features[key], (int, float)):
                    noise = np.random.normal(0, 0.02 * abs(current_features[key]))
                    current_features[key] = current_features[key] + noise
        
        feature_cols = joblib.load('data/feature_cols.pkl')
        X_pred = pd.DataFrame([current_features])[feature_cols]
        
        individual_preds = {}
        for name, model in models.items():
            pred = model.predict(X_pred)[0]
            individual_preds[name] = pred
        
        preds_array = np.array([individual_preds[r] for r in individual_preds])
        ensemble_pred = np.average(preds_array, weights=weights)
        
        predictions.append({
            'timestamp': ts,
            'predicted_aqi': max(0, ensemble_pred),
            **individual_preds
        })
    
    return pd.DataFrame(predictions)

def get_aqi_category(aqi):
    """Determine AQI category and styling"""
    if aqi <= 50:
        return "Good", "#27ae60", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "#f39c12", "Acceptable air quality."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "#e74c3c", "Everyone may experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "#9b59b6", "Health alert: everyone is at risk."
    else:
        return "Hazardous", "#7f1d1d", "Emergency conditions: avoid all outdoor activity."

def main():
    # Header
    st.markdown('<h1 class="main-header">Air Quality Index Forecast System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; margin-bottom: 2rem;">Predicting AQI for the next 72 hours using ensemble machine learning models</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    city = st.sidebar.text_input("City", "Islamabad")
    st.sidebar.markdown("---")
    st.sidebar.info("Data source: AQICN API | Model: Weighted Ensemble (RF, XGB, HistGB, Ridge, ElasticNet)")
    
    # Load model
    models, weights = load_model()
    
    if models is None:
        st.error("Model files not found. Please ensure training has been completed and model artifacts are present in the data/ directory.")
        return
    
    # Fetch latest data
    with st.spinner("Loading data..."):
        latest = get_latest_features(city)
        historical_df = get_historical_data(city)
    
    if latest is None:
        st.error(f"No data available for {city}. Please verify data ingestion pipeline.")
        return
    
    # Current AQI metrics
    current_aqi = latest.get('target_aqi', latest.get('aqi', 0))
    category, color, message = get_aqi_category(current_aqi)
    
    st.markdown(f'<h2 class="sub-header">Current Conditions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<p class="metric-label">AQI Index</p><p class="metric-value">{current_aqi:.0f}</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="metric-label">Category</p><p class="metric-value" style="color: {color}">{category}</p>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<p class="metric-label">PM2.5</p><p class="metric-value">{latest.get("pm25", 0):.1f} <span style="font-size: 0.9rem; color: #7f8c8d">µg/m³</span></p>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<p class="metric-label">PM10</p><p class="metric-value">{latest.get("pm10", 0):.1f} <span style="font-size: 0.9rem; color: #7f8c8d">µg/m³</span></p>', unsafe_allow_html=True)
    
    # Alert box
    alert_style = f"background-color: {color}15; border-color: {color};"
    st.markdown(f'<div class="alert-box" style="{alert_style}">{message}</div>', unsafe_allow_html=True)
    
    # Forecast section
    st.markdown(f'<h2 class="sub-header">72-Hour Forecast</h2>', unsafe_allow_html=True)
    with st.spinner("Generating predictions..."):
        forecast_df = predict_next_3_days(latest, models, weights)
    
    if forecast_df is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['predicted_aqi'],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(color='#3498db', width=2.5),
            marker=dict(size=5, color='#3498db')
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="#f39c12", annotation_text="Moderate/Unhealthy Threshold")
        fig.add_hline(y=150, line_dash="dash", line_color="#e74c3c", annotation_text="Unhealthy Threshold")
        fig.add_hline(y=200, line_dash="dash", line_color="#9b59b6", annotation_text="Very Unhealthy Threshold")
        
        fig.update_layout(
            title="Predicted Air Quality Index - Next 72 Hours",
            xaxis_title="Date and Time",
            yaxis_title="AQI Value",
            hovermode='x unified',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Forecast", f"{forecast_df['predicted_aqi'].mean():.1f}")
        with col2:
            st.metric("Maximum Forecast", f"{forecast_df['predicted_aqi'].max():.1f}")
        with col3:
            st.metric("Minimum Forecast", f"{forecast_df['predicted_aqi'].min():.1f}")
    
    # Historical trends
    if not historical_df.empty:
        st.markdown(f'<h2 class="sub-header">Historical Trends</h2>', unsafe_allow_html=True)
        historical_df['date'] = historical_df['timestamp'].dt.date
        daily_avg = historical_df.groupby('date')['target_aqi'].mean().reset_index()
        
        fig_hist = px.bar(
            daily_avg.tail(7),
            x='date',
            y='target_aqi',
            title="Daily Average AQI - Past 7 Days",
            labels={'target_aqi': 'AQI', 'date': 'Date'},
            color='target_aqi',
            color_continuous_scale='RdYlGn_r'
        )
        fig_hist.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Feature importance
    st.markdown(f'<h2 class="sub-header">Model Interpretability</h2>', unsafe_allow_html=True)
    try:
        feature_importance = joblib.load('data/feature_importance.pkl')
        fig_shap = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Features by SHAP Importance",
            labels={'importance': 'Mean Absolute SHAP Value', 'feature': 'Feature Name'}
        )
        fig_shap.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_shap, use_container_width=True)
    except FileNotFoundError:
        st.info("Feature importance data not available. Run training pipeline to generate SHAP explanations.")
    
    # Raw data viewer
    with st.expander("View Recent Data Points"):
        display_cols = [c for c in historical_df.columns if c not in ['_id']]
        st.dataframe(historical_df[display_cols].tail(50), use_container_width=True)
    
    # Footer
    st.markdown('<div class="footer">System built with Streamlit, scikit-learn, XGBoost, and MongoDB. Data sourced from AQICN API.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()