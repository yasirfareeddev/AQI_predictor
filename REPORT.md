# AQI Predictor: Project Report

## 1. Executive Summary
This project delivers an end-to-end, serverless Air Quality Index (AQI) forecasting system for Islamabad. It utilizes a 100% serverless stack (MongoDB Atlas, GitHub Actions, Streamlit) to fetch real-time data, engineer time-series features, train an ensemble machine learning model, and serve interactive 72-hour forecasts with SHAP-based explainability.

## 2. System Architecture
The system follows a modular, automated pipeline:
1. **Feature Pipeline (`fetch_data.py`, `feature_pipeline.py`)**: Fetches raw pollutant/weather data from AQICN API, engineers 27 temporal and derived features (e.g., `aqi_roll_mean3`, `pm_ratio`), and stores them in MongoDB.
2. **Training Pipeline (`train_model.py`)**: Pulls historical data, applies chronological time-series splitting (80/10/10), trains 5 models, and registers the best weighted ensemble in the MongoDB Model Registry.
3. **Web Application (`app.py`)**: A Streamlit dashboard that loads the ensemble model, computes 72-hour rolling forecasts, and visualizes trends with hazard alerts.
4. **CI/CD Automation**: GitHub Actions triggers the feature pipeline hourly and the training pipeline daily at 2 AM UTC.

## 3. Exploratory Data Analysis (EDA) Insights
- **Diurnal Cycle**: AQI exhibits strong hourly patterns, peaking during morning/evening rush hours.
- **Feature Correlation**: PM2.5 and PM10 showed the highest correlation with target AQI, aligning with CPCB guidelines.
- **Data Quality**: Chronological splitting prevented future-to-past data leakage, ensuring realistic evaluation.

## 4. Model Training & Evaluation
We evaluated 5 models using a chronological split. Tree-based ensembles significantly outperformed linear baselines due to the non-linear nature of atmospheric pollution.

| Model | Validation RMSE | Test R² | Ensemble Weight |
|-------|----------------|---------|-----------------|
| HistGradientBoosting | 3.29 | 0.9685 | 33.8% |
| XGBoost | 3.79 | 0.9636 | 25.4% |
| RandomForest | 4.47 | 0.9565 | 18.3% |
| Ridge | 5.69 | 0.9485 | 11.3% |
| ElasticNet | 5.74 | 0.9480 | 11.1% |
| **Weighted Ensemble** | **3.84** | **0.9726** | **100%** |

*Note: The weighted ensemble (inverse RMSE weighting) reduced variance and achieved the highest generalization (Test R² = 0.9726).*

## 5. Model Explainability (SHAP)
SHapley Additive exPlanations were computed on the RandomForest base model. The top drivers of AQI predictions were:
1. `hour` (8.90): Captures daily traffic/industrial cycles.
2. `aqi_roll_mean3` (8.46): Short-term pollution momentum.
3. `aqi_change_rate` (4.20): Acceleration of pollution buildup.
4. `pm25` (2.88): Primary particulate matter driving the AQI index.

## 6. CI/CD & Deployment
- **Feature Store & Model Registry**: MongoDB Atlas (Free Tier) serves as the centralized, cloud-accessible database.
- **Automation**: GitHub Actions workflows (`feature_pipeline.yml`, `train_pipeline.yml`) ensure the system self-heals and stays up-to-date without manual intervention.
- **Dashboard**: Deployed via Streamlit Community Cloud, providing a responsive, real-time UI with dynamic Plotly visualizations and hazardous AQI alerts.

## 7. Limitations & Future Work
- **Forecast Horizon**: The current 72-hour forecast relies on rolling historical features with minor noise injection. Integrating a dedicated meteorological forecast API (e.g., OpenWeather 5-day forecast) would improve long-term accuracy.
- **Spatial Data**: Adding data from neighboring monitoring stations could capture regional pollution drift.
- **Deep Learning**: Future iterations could explore LSTMs or Temporal Fusion Transformers (TFT) for advanced sequence modeling.

---
*Developed as part of the 10 Pearls AQI Predictor internship project.*