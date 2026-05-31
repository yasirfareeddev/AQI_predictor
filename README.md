# AQI_predictor
End-to-end serverless ML system predicting 3 day AQI with automated pipelines, feature/model stores and Streamlit dashboard.

## Model Performance Summary

I trained five regression models using a chronological 80/10/10 time series split to avoid data leakage and preserve temporal order. Tree-based ensemble methods significantly outperformed linear baselines, with HistGradientBoosting achieving a Test R² score of 0.9685 compared to Ridge Regression at 0.9485.

A weighted ensemble approach was then implemented, where model weights were assigned inversely proportional to validation RMSE. The final weighted ensemble achieved the best overall generalization performance with:

* **Test R² = 0.9726**
* **RMSE = 3.84**
* **MAE = 3.00**

These results demonstrate that combining diverse learners reduces prediction variance and improves forecasting stability without overfitting.

---

## Feature Importance (SHAP Analysis)

SHAP analysis was performed using the RandomForest model to interpret feature contributions. The most influential features were:

1. `hour`
2. `aqi_roll_mean3`
3. `aqi_change_rate`
4. `pm25`
5. `aqi_lag1`

The results indicate that temporal patterns and short-term AQI momentum are the strongest predictors of future air quality conditions. PM₂.₅ concentration also showed high importance, which aligns with real-world AQI standards where particulate matter is a major contributor to hazardous air quality levels.

---

## Why Linear Models Underperformed

Linear models such as Ridge Regression and ElasticNet received lower ensemble weights because AQI prediction contains strong non-linear relationships, threshold effects, and complex interactions between pollutants and environmental conditions.

Tree-based ensemble methods naturally capture:

* non-linear pollutant interactions
* sudden AQI spikes
* temporal dependencies
* changing environmental patterns

As a result, ensemble tree models consistently outperformed linear approaches for AQI forecasting.
