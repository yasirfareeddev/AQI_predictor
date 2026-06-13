# AQI Predictor (ATMOS) : End-to-End Serverless ML System

An end-to-end serverless machine learning system that forecasts **3-day Air Quality Index (AQI)** using automated pipelines, feature/model stores, SHAP based explainability and a real-time Streamlit dashboard.

---

## Table of Contents

- [Features](#-features)
- [Live Demo](#-live-demo)
- [Model Architecture](#-model-architecture)
- [Dataset Information](#-dataset-information)
- [Feature Importance (SHAP)](#-feature-importance-shap)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Automated Pipelines](#-automated-pipelines)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## Features

| Feature | Description |
|---------|-------------|
| **3-Day AQI Forecasting** | Predicts future air quality index up to 3 days ahead |
| **Weighted Ensemble Model** | Combines 5 regression models for best generalization |
| **SHAP Explainability** | Feature importance analysis via beeswarm and summary plots |
| **Automated Pipelines** | GitHub Actions workflows for feature & model retraining |
| **Feature Store** | Versioned, serialized feature columns and scaler artifacts |
| **Model Store** | Persisted trained models and ensemble weights |
| **MongoDB Integration** | Raw and processed data stored in MongoDB |
| **Streamlit Dashboard** | Interactive real time web interface with dark terminal aesthetic |

---

## Live Demo

🌐 **[aqi-predictor-atmos.streamlit.app](https://aqi-predictor-atmos.streamlit.app/)**

---

## Model Architecture

This project uses a **Weighted Ensemble** of five regression models, with weights assigned inversely proportional to validation RMSE to reward better performing models.

### **Ensemble Composition**

```
                        Raw AQI + Pollutant Features
                                    ↓
                         ┌──────────────────────┐
                         │   Feature Pipeline   │
                         │  (Lag, Roll, Delta)  │
                         └──────────┬───────────┘
                                    ↓
          ┌─────────────────────────┼─────────────────────────┐
          ↓                         ↓                         ↓
  HistGradientBoosting        RandomForest              GradientBoosting
  (Highest Weight)             (SHAP Source)             (Mid Weight)
          ↓                         ↓                         ↓
     XGBoost / LightGBM        Ridge Regression
     (Ensemble Members)        (Lowest Weight)
          ↓                         ↓                         ↓
          └─────────────────────────┼─────────────────────────┘
                                    ↓
                       Weighted Average Prediction
                                    ↓
                          3-Day AQI Forecast Output
```

### **Model Specifications**

| Parameter | Value |
|-----------|-------|
| **Ensemble Strategy** | Inverse RMSE Weighted Average |
| **Train/Val/Test Split** | 80% / 10% / 10% (chronological) |
| **Split Strategy** | Time series (no data leakage) |
| **Optimizer** | Per model (tree splits / gradient descent) |
| **Loss Function** | Mean Squared Error |
| **Scaler** | StandardScaler (serialized to `scaler.pkl`) |

### **Why a Weighted Ensemble?**

| Single Best Model | Weighted Ensemble |
|------------------|-------------------|
| Prone to high variance | Reduces prediction variance |
| Overfits specific patterns | Better generalization |
| HistGB R² = 0.9685 | Ensemble R² = **0.9726** |

---

## Dataset Information

### **Dataset Statistics**

| Metric | Value |
|--------|-------|
| **Data Source** | Real time air quality API + MongoDB |
| **Target Variable** | AQI (Air Quality Index) |
| **Forecast Horizon** | 3 days |
| **Train Split** | 80% (chronological) |
| **Validation Split** | 10% |
| **Test Split** | 10% |
| **Temporal Order** | Preserved (no shuffling) |

### **Input Features**

| Category | Features |
|----------|----------|
| **Pollutants** | `pm25`, `pm10`, `no2`, `o3`, `co`, `so2` |
| **Temporal** | `hour`, `day_of_week`, `month` |
| **Lag Features** | `aqi_lag1`, `aqi_lag2`, `aqi_lag3` |
| **Rolling Stats** | `aqi_roll_mean3`, `aqi_roll_std3` |
| **Momentum** | `aqi_change_rate` |

### **Feature Engineering**

The following transformations were applied during the feature pipeline:

- AQI lag features (1, 2, 3 steps back)
- Rolling mean and standard deviation (3-step window)
- AQI change rate (momentum/delta)
- Temporal decomposition (hour, weekday, month)
- StandardScaler normalization

---

## Feature Importance (SHAP)

SHAP analysis was performed using the **RandomForest** model to interpret feature contributions across predictions.

### **Top 5 Most Influential Features**

| Rank | Feature | Interpretation |
|------|---------|----------------|
| 1 | `hour` | Time-of-day drives strong diurnal AQI cycles |
| 2 | `aqi_roll_mean3` | Short-term AQI momentum is highly predictive |
| 3 | `aqi_change_rate` | Rate of change signals upcoming spikes |
| 4 | `pm25` | Fine particulate matter is the primary AQI driver |
| 5 | `aqi_lag1` | Previous hour AQI directly informs next prediction |

Temporal patterns and short-term AQI momentum dominate predictions, with PM₂.₅ confirming real world AQI standards where particulate matter is a major hazard contributor.

SHAP plots are saved to `data/shap_beeswarm.png` and `data/shap_summary.png`.

---

## Why Linear Models Underperformed

Linear models (Ridge, ElasticNet) received lower ensemble weights due to the inherently non-linear nature of AQI prediction. Tree-based ensembles naturally handle:

- Non-linear pollutant interactions
- Sudden AQI spike events
- Temporal dependencies and seasonality
- Shifting environmental baselines

| Model | Test R² | Relative Weight |
|-------|---------|----------------|
| **HistGradientBoosting** | 0.9685 | Highest |
| **RandomForest** | ~0.96+ | High |
| **GradientBoosting** | ~0.96+ | Medium |
| **Ridge Regression** | 0.9485 | Lowest |
| **Weighted Ensemble** | **0.9726** | — |

---

## Project Structure

```
AQI_PREDICTOR/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml          # Automated feature refresh
│       └── train_pipeline.yml            # Automated model retraining
│
├── data/
│   ├── all_models.pkl                    # Serialized trained models
│   ├── ensemble_weights.pkl              # Model ensemble weights
│   ├── feature_cols.pkl                  # Feature column definitions
│   ├── feature_importance.pkl            # SHAP importance values
│   ├── scaler.pkl                        # Fitted StandardScaler
│   ├── shap_beeswarm.png                 # SHAP beeswarm visualization
│   ├── shap_summary.png                  # SHAP summary plot
│   ├── test.pkl                          # Test split
│   ├── train.pkl                         # Train split
│   └── val.pkl                           # Validation split
│
├── mongo_data/                           # MongoDB raw/processed data
│
├── notebooks/
│   └── EDA_analysis.ipynb               # Exploratory data analysis
│
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   └── app.py                        # Streamlit dashboard
│   ├── __init__.py
│   ├── backfill_data.py                  # Historical data backfill
│   ├── feature_pipeline.py               # Feature engineering pipeline
│   ├── fetch_data.py                     # API data fetching
│   ├── prepare_data.py                   # Data preprocessing
│   └── train_model.py                    # Model training & ensemble
│
├── venv/
├── .env                                  # Environment variables (API keys)
├── .gitignore
├── LICENSE
├── README.md
├── REPORT.md
└── requirements.txt
```

---

## Requirements

### **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.12+ |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 1 GB | 3 GB+ |
| **MongoDB** | Any | MongoDB Atlas (free tier) |

### **Python Dependencies**

All dependencies are listed in `requirements.txt`:

```txt
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.43.0
pymongo>=4.5.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

---

## Installation

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yasirfareeddev/AQI_predictor.git
cd AQI_PREDICTOR
```

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Configure Environment**

```bash
# Copy and fill in your credentials
cp .env.example .env
# Add: MONGO_URI, AQI_API_KEY, etc.
```

### **Step 5: Verify Installation**

```bash
python -c "import sklearn; print(sklearn.__version__)"
python -c "import streamlit as st; print(st.__version__)"
```

---

## Usage

### **Option 1: Use Live Dashboard (Recommended)**

No setup required — visit the deployed app directly:

🌐 **[aqi-predictor-atmos.streamlit.app](https://aqi-predictor-atmos.streamlit.app/)**

### **Option 2: Run Locally**

```bash
# Run the Streamlit dashboard
streamlit run src/app/app.py
```

The app will open in your browser at: **`http://localhost:8501`**

### **Option 3: Retrain the Model**

```bash
# 1. Fetch fresh data
python src/fetch_data.py

# 2. Prepare and engineer features
python src/prepare_data.py
python src/feature_pipeline.py

# 3. Train models and save artifacts
python src/train_model.py
```

---

## Automated Pipelines

This project uses **GitHub Actions** for fully automated CI/CD-style ML pipelines.

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Feature Pipeline** | `feature_pipeline.yml` | Scheduled / Push | Fetches new data, engineers features, updates store |
| **Train Pipeline** | `train_pipeline.yml` | Scheduled / Push | Retrains all models, updates ensemble weights |

Artifacts (models, scaler, feature cols) are committed back to `data/` automatically on each run.

---

## Results

### **Model Performance**

| Metric | Value |
|--------|-------|
| **Final Test R²** | **0.9726** |
| **Test RMSE** | 3.84 |
| **Test MAE** | 3.00 |
| **Best Single Model R²** | 0.9685 (HistGradientBoosting) |
| **Baseline Linear R²** | 0.9485 (Ridge Regression) |

### **Validation Accuracy Progression**

```
Ridge Regression:       ████████████████████░░░ 94.85%
HistGradientBoosting:   █████████████████████░░ 96.85%
Weighted Ensemble:      ██████████████████████░ 97.26%
```

### **SHAP Feature Importance**

```
hour              ████████████████████  (highest impact)
aqi_roll_mean3    ████████████████░░░░
aqi_change_rate   ██████████████░░░░░░
pm25              ████████████░░░░░░░░
aqi_lag1          ██████████░░░░░░░░░░
```

---

### **Areas for Improvement**

- Extend forecast horizon beyond 3 days
- Add more pollutant sources and sensor locations
- Implement deep learning baselines (LSTM, Transformer)
- Add multi city support
- Build REST API endpoint for third party integration
- Add alerting system for hazardous AQI thresholds

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Ensemble Test R²** | 0.9726 |
| **Test RMSE** | 3.84 |
| **Models in Ensemble** | 5 |
| **Top Feature** | `hour` (SHAP) |
| **Forecast Horizon** | 3 days |
| **Pipeline Automation** | GitHub Actions (2 workflows) |
| **Data Store** | MongoDB |

---

<div align="center">

**Made by Yasir Fareed using Streamlit, Scikit-Learn & MongoDB**

⭐ **Star this repository if you found it helpful!**

</div>