import pandas as pd
import numpy as np
from utils import get_mongo_client

def load_raw_data():
    db = get_mongo_client()
    cursor = db.raw_aqi.find()
    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError("No raw data found. Run backfill_data.py or fetch_data.py first.")

    # Handle mixed timestamp formats safely
    df['timestamp'] = pd.to_datetime(
        df['timestamp'],
        format='mixed',
        utc=True,
        errors='coerce'
    )

    # Remove invalid timestamps if any
    df = df.dropna(subset=['timestamp'])

    # Convert timezone aware UTC to timezone naive
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

def engineer_features(df):
    # 1. Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 2. Target
    df['target_aqi'] = df['aqi'].astype(float)

    # 3. Lag & Rolling features
    for col in ['pm25', 'pm10', 'aqi']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_roll_mean3'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_roll_std3'] = df[col].rolling(window=3, min_periods=1).std()

    # 4. Rate of change & ratios
    df['aqi_change_rate'] = df['aqi'].pct_change()
    df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-5)

    # 5. Clean & select
    df = df.dropna()
    feature_cols = [
    c for c in df.columns
    if c not in [
        '_id',
        'city',
        'dominant_pollutant',
        'aqi',
        'timestamp',
        'target_aqi'
    ]]
    df_feat = df[['timestamp', 'city'] + feature_cols + ['target_aqi']].copy()
    return df_feat

def store_features(df):
    db = get_mongo_client()
    col = db.features
    col.drop()  # Clear old features during dev
    records = df.to_dict(orient='records')
    col.insert_many(records)
    
    # Index for fast time-series queries
    col.create_index([("city", 1), ("timestamp", 1)])
    print(f"Stored {len(records)} feature documents in 'features' collection.")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    print("Loading raw data & engineering features...")
    df_raw = load_raw_data()
    df_feat = engineer_features(df_raw)
    store_features(df_feat)