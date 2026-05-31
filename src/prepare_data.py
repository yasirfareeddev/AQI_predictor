import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils import get_mongo_client

def load_and_prepare():
    db = get_mongo_client()
    df = pd.DataFrame(list(db.features.find()))
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def time_series_split(df, train_ratio=0.8, val_ratio=0.1):
    """Time-series aware split (no shuffling!)"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} samples ({train_end/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({(val_end-train_end)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({(n-val_end)/n*100:.1f}%)")
    
    return train_df, val_df, test_df

def prepare_features(df):
    """Separate features and target, handle missing values"""
    feature_cols = [c for c in df.columns if c not in ['_id', 'timestamp', 'city', 'target_aqi']]
    
    X = df[feature_cols].copy()
    y = df['target_aqi'].copy()
    
    # Drop rows with any NaN
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"Final feature matrix shape: {X.shape}")
    return X, y, feature_cols

if __name__ == "__main__":
    print("Loading data from MongoDB")
    df = load_and_prepare()
    
    print("\nPerforming time-series split")
    train_df, val_df, test_df = time_series_split(df)
    
    print("\n Preparing features and targets")
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    print("\nScaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nSaving processed data...")
    joblib.dump((X_train_scaled, y_train.values), 'data/train.pkl')
    joblib.dump((X_val_scaled, y_val.values), 'data/val.pkl')
    joblib.dump((X_test_scaled, y_test.values), 'data/test.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    joblib.dump(feature_cols, 'data/feature_cols.pkl')
    
    print("\nData preparation completed")
    print(f"Saved: data/train.pkl, data/val.pkl, data/test.pkl, data/scaler.pkl")
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")