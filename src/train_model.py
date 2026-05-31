import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import get_mongo_client
import warnings
warnings.filterwarnings('ignore')

def train_ensemble():
    # Load data
    print(" Loading training data...")
    X_train, y_train = joblib.load('data/train.pkl')
    X_val, y_val = joblib.load('data/val.pkl')
    X_test, y_test = joblib.load('data/test.pkl')
    feature_cols = joblib.load('data/feature_cols.pkl')
    
    # IMPROVED: Better model configurations
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,  # More trees
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,  # More estimators
            learning_rate=0.05,  # Lower learning rate
            max_depth=6,  # Control depth
            subsample=0.8,  # Prevent overfitting
            colsample_bytree=0.8,  # Feature sampling
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(  # Faster & better
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000))
        ])
    }
    
    # Train models
    trained_models = {}
    results = []
    
    print("Training individual models...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Early stopping for XGBoost
        if name == 'XGBoost':
            model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
        else:
            model.fit(X_train, y_train)
        
        trained_models[name] = model
        
        # Validation metrics
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        results.append({'name': name, 'val_rmse': rmse, 'val_mae': mae, 'val_r2': r2})
        print(f"  Val RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Sort by validation RMSE
    results_sorted = sorted(results, key=lambda x: x['val_rmse'])
    
    # Compute inverse RMSE weights
    # Compute stronger inverse-square RMSE weights
    weights = [
    1.0 / (r['val_rmse'] ** 2 + 1e-6)
    for r in results_sorted
    ]

    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    print(f"\nEnsemble Weights (by validation performance):")
    for r, w in zip(results_sorted, weights):
        print(f"  {r['name']}: {w:.3f}")
    
    # MANUAL ENSEMBLE AVERAGING (No VotingRegressor.fit())
    print(f"\nComputing Manual Weighted Ensemble Predictions...")
    
    # Get individual predictions on test set
    individual_preds = {}
    for name in trained_models:
        individual_preds[name] = trained_models[name].predict(X_test)
    
    # Stack predictions
    preds_stack = np.column_stack([individual_preds[r['name']] for r in results_sorted])
    
    # Weighted average
    ensemble_pred = np.average(preds_stack, axis=1, weights=weights)
    
    # Evaluate ensemble
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ens_mae = mean_absolute_error(y_test, ensemble_pred)
    ens_r2 = r2_score(y_test, ensemble_pred)
    
    # Print all results
    print(f"\nFinal Test Evaluation:")
    print("="*75)
    print(f"{'Model':<25} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8}")
    print("="*75)
    
    for name in ['RandomForest', 'XGBoost', 'HistGradientBoosting', 'Ridge', 'ElasticNet']:
        y_pred = individual_preds[name]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name:<25} | {rmse:>8.2f} | {mae:>8.2f} | {r2:>8.4f}")
    
    print("-"*75)
    print(f"{'Ensemble (Weighted Avg)':<25} | {ens_rmse:>8.2f} | {ens_mae:>8.2f} | {ens_r2:>8.4f} ")
    print("="*75)
    
    # SHAP with DataFrame (safer)
    print(f"\nGenerating SHAP Explanations (Random Forest)...")
    rf_model = trained_models['RandomForest']
    
    # Convert to DataFrame for SHAP
    X_test_df = pd.DataFrame(X_test[:100], columns=feature_cols)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_df)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Features (SHAP):")
    print(feature_importance.head(5))
    
    # Save SHAP plots
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=14)
    plt.savefig('data/shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title('SHAP Summary Plot', fontsize=14)
    plt.savefig('data/shap_beeswarm.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"SHAP plots saved to data/shap_summary.png & shap_beeswarm.png")
    
    # Save to MongoDB
    print(f"\nSaving to Model Registry...")
    db = get_mongo_client()
    col = db.models
    
    model_doc = {
        "type": "weighted_ensemble_manual",
        "base_models": [r['name'] for r in results_sorted],
        "weights": weights,
        "metrics": {
            "test_rmse": float(ens_rmse),
            "test_mae": float(ens_mae),
            "test_r2": float(ens_r2)
        },
        "individual_models": {
            r['name']: {
                'val_rmse': float(r['val_rmse']),
                'val_mae': float(r['val_mae']),
                'val_r2': float(r['val_r2']),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, individual_preds[r['name']]))),
                'test_r2': float(r2_score(y_test, individual_preds[r['name']]))
            } for r in results
        },
        "feature_importance": feature_importance.to_dict('records'),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    col.insert_one(model_doc)
    
    # Save locally
    joblib.dump(trained_models, 'data/all_models.pkl')  # Save all models
    joblib.dump(weights, 'data/ensemble_weights.pkl')
    joblib.dump(feature_cols, 'data/feature_cols.pkl')
    joblib.dump(feature_importance, 'data/feature_importance.pkl')
    
    print(f"Models saved to data/all_models.pkl")
    print(f"Weights saved to data/ensemble_weights.pkl")
    print(f"Model registry updated in MongoDB")
    
    return trained_models, weights, feature_importance

if __name__ == "__main__":
    train_ensemble()