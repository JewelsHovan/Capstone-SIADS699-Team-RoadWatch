#!/usr/bin/env python3
"""
Segment-Level ML Model Training

Trains count regression models to predict crash counts on road segments.
Used for risk assessment and work zone planning.

Models:
- Poisson Regression (baseline for count data)
- Negative Binomial Regression (handles overdispersion)
- Zero-Inflated Poisson (handles excess zeros)
- XGBoost Regressor (tree-based benchmark)

Target: crash_count (total crashes per segment over 7 years)

Usage:
    python -m ml_engineering.train_segment_models
    python -m ml_engineering.train_segment_models --model all

Author: ML Engineering Team
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
import xgboost as xgb
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import SEGMENT_LEVEL_ML
from ml_engineering.utils.persistence import save_model_artifact

print(f'\n{"#"*70}')
print('# SEGMENT-LEVEL MODEL TRAINING')
print(f'{"#"*70}\n')

# ============================================================================
# CONFIGURATION
# ============================================================================

NUMERIC_FEATURES = [
    'speed_limit',
    'through_lanes',
    'f_system',
    'urban_id',
    'aadt',
    'speed_x_aadt',
    'fsystem_x_urban',
    'lanes_x_aadt'
]

TARGET = 'crash_count'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_segment_data():
    """Load segment-level train/val/test data"""
    print(f'{"="*70}')
    print('LOADING SEGMENT DATA')
    print(f'{"="*70}\n')

    train_path = SEGMENT_LEVEL_ML / 'train_latest.csv'
    val_path = SEGMENT_LEVEL_ML / 'val_latest.csv'
    test_path = SEGMENT_LEVEL_ML / 'test_latest.csv'

    print(f'Loading from: {SEGMENT_LEVEL_ML}')

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    print(f'  Train: {len(train):,} segments')
    print(f'  Val:   {len(val):,} segments')
    print(f'  Test:  {len(test):,} segments')

    # Check target distribution
    print(f'\nTarget distribution ({TARGET}):')
    print(f'  Train: mean={train[TARGET].mean():.2f}, median={train[TARGET].median():.0f}, max={train[TARGET].max():.0f}')
    print(f'  Val:   mean={val[TARGET].mean():.2f}, median={val[TARGET].median():.0f}, max={val[TARGET].max():.0f}')
    print(f'  Test:  mean={test[TARGET].mean():.2f}, median={test[TARGET].median():.0f}, max={test[TARGET].max():.0f}')

    # Segments with crashes
    train_nonzero = (train[TARGET] > 0).sum()
    val_nonzero = (val[TARGET] > 0).sum()
    test_nonzero = (test[TARGET] > 0).sum()

    print(f'\nSegments with crashes:')
    print(f'  Train: {train_nonzero:,} ({train_nonzero/len(train)*100:.1f}%)')
    print(f'  Val:   {val_nonzero:,} ({val_nonzero/len(val)*100:.1f}%)')
    print(f'  Test:  {test_nonzero:,} ({test_nonzero/len(test)*100:.1f}%)')

    return train, val, test


def prepare_features(train, val, test):
    """Prepare features for modeling"""
    print(f'\n{"="*70}')
    print('PREPARING FEATURES')
    print(f'{"="*70}\n')

    # Check feature availability
    available_features = [f for f in NUMERIC_FEATURES if f in train.columns]
    missing_features = [f for f in NUMERIC_FEATURES if f not in train.columns]

    print(f'Available features: {len(available_features)}/{len(NUMERIC_FEATURES)}')
    if missing_features:
        print(f'  Missing: {missing_features}')

    # Extract features and target
    X_train = train[available_features].copy()
    X_val = val[available_features].copy()
    X_test = test[available_features].copy()

    y_train = train[TARGET].copy()
    y_val = val[TARGET].copy()
    y_test = test[TARGET].copy()

    # Fill missing values with 0 (segments with no data)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    print(f'\nFeature completeness (train):')
    for feat in available_features:
        completeness = (train[feat].notna()).mean() * 100
        print(f'  {feat:25s}: {completeness:5.1f}%')

    print(f'\n✓ Features prepared')
    print(f'  Shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}')

    return X_train, X_val, X_test, y_train, y_val, y_test, available_features


def evaluate_regression(y_true, y_pred, split_name='Test'):
    """Evaluate regression metrics"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Additional metrics for count data
    mean_pred = float(y_pred.mean())
    mean_true = float(y_true.mean())

    # Zero inflation metric
    pred_zeros = float((y_pred < 0.5).sum() / len(y_pred))
    true_zeros = float((y_true == 0).sum() / len(y_true))

    print(f'\n{split_name} Set Metrics:')
    print(f'  RMSE:          {rmse:.4f}')
    print(f'  MAE:           {mae:.4f}')
    print(f'  R²:            {r2:.4f}')
    print(f'  Mean Pred:     {mean_pred:.4f}')
    print(f'  Mean True:     {mean_true:.4f}')
    print(f'  % Pred Zeros:  {pred_zeros*100:.1f}%')
    print(f'  % True Zeros:  {true_zeros*100:.1f}%')

    return {
        f'{split_name.lower()}_rmse': rmse,
        f'{split_name.lower()}_mae': mae,
        f'{split_name.lower()}_r2': r2,
        f'{split_name.lower()}_mean_pred': mean_pred,
        f'{split_name.lower()}_mean_true': mean_true
    }


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_poisson_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Poisson Regression (GLM for count data)"""
    print(f'\n{"="*70}')
    print('TRAINING POISSON REGRESSION')
    print(f'{"="*70}\n')

    # Scale features (Poisson benefits from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print('Training Poisson GLM...')
    model = PoissonRegressor(max_iter=500, alpha=0.1)
    model.fit(X_train_scaled, y_train)

    print('✓ Training complete')

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate
    metrics = {}
    metrics.update(evaluate_regression(y_train, y_train_pred, 'Train'))
    metrics.update(evaluate_regression(y_val, y_val_pred, 'Val'))
    metrics.update(evaluate_regression(y_test, y_test_pred, 'Test'))

    return model, scaler, metrics


def train_xgboost_regressor(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost Regressor"""
    print(f'\n{"="*70}')
    print('TRAINING XGBOOST REGRESSOR')
    print(f'{"="*70}\n')

    # Configure for count data
    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'count:poisson',  # Poisson objective for count data
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50,
        'random_state': 42,
        'n_jobs': -1
    }

    print('Training XGBoost with Poisson objective...')
    print(f'  Max rounds: {params["n_estimators"]}')
    print(f'  Early stopping: {params["early_stopping_rounds"]} rounds\n')

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators']
    print(f'\n✓ Training stopped at iteration {best_iteration}')

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Clip negative predictions (shouldn't happen with Poisson objective, but safety check)
    y_train_pred = np.clip(y_train_pred, 0, None)
    y_val_pred = np.clip(y_val_pred, 0, None)
    y_test_pred = np.clip(y_test_pred, 0, None)

    # Evaluate
    metrics = {}
    metrics.update(evaluate_regression(y_train, y_train_pred, 'Train'))
    metrics.update(evaluate_regression(y_val, y_val_pred, 'Val'))
    metrics.update(evaluate_regression(y_test, y_test_pred, 'Test'))
    metrics['best_iteration'] = best_iteration

    return model, None, metrics


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train segment-level crash count models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['poisson', 'xgboost', 'all'],
                       help='Which model to train')
    args = parser.parse_args()

    # Load data
    train, val, test = load_segment_data()

    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_features(train, val, test)

    # Start MLflow experiment
    mlflow.set_experiment('segment_crash_prediction')

    # Train models
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['poisson', 'xgboost']
    else:
        models_to_train = [args.model]

    for model_name in models_to_train:
        with mlflow.start_run(run_name=f'{model_name}_segment_model'):
            # Log parameters
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('target', TARGET)
            mlflow.log_param('num_features', len(features))
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('val_size', len(X_val))
            mlflow.log_param('test_size', len(X_test))

            # Train model
            if model_name == 'poisson':
                model, scaler, metrics = train_poisson_model(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
            elif model_name == 'xgboost':
                model, scaler, metrics = train_xgboost_regressor(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(model, f'{model_name}_model')

            # Save model artifact
            artifact_path = save_model_artifact(
                pipeline=model,
                feature_cols=features,
                metrics=metrics,
                model_name=f'{model_name}_segment',
                run_id=mlflow.active_run().info.run_id
            )
            mlflow.log_param('artifact_dir', str(artifact_path))

            print(f'\n✓ Model saved to: {artifact_path}')

    print(f'\n{"#"*70}')
    print('# TRAINING COMPLETE!')
    print(f'{"#"*70}\n')
    print('View results in MLflow:')
    print('  mlflow ui')
    print('  Open http://localhost:5000')
    print()


if __name__ == '__main__':
    main()
