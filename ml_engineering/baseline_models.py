#!/usr/bin/env python3
"""
Baseline Models - Logistic Regression and Random Forest

Trains and evaluates baseline models on both datasets:
- Crash-level: Binary classification (high severity prediction)
- Segment-level: Regression (crash count prediction)

Models:
  - Logistic Regression (crash-level)
  - Random Forest Classifier (crash-level)
  - Random Forest Regressor (segment-level)

Usage:
  python -m ml_engineering.baseline_models --dataset crash
  python -m ml_engineering.baseline_models --dataset segment
  python -m ml_engineering.baseline_models --both
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import joblib
from datetime import datetime
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import CRASH_LEVEL_ML, SEGMENT_LEVEL_ML

def load_data(dataset_type):
    """Load train, val, test datasets"""
    if dataset_type == 'crash':
        data_dir = CRASH_LEVEL_ML
        target = 'high_severity'
    else:
        data_dir = SEGMENT_LEVEL_ML
        target = 'crash_count'

    train = pd.read_csv(data_dir / 'train_latest.csv')
    val = pd.read_csv(data_dir / 'val_latest.csv')
    test = pd.read_csv(data_dir / 'test_latest.csv')

    print(f"✓ Loaded {dataset_type} dataset:")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")

    return train, val, test, target

def prepare_features(train, val, test, target):
    """Prepare X, y and handle missing values"""
    # Select numeric features only
    feature_cols = train.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and ID columns
    feature_cols = [col for col in feature_cols if col not in [target, 'ID', 'id', 'segment_id']]

    print(f"\n  Using {len(feature_cols)} numeric features")

    # Split features and target (use .copy() to avoid SettingWithCopyWarning)
    X_train = train[feature_cols].copy()
    y_train = train[target]

    X_val = val[feature_cols].copy()
    y_val = val[target]

    X_test = test[feature_cols].copy()
    y_test = test[target]

    # Handle missing values (median imputation)
    print(f"  Handling missing values...")
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression (classification only)"""
    print(f"\n{'='*70}")
    print(f"TRAINING: Logistic Regression")
    print(f"{'='*70}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    print("Training...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    val_proba = model.predict_proba(X_val_scaled)[:, 1]

    print("\nPerformance:")
    print(f"  Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"  Train AUC:      {roc_auc_score(y_train, train_proba):.4f}")
    print(f"  Val Accuracy:   {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val AUC:        {roc_auc_score(y_val, val_proba):.4f}")
    print(f"  Val Precision:  {precision_score(y_val, val_pred):.4f}")
    print(f"  Val Recall:     {recall_score(y_val, val_pred):.4f}")
    print(f"  Val F1:         {f1_score(y_val, val_pred):.4f}")

    return model, scaler

def train_random_forest_classifier(X_train, y_train, X_val, y_val):
    """Train Random Forest Classifier"""
    print(f"\n{'='*70}")
    print(f"TRAINING: Random Forest Classifier")
    print(f"{'='*70}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=100,
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    print("\nPerformance:")
    print(f"  Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"  Train AUC:      {roc_auc_score(y_train, train_proba):.4f}")
    print(f"  Val Accuracy:   {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val AUC:        {roc_auc_score(y_val, val_proba):.4f}")
    print(f"  Val Precision:  {precision_score(y_val, val_pred):.4f}")
    print(f"  Val Recall:     {recall_score(y_val, val_pred):.4f}")
    print(f"  Val F1:         {f1_score(y_val, val_pred):.4f}")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
        print(f"  {i:2d}. {feat:30s} {imp:.4f}")

    return model

def train_random_forest_regressor(X_train, y_train, X_val, y_val):
    """Train Random Forest Regressor"""
    print(f"\n{'='*70}")
    print(f"TRAINING: Random Forest Regressor")
    print(f"{'='*70}")

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=100,
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    print("\nPerformance:")
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
    print(f"  Train MAE:  {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"  Train R²:   {r2_score(y_train, train_pred):.4f}")
    print(f"  Val RMSE:   {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")
    print(f"  Val MAE:    {mean_absolute_error(y_val, val_pred):.4f}")
    print(f"  Val R²:     {r2_score(y_val, val_pred):.4f}")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
        print(f"  {i:2d}. {feat:30s} {imp:.4f}")

    return model

def run_baseline_models(dataset_type):
    """Run all baseline models for dataset"""
    print(f"\n{'#'*70}")
    print(f"# BASELINE MODELS: {dataset_type.upper()}")
    print(f"{'#'*70}\n")

    # Load data
    train, val, test, target = load_data(dataset_type)

    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_features(
        train, val, test, target
    )

    # Train models
    if dataset_type == 'crash':
        # Classification task
        lr_model, scaler = train_logistic_regression(X_train, y_train, X_val, y_val)
        rf_model = train_random_forest_classifier(X_train, y_train, X_val, y_val)

        print(f"\n{'='*70}")
        print(f"✓ Trained 2 baseline models for crash-level dataset")
        print(f"{'='*70}\n")

    else:
        # Regression task
        rf_model = train_random_forest_regressor(X_train, y_train, X_val, y_val)

        print(f"\n{'='*70}")
        print(f"✓ Trained 1 baseline model for segment-level dataset")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument('--dataset', choices=['crash', 'segment'],
                       help='Dataset to train on')
    parser.add_argument('--both', action='store_true',
                       help='Train on both datasets')

    args = parser.parse_args()

    if args.both:
        run_baseline_models('crash')
        run_baseline_models('segment')
    elif args.dataset:
        run_baseline_models(args.dataset)
    else:
        print("Please specify --dataset crash, --dataset segment, or --both")
        parser.print_help()

if __name__ == "__main__":
    main()
