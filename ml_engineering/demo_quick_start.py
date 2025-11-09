#!/usr/bin/env python3
"""
Quick Start Demo - No MLflow Required

Demonstrates the core ML infrastructure without requiring MLflow installation.
Shows:
- Pipeline preprocessing
- Feature validation
- Model training with class balancing
- Calibration
- Test set evaluation

Usage:
    python ml_engineering/demo_quick_start.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning to prioritize
from config.paths import CRASH_LEVEL_ML

# Import our new infrastructure (no MLflow required)
from ml_engineering.preprocessing import (
    create_crash_classifier_pipeline,
    CRASH_NUMERIC_FEATURES,
    CRASH_CATEGORICAL_FEATURES,
    CRASH_TARGET,
    check_for_leakage
)

from ml_engineering.evaluation import (
    evaluate_classifier,
    calibrate_model,
    find_optimal_threshold
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    print('\n' + '='*70)
    print('CRASH PREDICTION - QUICK START DEMO')
    print('='*70 + '\n')

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print('STEP 1: Loading data...')
    data_dir = CRASH_LEVEL_ML

    train = pd.read_csv(data_dir / 'train_latest.csv', low_memory=False)
    val = pd.read_csv(data_dir / 'val_latest.csv', low_memory=False)
    test = pd.read_csv(data_dir / 'test_latest.csv', low_memory=False)

    print(f'  ✓ Train: {len(train):,} samples')
    print(f'  ✓ Val:   {len(val):,} samples')
    print(f'  ✓ Test:  {len(test):,} samples')

    # ========================================================================
    # 2. DATA VALIDATION & CLEANUP
    # ========================================================================
    print('\nSTEP 2: Validating data (checking for leakage)...')

    # Remove Description if it exists (contains outcome information)
    if 'Description' in train.columns:
        print('  ⚠️  Removing Description column (data leakage risk)')
        train = train.drop(columns=['Description'])
        val = val.drop(columns=['Description'])
        test = test.drop(columns=['Description'])

    # Fix categorical columns - convert NaN to 'missing' string
    print('\n  Cleaning categorical features...')
    categorical_cols = [col for col in CRASH_CATEGORICAL_FEATURES if col in train.columns]
    for col in categorical_cols:
        # Convert all values to strings, replacing NaN with 'missing'
        train[col] = train[col].fillna('missing').astype(str)
        val[col] = val[col].fillna('missing').astype(str)
        test[col] = test[col].fillna('missing').astype(str)
    print(f'    ✓ Fixed {len(categorical_cols)} categorical columns')

    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        try:
            check_for_leakage(df, dataset_type='crash')
        except ValueError as e:
            print(f'  ❌ {e}')
            return

    # ========================================================================
    # 3. PREPARE FEATURES
    # ========================================================================
    print('\nSTEP 3: Preparing features...')

    # Filter to available features
    available_numeric = [f for f in CRASH_NUMERIC_FEATURES if f in train.columns]
    available_categorical = [f for f in CRASH_CATEGORICAL_FEATURES if f in train.columns]

    print(f'  Numeric features: {len(available_numeric)}/{len(CRASH_NUMERIC_FEATURES)}')
    print(f'  Categorical features: {len(available_categorical)}/{len(CRASH_CATEGORICAL_FEATURES)}')

    if len(available_numeric) < 5:
        print(f'  ⚠️  WARNING: Only {len(available_numeric)} numeric features available!')
        print(f'  Available: {available_numeric}')

    feature_cols = available_numeric + available_categorical

    X_train = train[feature_cols]
    y_train = train[CRASH_TARGET]

    X_val = val[feature_cols]
    y_val = val[CRASH_TARGET]

    X_test = test[feature_cols]
    y_test = test[CRASH_TARGET]

    print(f'  ✓ Using {len(feature_cols)} features')
    print(f'  ✓ Target distribution: {y_train.mean()*100:.1f}% high-severity crashes')

    # ========================================================================
    # 4. TRAIN LOGISTIC REGRESSION
    # ========================================================================
    print('\n' + '='*70)
    print('STEP 4: Training Logistic Regression (with Pipeline)')
    print('='*70)

    # Create sklearn Pipeline
    lr_pipeline = create_crash_classifier_pipeline(
        numeric_features=available_numeric,
        categorical_features=available_categorical
    )

    # Set model with class balancing
    lr_pipeline.set_params(
        classifier=LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handles class imbalance
        )
    )

    # Train
    print('\nTraining...')
    lr_pipeline.fit(X_train, y_train)
    print('  ✓ Training complete')

    # Evaluate on validation
    val_metrics = evaluate_classifier(lr_pipeline, X_val, y_val, name='Validation Set')

    # Calibrate probabilities for better confidence estimates
    print('\nCalibrating probabilities (isotonic regression)...')
    lr_calibrated = calibrate_model(lr_pipeline, X_val, y_val, method='isotonic')

    # Final test evaluation
    test_metrics_lr = evaluate_classifier(lr_calibrated, X_test, y_test, name='Test Set')

    # ========================================================================
    # 5. TRAIN RANDOM FOREST
    # ========================================================================
    print('\n' + '='*70)
    print('STEP 5: Training Random Forest (with Pipeline + Calibration)')
    print('='*70)

    # Create pipeline
    rf_pipeline = create_crash_classifier_pipeline(
        numeric_features=available_numeric,
        categorical_features=available_categorical
    )

    # Set model with class balancing
    rf_pipeline.set_params(
        classifier=RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'  # Handles class imbalance
        )
    )

    # Train
    print('\nTraining...')
    rf_pipeline.fit(X_train, y_train)
    print('  ✓ Training complete')

    # Evaluate on validation
    val_metrics = evaluate_classifier(rf_pipeline, X_val, y_val, name='Validation Set')

    # Show feature importance
    rf_model = rf_pipeline.named_steps['classifier']
    from ml_engineering.preprocessing.pipelines import get_feature_names
    feature_names = get_feature_names(rf_pipeline)
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print('\nTop 10 Feature Importances:')
    for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
        print(f'  {i:2d}. {feat:30s} {imp:.4f}')

    # Calibrate
    print('\nCalibrating probabilities...')
    rf_calibrated = calibrate_model(rf_pipeline, X_val, y_val, method='isotonic')

    # Find optimal threshold for F1 score
    y_val_proba = rf_calibrated.predict_proba(X_val)[:, 1]
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_val, y_val_proba, metric='f1')

    # Final test evaluation with optimal threshold
    print(f'\nEvaluating on test set with optimal threshold ({optimal_threshold:.2f})...')
    test_metrics_rf = evaluate_classifier(
        rf_calibrated, X_test, y_test,
        name='Test Set (Optimized)',
        threshold=optimal_threshold
    )

    # ========================================================================
    # 6. COMPARE MODELS
    # ========================================================================
    print('\n' + '='*70)
    print('STEP 6: Model Comparison (Test Set)')
    print('='*70 + '\n')

    comparison = pd.DataFrame({
        'Logistic Regression': test_metrics_lr,
        'Random Forest (Optimized)': test_metrics_rf
    }).T

    print(comparison[['accuracy', 'precision', 'recall', 'f1', 'auc', 'brier_score']].to_string())

    # Determine winner
    print('\n' + '='*70)
    if test_metrics_rf['auc'] > test_metrics_lr['auc']:
        winner = 'Random Forest'
        winner_auc = test_metrics_rf['auc']
        improvement = ((test_metrics_rf['auc'] - test_metrics_lr['auc']) / test_metrics_lr['auc'] * 100)
    else:
        winner = 'Logistic Regression'
        winner_auc = test_metrics_lr['auc']
        improvement = ((test_metrics_lr['auc'] - test_metrics_rf['auc']) / test_metrics_rf['auc'] * 100)

    print(f'🏆 WINNER: {winner}')
    print(f'   Best AUC: {winner_auc:.4f}')
    print(f'   Improvement: +{improvement:.1f}%')
    print('='*70)

    # ========================================================================
    # 7. WHAT'S NEW?
    # ========================================================================
    print('\n' + '='*70)
    print('✨ WHAT\'S NEW IN THIS PIPELINE?')
    print('='*70)
    print('''
1. ✅ sklearn Pipelines - Ensures train/test consistency
   - Automatic preprocessing (imputation, scaling, encoding)
   - No more manually tracking transformations
   - Prevents data leakage

2. ✅ Class Imbalance Handling - Better minority class performance
   - class_weight='balanced' in Logistic Regression
   - class_weight='balanced_subsample' in Random Forest
   - Expect +5-10% F1 score improvement

3. ✅ Categorical Features - No longer silently dropped!
   - Automatic one-hot encoding via Pipeline
   - weather_category, temp_category now used

4. ✅ Probability Calibration - More reliable confidence scores
   - Isotonic regression on validation set
   - Lower Brier scores = better calibration

5. ✅ Threshold Optimization - Maximize your target metric
   - Found optimal threshold for F1 score
   - Can optimize for precision/recall instead

6. ✅ Data Leakage Detection - Automatic validation
   - Checks for forbidden features (Severity, End_Time, etc.)
   - Prevents accidental target leakage

NEXT STEPS:
- Install MLflow: pip install -r requirements_ml.txt
- Run full training: python -m ml_engineering.train_with_mlflow
- View MLflow UI: mlflow ui (open http://localhost:5000)
- Try XGBoost: python -m ml_engineering.train_with_mlflow --model xgboost
- Hyperparameter tuning: python -m ml_engineering.train_with_mlflow --tune
    ''')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
