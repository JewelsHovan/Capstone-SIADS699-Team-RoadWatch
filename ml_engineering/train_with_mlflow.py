#!/usr/bin/env python3
"""
Complete Training Script with MLflow Tracking

Demonstrates the full ML engineering infrastructure:
- Data validation
- Pipeline preprocessing
- Model training (RF, XGBoost)
- Hyperparameter tuning
- Calibration
- Test set evaluation
- MLflow tracking
- Model persistence

Usage:
    python -m ml_engineering.train_with_mlflow --dataset crash
    python -m ml_engineering.train_with_mlflow --dataset crash --tune
    python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import CRASH_LEVEL_ML, SEGMENT_LEVEL_ML

# Import our new infrastructure
from ml_engineering.preprocessing import (
    create_crash_classifier_pipeline,
    CRASH_NUMERIC_FEATURES,
    CRASH_CATEGORICAL_FEATURES,
    CRASH_TARGET,
    validate_features,
    check_for_leakage
)

from ml_engineering.utils import (
    start_experiment,
    log_model_run,
    save_model_artifact,
    log_feature_importance
)

from ml_engineering.evaluation import (
    evaluate_classifier,
    calibrate_model,
    find_optimal_threshold
)

from ml_engineering.models import (
    train_xgboost_classifier,
    train_catboost_classifier,
    train_lightgbm_classifier
)
from ml_engineering.utils.tuning import tune_random_forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_and_validate_data(dataset_type='crash'):
    """Load and validate datasets"""
    print(f'\n{"#"*70}')
    print(f'# LOADING AND VALIDATING DATA')
    print(f'{"#"*70}\n')

    if dataset_type == 'crash':
        data_dir = CRASH_LEVEL_ML
        target = CRASH_TARGET
    else:
        raise ValueError('Only crash dataset supported in this demo')

    # Load datasets
    print('Loading datasets...')
    train = pd.read_csv(data_dir / 'train_latest.csv', low_memory=False)
    val = pd.read_csv(data_dir / 'val_latest.csv', low_memory=False)
    test = pd.read_csv(data_dir / 'test_latest.csv', low_memory=False)

    print(f'  Train: {len(train):,} samples')
    print(f'  Val:   {len(val):,} samples')
    print(f'  Test:  {len(test):,} samples')

    # Remove Description if it exists (contains outcome information)
    if 'Description' in train.columns:
        print('\n  ⚠️  Removing Description column (data leakage risk)')
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

    # Check for data leakage
    print('\nChecking for data leakage...')
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        check_for_leakage(df, dataset_type='crash')

    # Validate features
    print('\nValidating features...')
    available_numeric, missing_numeric = validate_features(
        train, CRASH_NUMERIC_FEATURES, 'crash'
    )
    available_categorical, missing_categorical = validate_features(
        train, CRASH_CATEGORICAL_FEATURES, 'crash'
    )

    if missing_numeric:
        print(f'  ⚠️  Missing numeric features: {missing_numeric}')
        print(f'  Continuing with available features only...')

    if missing_categorical:
        print(f'  ⚠️  Missing categorical features: {missing_categorical}')
        print(f'  Continuing with available features only...')

    # Use only available features
    numeric_features = available_numeric
    categorical_features = available_categorical

    # Prepare X, y splits
    feature_cols = numeric_features + categorical_features

    X_train = train[feature_cols]
    y_train = train[target]

    X_val = val[feature_cols]
    y_val = val[target]

    X_test = test[feature_cols]
    y_test = test[target]

    print(f'\n✓ Data loaded and validated')
    print(f'  Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical')
    print(f'  Target class distribution:')
    print(f'    Train: {y_train.mean()*100:.1f}% positive')
    print(f'    Val:   {y_val.mean()*100:.1f}% positive')
    print(f'    Test:  {y_test.mean()*100:.1f}% positive')

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features)


def train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test,
                          numeric_features, categorical_features):
    """Train baseline models: Logistic Regression and Random Forest"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING BASELINE MODELS')
    print(f'{"#"*70}\n')

    # Start MLflow experiment
    experiment_id = start_experiment('crash_severity_prediction')

    # ========================================================================
    # 1. LOGISTIC REGRESSION
    # ========================================================================
    print(f'\n{"="*70}')
    print('1. LOGISTIC REGRESSION WITH PIPELINE')
    print(f'{"="*70}')

    # Create pipeline
    lr_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Set model
    lr_pipeline.set_params(
        classifier=LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    )

    # Train
    print('\nTraining...')
    lr_pipeline.fit(X_train, y_train)

    # Evaluate on validation
    val_metrics = evaluate_classifier(lr_pipeline, X_val, y_val, name='Validation Set')

    # Calibrate
    print('\nCalibrating probabilities...')
    lr_calibrated = calibrate_model(lr_pipeline, X_val, y_val, method='isotonic')

    # Final test evaluation
    test_metrics = evaluate_classifier(lr_calibrated, X_test, y_test, name='Test Set')

    # Save artifact
    artifact_path = save_model_artifact(
        pipeline=lr_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='logistic_regression_calibrated'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='logistic_regression_baseline',
        params={
            'model': 'LogisticRegression',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'calibration': 'isotonic'
        },
        metrics=test_metrics,
        model=lr_calibrated,
        tags={'type': 'baseline', 'calibrated': 'yes'}
    )

    # ========================================================================
    # 2. RANDOM FOREST
    # ========================================================================
    print(f'\n{"="*70}')
    print('2. RANDOM FOREST WITH PIPELINE')
    print(f'{"="*70}')

    # Create pipeline
    rf_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Set model
    rf_pipeline.set_params(
        classifier=RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
    )

    # Train
    print('\nTraining...')
    rf_pipeline.fit(X_train, y_train)

    # Evaluate on validation
    val_metrics = evaluate_classifier(rf_pipeline, X_val, y_val, name='Validation Set')

    # Get feature importance
    rf_model = rf_pipeline.named_steps['classifier']
    from ml_engineering.preprocessing.pipelines import get_feature_names
    feature_names = get_feature_names(rf_pipeline)
    importance_dict = dict(zip(feature_names, rf_model.feature_importances_))

    # Calibrate
    print('\nCalibrating probabilities...')
    rf_calibrated = calibrate_model(rf_pipeline, X_val, y_val, method='isotonic')

    # Final test evaluation
    test_metrics = evaluate_classifier(rf_calibrated, X_test, y_test, name='Test Set')

    # Find optimal threshold
    y_val_proba = rf_calibrated.predict_proba(X_val)[:, 1]
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_val, y_val_proba, metric='f1')

    # Re-evaluate with optimal threshold
    print(f'\nRe-evaluating with optimal threshold ({optimal_threshold:.2f})...')
    test_metrics_optimized = evaluate_classifier(
        rf_calibrated, X_test, y_test,
        name='Test Set (Optimized Threshold)',
        threshold=optimal_threshold
    )

    # Save artifact
    artifact_path = save_model_artifact(
        pipeline=rf_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics_optimized,
        model_name='random_forest_calibrated_optimized'
    )

    # Log to MLflow
    run_id = log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='random_forest_baseline',
        params={
            'model': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 15,
            'class_weight': 'balanced_subsample',
            'calibration': 'isotonic',
            'optimal_threshold': optimal_threshold
        },
        metrics=test_metrics_optimized,
        model=rf_calibrated,
        tags={'type': 'baseline', 'calibrated': 'yes', 'threshold_tuned': 'yes'}
    )

    # Log feature importance
    import mlflow
    with mlflow.start_run(run_id=run_id):
        log_feature_importance(importance_dict)

    print(f'\n{"="*70}')
    print('✓ BASELINE MODELS COMPLETE')
    print(f'{"="*70}')


def train_with_tuning(X_train, y_train, X_val, y_val, X_test, y_test,
                      numeric_features, categorical_features):
    """Train Random Forest with hyperparameter tuning"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING WITH HYPERPARAMETER TUNING')
    print(f'{"#"*70}\n')

    # Start experiment
    start_experiment('crash_severity_prediction')

    # Tune hyperparameters
    print('Tuning Random Forest hyperparameters...')
    best_params = tune_random_forest(
        X_train, y_train,
        task='classification',
        n_trials=30,  # Increase for better results
        cv_folds=3,
        scoring='roc_auc'
    )

    # Create pipeline with best params
    rf_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Add class_weight to best params
    best_params['class_weight'] = 'balanced_subsample'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    rf_pipeline.set_params(
        classifier=RandomForestClassifier(**best_params)
    )

    # Train
    print('\nTraining final model with best hyperparameters...')
    rf_pipeline.fit(X_train, y_train)

    # Calibrate
    rf_calibrated = calibrate_model(rf_pipeline, X_val, y_val)

    # Evaluate
    test_metrics = evaluate_classifier(rf_calibrated, X_test, y_test, name='Test Set')

    # Save
    artifact_path = save_model_artifact(
        pipeline=rf_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='random_forest_tuned'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='random_forest_tuned',
        params=best_params,
        metrics=test_metrics,
        model=rf_calibrated,
        tags={'type': 'tuned', 'calibrated': 'yes'}
    )

    print(f'\n{"="*70}')
    print('✓ TUNED MODEL COMPLETE')
    print(f'{"="*70}')


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                  numeric_features, categorical_features):
    """Train XGBoost model"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING XGBOOST')
    print(f'{"#"*70}\n')

    # Start experiment
    start_experiment('crash_severity_prediction')

    # Create pipeline
    xgb_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Preprocess data
    print('Preprocessing data...')
    xgb_pipeline.set_params(classifier='passthrough')
    X_train_processed = xgb_pipeline.fit_transform(X_train, y_train)
    X_val_processed = xgb_pipeline.transform(X_val)
    X_test_processed = xgb_pipeline.transform(X_test)

    # Train XGBoost
    xgb_model, train_metrics = train_xgboost_classifier(
        X_train_processed, y_train,
        X_val_processed, y_val,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        early_stopping_rounds=50,
        verbose=False
    )

    # Remove early_stopping_rounds to allow calibration to refit without validation set
    # The model already used early stopping during training and stopped at best iteration
    xgb_model.set_params(early_stopping_rounds=None)

    # Create new pipeline with XGBoost
    xgb_pipeline.set_params(classifier=xgb_model)

    # Calibrate
    print('\nCalibrating probabilities...')
    xgb_calibrated = calibrate_model(xgb_pipeline, X_val, y_val)

    # Final test evaluation
    test_metrics = evaluate_classifier(xgb_calibrated, X_test, y_test, name='Test Set')

    # Save
    artifact_path = save_model_artifact(
        pipeline=xgb_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='xgboost_calibrated'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='xgboost_early_stopping',
        params={
            'model': 'XGBoost',
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'calibration': 'isotonic'
        },
        metrics=test_metrics,
        model=xgb_calibrated,
        tags={'type': 'boosting', 'calibrated': 'yes'}
    )

    print(f'\n{"="*70}')
    print('✓ XGBOOST COMPLETE')
    print(f'{"="*70}')


def train_xgboost_tuned(X_train, y_train, X_val, y_val, X_test, y_test,
                        numeric_features, categorical_features):
    """Train XGBoost with hyperparameter tuning"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING XGBOOST WITH HYPERPARAMETER TUNING')
    print(f'{"#"*70}\n')

    # Start experiment
    start_experiment('crash_severity_prediction')

    # Create pipeline and preprocess
    print('Preprocessing data...')
    pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    pipeline.set_params(classifier='passthrough')
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    # Import tuning function
    from ml_engineering.utils.tuning import tune_xgboost

    # Tune hyperparameters with more focus on regularization
    print('Tuning XGBoost hyperparameters...')
    print('This will test 30 different parameter combinations.')
    print('Focus: Reducing overfitting with regularization\n')

    best_params = tune_xgboost(
        X_train_processed, y_train,
        task='classification',
        n_trials=30,
        cv_folds=3,
        scoring='roc_auc'
    )

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Add required params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'auc'

    # Remove early_stopping_rounds from best_params if it exists (will cause issues with calibration)
    best_params.pop('early_stopping_rounds', None)

    # Train final model with best hyperparameters
    print(f'\n{"="*70}')
    print('TRAINING FINAL XGBOOST WITH BEST HYPERPARAMETERS')
    print(f'{"="*70}')

    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(**best_params)

    print('\nTraining...')
    xgb_model.fit(X_train_processed, y_train)

    # Evaluate before calibration
    from ml_engineering.evaluation import evaluate_classifier
    val_metrics = evaluate_classifier(xgb_model, X_val_processed, y_val, name='Validation Set')

    # Put model in pipeline
    pipeline.set_params(classifier=xgb_model)

    # Calibrate
    print('\nCalibrating probabilities...')
    xgb_calibrated = calibrate_model(pipeline, X_val, y_val)

    # Final test evaluation
    test_metrics = evaluate_classifier(xgb_calibrated, X_test, y_test, name='Test Set')

    # Save
    artifact_path = save_model_artifact(
        pipeline=xgb_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='xgboost_tuned_calibrated'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='xgboost_tuned',
        params=best_params,
        metrics=test_metrics,
        model=xgb_calibrated,
        tags={'type': 'tuned', 'calibrated': 'yes'}
    )

    print(f'\n{"="*70}')
    print('✓ TUNED XGBOOST COMPLETE')
    print(f'{"="*70}')
    print(f'\nBest Hyperparameters:')
    for param, value in sorted(best_params.items()):
        print(f'  {param}: {value}')


def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   numeric_features, categorical_features):
    """Train CatBoost model"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING CATBOOST')
    print(f'{"#"*70}\n')

    # Start experiment
    start_experiment('crash_severity_prediction')

    # Create pipeline
    catboost_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Preprocess data
    print('Preprocessing data...')
    catboost_pipeline.set_params(classifier='passthrough')
    X_train_processed = catboost_pipeline.fit_transform(X_train, y_train)
    X_val_processed = catboost_pipeline.transform(X_val)
    X_test_processed = catboost_pipeline.transform(X_test)

    # Train CatBoost (it handles categorical features natively, but we already one-hot encoded)
    catboost_model, train_metrics = train_catboost_classifier(
        X_train_processed, y_train,
        X_val_processed, y_val,
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        early_stopping_rounds=50,
        verbose=100
    )

    # Put model in pipeline
    catboost_pipeline.set_params(classifier=catboost_model)

    # Calibrate
    print('\nCalibrating probabilities...')
    catboost_calibrated = calibrate_model(catboost_pipeline, X_val, y_val)

    # Final test evaluation
    test_metrics = evaluate_classifier(catboost_calibrated, X_test, y_test, name='Test Set')

    # Save
    artifact_path = save_model_artifact(
        pipeline=catboost_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='catboost_calibrated'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='catboost_early_stopping',
        params={
            'model': 'CatBoost',
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.05,
            'calibration': 'isotonic'
        },
        metrics=test_metrics,
        model=catboost_calibrated,
        tags={'type': 'boosting', 'calibrated': 'yes'}
    )

    print(f'\n{"="*70}')
    print('✓ CATBOOST COMPLETE')
    print(f'{"="*70}')


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                   numeric_features, categorical_features):
    """Train LightGBM model"""
    print(f'\n{"#"*70}')
    print(f'# TRAINING LIGHTGBM')
    print(f'{"#"*70}\n')

    # Start experiment
    start_experiment('crash_severity_prediction')

    # Create pipeline
    lgbm_pipeline = create_crash_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Preprocess data
    print('Preprocessing data...')
    lgbm_pipeline.set_params(classifier='passthrough')
    X_train_processed = lgbm_pipeline.fit_transform(X_train, y_train)
    X_val_processed = lgbm_pipeline.transform(X_val)
    X_test_processed = lgbm_pipeline.transform(X_test)

    # Train LightGBM
    lgbm_model, train_metrics = train_lightgbm_classifier(
        X_train_processed, y_train,
        X_val_processed, y_val,
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.05,
        early_stopping_rounds=50,
        verbose=100
    )

    # Put model in pipeline
    lgbm_pipeline.set_params(classifier=lgbm_model)

    # Calibrate
    print('\nCalibrating probabilities...')
    lgbm_calibrated = calibrate_model(lgbm_pipeline, X_val, y_val)

    # Final test evaluation
    test_metrics = evaluate_classifier(lgbm_calibrated, X_test, y_test, name='Test Set')

    # Save
    artifact_path = save_model_artifact(
        pipeline=lgbm_calibrated,
        feature_cols=numeric_features + categorical_features,
        metrics=test_metrics,
        model_name='lightgbm_calibrated'
    )

    # Log to MLflow
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='lightgbm_early_stopping',
        params={
            'model': 'LightGBM',
            'n_estimators': 1000,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'calibration': 'isotonic'
        },
        metrics=test_metrics,
        model=lgbm_calibrated,
        tags={'type': 'boosting', 'calibrated': 'yes'}
    )

    print(f'\n{"="*70}')
    print('✓ LIGHTGBM COMPLETE')
    print(f'{"="*70}')


def main():
    parser = argparse.ArgumentParser(description='Train crash prediction models with MLflow')
    parser.add_argument('--dataset', choices=['crash', 'segment'], default='crash',
                       help='Dataset to train on')
    parser.add_argument('--model', choices=['baseline', 'xgboost', 'catboost', 'lightgbm', 'all'], default='baseline',
                       help='Which models to train')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')

    args = parser.parse_args()

    # Load and validate data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     numeric_features, categorical_features) = load_and_validate_data(args.dataset)

    # Train models based on arguments
    if args.model == 'baseline' or args.model == 'all':
        train_baseline_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )

    if args.tune and args.model == 'baseline':
        train_with_tuning(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )

    if args.model == 'xgboost':
        if args.tune:
            # Tune XGBoost hyperparameters
            train_xgboost_tuned(
                X_train, y_train, X_val, y_val, X_test, y_test,
                numeric_features, categorical_features
            )
        else:
            # Use default XGBoost params
            train_xgboost(
                X_train, y_train, X_val, y_val, X_test, y_test,
                numeric_features, categorical_features
            )

    if args.model == 'catboost':
        train_catboost(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )

    if args.model == 'lightgbm':
        train_lightgbm(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )

    if args.model == 'all':
        train_xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )
        train_catboost(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )
        train_lightgbm(
            X_train, y_train, X_val, y_val, X_test, y_test,
            numeric_features, categorical_features
        )

    # Print summary
    print(f'\n{"#"*70}')
    print('# TRAINING COMPLETE!')
    print(f'{"#"*70}\n')
    print('View results in MLflow:')
    print('  cd /Users/julienh/Desktop/MADS/Capstone')
    print('  mlflow ui')
    print('  Open http://localhost:5000 in your browser')
    print('')
    print('Saved model artifacts in: models/artifacts/')
    print('')


if __name__ == '__main__':
    main()
