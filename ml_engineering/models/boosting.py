#!/usr/bin/env python3
"""
Gradient Boosting Models

XGBoost and LightGBM implementations for crash prediction with automatic
class imbalance handling and early stopping.

Usage:
    from ml_engineering.models.boosting import train_xgboost_classifier

    model, metrics = train_xgboost_classifier(
        X_train, y_train,
        X_val, y_val,
        n_estimators=500,
        early_stopping_rounds=50
    )
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Tuple, Dict, Optional


def train_xgboost_classifier(
    X_train, y_train,
    X_val, y_val,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
    **kwargs
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Train XGBoost classifier with early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_estimators: Maximum number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Print progress
        **kwargs: Additional XGBoost parameters

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: XGBoost Classifier')
    print(f'{"="*70}')

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f'  Class imbalance ratio: {scale_pos_weight:.2f}')
    print(f'  Max boosting rounds: {n_estimators}')
    print(f'  Early stopping: {early_stopping_rounds} rounds\n')

    # Default parameters
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': early_stopping_rounds,  # Set in constructor for XGBoost 2.0+
        **kwargs
    }

    # Create and train model
    model = xgb.XGBClassifier(**params)

    print('Training...')
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )

    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else n_estimators

    print(f'\n✓ Training stopped at iteration {best_iteration}')

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'train_auc': roc_auc_score(y_train, train_proba),
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_auc': roc_auc_score(y_val, val_proba),
        'val_precision': precision_score(y_val, val_pred),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'best_iteration': best_iteration
    }

    print('\nPerformance:')
    print(f'  Train Accuracy: {metrics["train_accuracy"]:.4f}')
    print(f'  Train AUC:      {metrics["train_auc"]:.4f}')
    print(f'  Val Accuracy:   {metrics["val_accuracy"]:.4f}')
    print(f'  Val AUC:        {metrics["val_auc"]:.4f}')
    print(f'  Val Precision:  {metrics["val_precision"]:.4f}')
    print(f'  Val Recall:     {metrics["val_recall"]:.4f}')
    print(f'  Val F1:         {metrics["val_f1"]:.4f}')

    return model, metrics


def train_xgboost_regressor(
    X_train, y_train,
    X_val, y_val,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
    **kwargs
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train XGBoost regressor with early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_estimators: Maximum number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Print progress
        **kwargs: Additional XGBoost parameters

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: XGBoost Regressor')
    print(f'{"="*70}')

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': early_stopping_rounds,  # Set in constructor for XGBoost 2.0+
        **kwargs
    }

    model = xgb.XGBRegressor(**params)

    print('Training...')
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'train_r2': r2_score(y_train, train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'val_r2': r2_score(y_val, val_pred),
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else n_estimators
    }

    print('\nPerformance:')
    print(f'  Train RMSE: {metrics["train_rmse"]:.4f}')
    print(f'  Train MAE:  {metrics["train_mae"]:.4f}')
    print(f'  Train R²:   {metrics["train_r2"]:.4f}')
    print(f'  Val RMSE:   {metrics["val_rmse"]:.4f}')
    print(f'  Val MAE:    {metrics["val_mae"]:.4f}')
    print(f'  Val R²:     {metrics["val_r2"]:.4f}')

    return model, metrics


def train_lightgbm_classifier(
    X_train, y_train,
    X_val, y_val,
    n_estimators: int = 500,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    **kwargs
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """
    Train LightGBM classifier with early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_estimators: Maximum number of boosting rounds
        num_leaves: Maximum number of leaves
        learning_rate: Learning rate
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Print every N iterations
        **kwargs: Additional LightGBM parameters

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: LightGBM Classifier')
    print(f'{"="*70}')

    # Calculate is_unbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()

    print(f'  Class distribution: {neg_count} negative, {pos_count} positive')
    print(f'  Max boosting rounds: {n_estimators}')
    print(f'  Early stopping: {early_stopping_rounds} rounds\n')

    params = {
        'n_estimators': n_estimators,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'is_unbalance': True,  # Handle class imbalance
        'metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,  # Suppress warnings
        **kwargs
    }

    model = lgb.LGBMClassifier(**params)

    print('Training...')
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(verbose)]
    )

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'train_auc': roc_auc_score(y_train, train_proba),
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_auc': roc_auc_score(y_val, val_proba),
        'val_precision': precision_score(y_val, val_pred),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else n_estimators
    }

    print('\nPerformance:')
    print(f'  Train Accuracy: {metrics["train_accuracy"]:.4f}')
    print(f'  Train AUC:      {metrics["train_auc"]:.4f}')
    print(f'  Val Accuracy:   {metrics["val_accuracy"]:.4f}')
    print(f'  Val AUC:        {metrics["val_auc"]:.4f}')
    print(f'  Val Precision:  {metrics["val_precision"]:.4f}')
    print(f'  Val Recall:     {metrics["val_recall"]:.4f}')
    print(f'  Val F1:         {metrics["val_f1"]:.4f}')

    return model, metrics
