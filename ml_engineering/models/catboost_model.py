#!/usr/bin/env python3
"""
CatBoost Models

CatBoost implementations for crash prediction with automatic class imbalance
handling and native categorical feature support.

Usage:
    from ml_engineering.models.catboost_model import train_catboost_classifier

    model, metrics = train_catboost_classifier(
        X_train, y_train,
        X_val, y_val,
        categorical_features=['weather_category', 'region'],
        iterations=1000
    )
"""

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Tuple, Dict, Optional, List


def train_catboost_classifier(
    X_train, y_train,
    X_val, y_val,
    categorical_features: Optional[List[str]] = None,
    iterations: int = 1000,
    depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    **kwargs
) -> Tuple[CatBoostClassifier, Dict[str, float]]:
    """
    Train CatBoost classifier with early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        categorical_features: List of categorical column names
        iterations: Maximum number of boosting rounds
        depth: Maximum tree depth
        learning_rate: Learning rate
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Print every N iterations (0 for silent)
        **kwargs: Additional CatBoost parameters

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: CatBoost Classifier')
    print(f'{"="*70}')

    # Calculate class distribution
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    class_ratio = neg_count / pos_count

    print(f'  Class distribution: {neg_count:,} negative, {pos_count:,} positive')
    print(f'  Class ratio: {class_ratio:.2f}')
    print(f'  Max boosting rounds: {iterations}')
    print(f'  Early stopping: {early_stopping_rounds} rounds')
    if categorical_features:
        print(f'  Categorical features: {len(categorical_features)}')
    print()

    # Get categorical feature indices if provided
    cat_features = None
    if categorical_features is not None:
        if hasattr(X_train, 'columns'):
            # DataFrame
            cat_features = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
        else:
            # Assume categorical_features are indices
            cat_features = categorical_features

    # Default parameters
    params = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'early_stopping_rounds': early_stopping_rounds,
        'auto_class_weights': 'Balanced',  # Handle class imbalance
        'random_seed': 42,
        'verbose': verbose,
        **kwargs
    }

    # Create and train model
    model = CatBoostClassifier(**params)

    print('Training...')
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    # Get best iteration
    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else iterations

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


def train_catboost_regressor(
    X_train, y_train,
    X_val, y_val,
    categorical_features: Optional[List[str]] = None,
    iterations: int = 1000,
    depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    **kwargs
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    """
    Train CatBoost regressor with early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        categorical_features: List of categorical column names
        iterations: Maximum number of boosting rounds
        depth: Maximum tree depth
        learning_rate: Learning rate
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Print every N iterations (0 for silent)
        **kwargs: Additional CatBoost parameters

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: CatBoost Regressor')
    print(f'{"="*70}')

    # Get categorical feature indices if provided
    cat_features = None
    if categorical_features is not None:
        if hasattr(X_train, 'columns'):
            cat_features = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
        else:
            cat_features = categorical_features

    params = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': early_stopping_rounds,
        'random_seed': 42,
        'verbose': verbose,
        **kwargs
    }

    model = CatBoostRegressor(**params)

    print('Training...')
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
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
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else iterations
    }

    print('\nPerformance:')
    print(f'  Train RMSE: {metrics["train_rmse"]:.4f}')
    print(f'  Train MAE:  {metrics["train_mae"]:.4f}')
    print(f'  Train R²:   {metrics["train_r2"]:.4f}')
    print(f'  Val RMSE:   {metrics["val_rmse"]:.4f}')
    print(f'  Val MAE:    {metrics["val_mae"]:.4f}')
    print(f'  Val R²:     {metrics["val_r2"]:.4f}')

    return model, metrics
