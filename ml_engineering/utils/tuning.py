#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna

Automated hyperparameter search for crash prediction models.
Integrates with MLflow for tracking tuning runs.

Usage:
    from ml_engineering.utils.tuning import tune_random_forest, tune_xgboost

    # Tune RandomForest
    best_params = tune_random_forest(
        X_train, y_train,
        n_trials=50,
        cv_folds=3
    )

    # Train final model with best params
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
"""

import optuna
from optuna.samplers import TPESampler
import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Optional, Literal


def tune_random_forest(
    X_train,
    y_train,
    task: Literal['classification', 'regression'] = 'classification',
    n_trials: int = 50,
    cv_folds: int = 3,
    scoring: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Tune RandomForest hyperparameters with Optuna

    Args:
        X_train: Training features
        y_train: Training labels
        task: 'classification' or 'regression'
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric (default: roc_auc for classification, neg_rmse for regression)
        random_state: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        Dict of best hyperparameters
    """
    if scoring is None:
        scoring = 'roc_auc' if task == 'classification' else 'neg_root_mean_squared_error'

    print(f'\n{"="*70}')
    print(f'HYPERPARAMETER TUNING: RandomForest{task.capitalize()}')
    print(f'{"="*70}')
    print(f'  Trials: {n_trials}')
    print(f'  CV folds: {cv_folds}')
    print(f'  Scoring: {scoring}\n')

    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 50, 500, step=50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100, step=10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': random_state,
            'n_jobs': n_jobs
        }

        # Add task-specific params
        if task == 'classification':
            params['class_weight'] = 'balanced_subsample'
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)

        # Cross-validate
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=1)

        return scores.mean()

    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n✓ Tuning complete!')
    print(f'  Best {scoring}: {study.best_value:.4f}')
    print(f'  Best params:')
    for param, value in study.best_params.items():
        print(f'    - {param}: {value}')

    return study.best_params


def tune_xgboost(
    X_train,
    y_train,
    task: Literal['classification', 'regression'] = 'classification',
    n_trials: int = 50,
    cv_folds: int = 3,
    scoring: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters with Optuna

    Args:
        X_train: Training features
        y_train: Training labels
        task: 'classification' or 'regression'
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        random_state: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        Dict of best hyperparameters
    """
    import xgboost as xgb

    if scoring is None:
        scoring = 'roc_auc' if task == 'classification' else 'neg_root_mean_squared_error'

    print(f'\n{"="*70}')
    print(f'HYPERPARAMETER TUNING: XGBoost{task.capitalize()}')
    print(f'{"="*70}')
    print(f'  Trials: {n_trials}')
    print(f'  CV folds: {cv_folds}')
    print(f'  Scoring: {scoring}\n')

    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': random_state,
            'n_jobs': n_jobs
        }

        # Add task-specific params
        if task == 'classification':
            # Calculate scale_pos_weight for imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params['scale_pos_weight'] = scale_pos_weight
            params['eval_metric'] = 'auc'
            model = xgb.XGBClassifier(**params)
        else:
            params['eval_metric'] = 'rmse'
            model = xgb.XGBRegressor(**params)

        # Cross-validate
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=1)

        return scores.mean()

    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n✓ Tuning complete!')
    print(f'  Best {scoring}: {study.best_value:.4f}')
    print(f'  Best params:')
    for param, value in study.best_params.items():
        print(f'    - {param}: {value}')

    return study.best_params
