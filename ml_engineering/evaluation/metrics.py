#!/usr/bin/env python3
"""
Model Evaluation and Calibration

Comprehensive evaluation metrics, calibration, and test set evaluation.

Usage:
    from ml_engineering.evaluation.metrics import evaluate_classifier, calibrate_model

    # Evaluate model
    metrics = evaluate_classifier(model, X_test, y_test, name='Test Set')

    # Calibrate probabilities
    calibrated_model = calibrate_model(model, X_val, y_val)
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Optional, Tuple
import pandas as pd


def evaluate_classifier(
    model,
    X,
    y,
    name: str = 'Dataset',
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive evaluation of binary classifier

    Args:
        model: Trained classifier with predict_proba
        X: Features
        y: True labels
        name: Dataset name for logging
        threshold: Decision threshold (default: 0.5)

    Returns:
        Dict of metrics
    """
    print(f'\n{"="*70}')
    print(f'EVALUATION: {name}')
    print(f'{"="*70}')

    # Predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_proba),
        'brier_score': brier_score_loss(y, y_proba),
        'threshold': threshold
    }

    print(f'\nMetrics:')
    print(f'  Accuracy:     {metrics["accuracy"]:.4f}')
    print(f'  Precision:    {metrics["precision"]:.4f}')
    print(f'  Recall:       {metrics["recall"]:.4f}')
    print(f'  F1 Score:     {metrics["f1"]:.4f}')
    print(f'  AUC-ROC:      {metrics["auc"]:.4f}')
    print(f'  Brier Score:  {metrics["brier_score"]:.4f} (lower is better)')

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f'\nConfusion Matrix:')
    print(f'  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}')
    print(f'  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}')

    # Class distribution
    pred_dist = pd.Series(y_pred).value_counts()
    true_dist = pd.Series(y).value_counts()
    print(f'\nClass Distribution:')
    print(f'  True:      {true_dist.get(0, 0):,} neg, {true_dist.get(1, 0):,} pos ({true_dist.get(1, 0)/len(y)*100:.1f}% positive)')
    print(f'  Predicted: {pred_dist.get(0, 0):,} neg, {pred_dist.get(1, 0):,} pos ({pred_dist.get(1, 0)/len(y_pred)*100:.1f}% positive)')

    return metrics


def evaluate_regressor(
    model,
    X,
    y,
    name: str = 'Dataset'
) -> Dict[str, float]:
    """
    Comprehensive evaluation of regressor

    Args:
        model: Trained regressor
        X: Features
        y: True labels
        name: Dataset name for logging

    Returns:
        Dict of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

    print(f'\n{"="*70}')
    print(f'EVALUATION: {name}')
    print(f'{"="*70}')

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'mape': mean_absolute_percentage_error(y, y_pred) if (y != 0).all() else None
    }

    print(f'\nMetrics:')
    print(f'  RMSE: {metrics["rmse"]:.4f}')
    print(f'  MAE:  {metrics["mae"]:.4f}')
    print(f'  R²:   {metrics["r2"]:.4f}')
    if metrics["mape"] is not None:
        print(f'  MAPE: {metrics["mape"]:.2f}%')

    # Prediction statistics
    print(f'\nPredictions:')
    print(f'  Min:    {y_pred.min():.2f}')
    print(f'  Max:    {y_pred.max():.2f}')
    print(f'  Mean:   {y_pred.mean():.2f}')
    print(f'  Median: {np.median(y_pred):.2f}')

    return metrics


def calibrate_model(
    model,
    X_val,
    y_val,
    method: str = 'isotonic',
    cv: int = 3
):
    """
    Calibrate classifier probabilities using validation set

    Args:
        model: Trained classifier
        X_val: Validation features
        y_val: Validation labels
        method: 'sigmoid' or 'isotonic' (default: isotonic)
        cv: Number of CV folds for calibration

    Returns:
        Calibrated classifier
    """
    print(f'\n{"="*70}')
    print(f'CALIBRATING MODEL ({method})')
    print(f'{"="*70}')

    # Get uncalibrated probabilities
    y_proba_before = model.predict_proba(X_val)[:, 1]
    brier_before = brier_score_loss(y_val, y_proba_before)

    print(f'  Before calibration:')
    print(f'    Brier score: {brier_before:.4f}')

    # Calibrate
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )

    calibrated.fit(X_val, y_val)

    # Get calibrated probabilities
    y_proba_after = calibrated.predict_proba(X_val)[:, 1]
    brier_after = brier_score_loss(y_val, y_proba_after)

    print(f'  After calibration:')
    print(f'    Brier score: {brier_after:.4f}')
    print(f'    Improvement: {((brier_before - brier_after) / brier_before * 100):.1f}%')

    return calibrated


def find_optimal_threshold(
    y_true,
    y_proba,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for a given metric

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    print(f'\n{"="*70}')
    print(f'FINDING OPTIMAL THRESHOLD (maximizing {metric})')
    print(f'{"="*70}')

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f'Unknown metric: {metric}')

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(f'  Optimal threshold: {best_threshold:.2f}')
    print(f'  {metric.capitalize()}: {best_score:.4f}')

    # Show default threshold performance for comparison
    y_pred_default = (y_proba >= 0.5).astype(int)
    if metric == 'f1':
        default_score = f1_score(y_true, y_pred_default, zero_division=0)
    elif metric == 'precision':
        default_score = precision_score(y_true, y_pred_default, zero_division=0)
    elif metric == 'recall':
        default_score = recall_score(y_true, y_pred_default, zero_division=0)
    else:
        default_score = accuracy_score(y_true, y_pred_default)

    print(f'  Default (0.5) {metric}: {default_score:.4f}')
    print(f'  Improvement: {((best_score - default_score) / default_score * 100):.1f}%')

    return best_threshold, best_score


def analyze_calibration(
    y_true,
    y_proba,
    n_bins: int = 10
):
    """
    Analyze calibration curve

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
    """
    print(f'\n{"="*70}')
    print(f'CALIBRATION ANALYSIS')
    print(f'{"="*70}')

    # Get calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')

    print(f'\nCalibration by bin (predicted vs actual probability):')
    for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
        diff = abs(pred - true)
        status = '✓' if diff < 0.1 else '⚠️' if diff < 0.2 else '❌'
        print(f'  {status} Bin {i+1}: Predicted {pred:.3f}, Actual {true:.3f} (diff: {diff:.3f})')

    # Brier score
    brier = brier_score_loss(y_true, y_proba)
    print(f'\nBrier Score: {brier:.4f} (lower is better, perfect = 0)')
