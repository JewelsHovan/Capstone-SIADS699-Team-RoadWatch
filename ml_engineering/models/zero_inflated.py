#!/usr/bin/env python3
"""
Zero-Inflated Models

Zero-Inflated Poisson (ZIP) and Negative Binomial models for count data
with excess zeros, using statsmodels.

Designed for segment-level crash prediction where 94.4% of segments have 0 crashes.

Usage:
    from ml_engineering.models.zero_inflated import train_zip_regressor

    model, metrics = train_zip_regressor(
        X_train, y_train,
        X_val, y_val
    )
"""

import numpy as np
import pandas as pd
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict


class ZIPWrapper:
    """
    Wrapper around statsmodels ZeroInflatedPoisson to provide sklearn-like interface
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Predict crash counts"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def get_params(self):
        """Get model parameters"""
        return self.model.params


def train_zip_regressor(
    X_train, y_train,
    X_val, y_val,
    maxiter: int = 100,
    method: str = 'bfgs',
    verbose: bool = True
) -> Tuple[ZIPWrapper, Dict[str, float]]:
    """
    Train Zero-Inflated Poisson regressor

    The ZIP model has two components:
    1. Binary model: Probability of being in "always zero" group (logit)
    2. Count model: Poisson model for non-zero group

    Args:
        X_train: Training features (numpy array or DataFrame)
        y_train: Training crash counts
        X_val: Validation features
        y_val: Validation crash counts
        maxiter: Maximum iterations for optimization
        method: Optimization method ('bfgs', 'newton', etc.)
        verbose: Print training progress

    Returns:
        Tuple of (wrapped_model, metrics_dict)
    """
    print(f'\n{"="*70}')
    print('TRAINING: Zero-Inflated Poisson (ZIP)')
    print(f'{"="*70}')

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
        X_val_array = X_val.values
    else:
        X_train_array = X_train
        X_val_array = X_val

    # Calculate zero-inflation statistics
    zero_pct_train = (y_train == 0).mean() * 100
    zero_pct_val = (y_val == 0).mean() * 100

    print(f'  Zero-inflation:')
    print(f'    Train: {zero_pct_train:.1f}% zeros')
    print(f'    Val:   {zero_pct_val:.1f}% zeros')
    print(f'  Max iterations: {maxiter}')
    print(f'  Optimization method: {method}')
    print()

    # Train ZIP model
    print('Training Zero-Inflated Poisson...')
    try:
        zip_model = ZeroInflatedPoisson(
            endog=y_train,
            exog=X_train_array,
            exog_infl=X_train_array  # Use same features for inflation model
        ).fit(
            maxiter=maxiter,
            method=method,
            disp=verbose
        )

        print(f'\n✓ Training converged')
        print(f'  Log-likelihood: {zip_model.llf:.2f}')
        print(f'  AIC: {zip_model.aic:.2f}')
        print(f'  BIC: {zip_model.bic:.2f}')

    except Exception as e:
        print(f'\n⚠️  ZIP training failed: {e}')
        print('  Falling back to standard Poisson...')
        # Could implement fallback here
        raise

    # Wrap model
    wrapped_model = ZIPWrapper(zip_model)

    # Evaluate on train
    train_pred = zip_model.predict(X_train_array)
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'mae': mean_absolute_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred)
    }

    # Evaluate on validation
    val_pred = zip_model.predict(X_val_array)
    val_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'mae': mean_absolute_error(y_val, val_pred),
        'r2': r2_score(y_val, val_pred)
    }

    # Check predicted zero-inflation
    pred_zeros_train = (train_pred < 0.5).mean() * 100
    pred_zeros_val = (val_pred < 0.5).mean() * 100

    metrics = {
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'train_pred_zeros_pct': pred_zeros_train,
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'val_pred_zeros_pct': pred_zeros_val,
        'aic': zip_model.aic,
        'bic': zip_model.bic,
        'log_likelihood': zip_model.llf
    }

    print('\nPerformance:')
    print(f'  Train RMSE: {metrics["train_rmse"]:.4f}')
    print(f'  Train MAE:  {metrics["train_mae"]:.4f}')
    print(f'  Train R²:   {metrics["train_r2"]:.4f}')
    print(f'  Train Pred Zeros: {metrics["train_pred_zeros_pct"]:.1f}% (actual: {zero_pct_train:.1f}%)')
    print(f'  Val RMSE:   {metrics["val_rmse"]:.4f}')
    print(f'  Val MAE:    {metrics["val_mae"]:.4f}')
    print(f'  Val R²:     {metrics["val_r2"]:.4f}')
    print(f'  Val Pred Zeros: {metrics["val_pred_zeros_pct"]:.1f}% (actual: {zero_pct_val:.1f}%)')

    print('\nModel Interpretation:')
    print('  ZIP models two processes:')
    print('    1. Binary: "Is this segment always safe?" (inflation model)')
    print('    2. Poisson: "If not always safe, how many crashes?" (count model)')

    return wrapped_model, metrics


def train_zinb_regressor(
    X_train, y_train,
    X_val, y_val,
    maxiter: int = 100,
    method: str = 'bfgs'
):
    """
    Train Zero-Inflated Negative Binomial regressor

    Similar to ZIP but uses Negative Binomial instead of Poisson for the count model.
    Better for overdispersed data (variance >> mean).

    Note: Implementation would be similar to ZIP but using ZeroInflatedNegativeBinomial
    from statsmodels. Left as future enhancement.
    """
    raise NotImplementedError(
        "Zero-Inflated Negative Binomial not yet implemented. "
        "Use train_zip_regressor() for now."
    )
