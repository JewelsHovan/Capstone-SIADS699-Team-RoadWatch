#!/usr/bin/env python3
"""
MLflow Experiment Tracking Utilities

Provides wrapper functions for logging experiments, parameters, metrics,
and artifacts to MLflow.

Usage:
    from ml_engineering.utils.tracking import start_experiment, log_model_run

    # Start experiment
    experiment_id = start_experiment('crash_severity_prediction')

    # Train model...

    # Log results
    log_model_run(
        experiment_name='crash_severity_prediction',
        run_name='random_forest_baseline',
        params={'n_estimators': 100, 'max_depth': 15},
        metrics={'val_auc': 0.85, 'val_f1': 0.42},
        model=trained_pipeline,
        tags={'model_type': 'RandomForest', 'dataset': 'crash_level'}
    )
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from pathlib import Path


def start_experiment(experiment_name: str, tracking_uri: Optional[str] = None) -> str:
    """
    Initialize or get existing MLflow experiment

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking server URI (default: local ./mlruns)

    Returns:
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local mlruns directory
        mlruns_dir = Path('mlruns').absolute()
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f'file://{mlruns_dir}')

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'✓ Created new MLflow experiment: {experiment_name} (ID: {experiment_id})')
    else:
        experiment_id = experiment.experiment_id
        print(f'✓ Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})')

    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_model_run(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Log a complete model training run to MLflow

    Args:
        experiment_name: Name of the experiment
        run_name: Name for this specific run
        params: Model hyperparameters (e.g., {'n_estimators': 100})
        metrics: Performance metrics (e.g., {'val_auc': 0.85})
        model: Trained model to log (optional)
        artifacts: Dict of {artifact_name: file_path} to log (optional)
        tags: Additional tags (e.g., {'model_type': 'RF', 'dataset': 'crash'})

    Returns:
        Run ID
    """
    # Ensure experiment exists
    start_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log tags
        if tags:
            for tag_name, tag_value in tags.items():
                mlflow.set_tag(tag_name, tag_value)

        # Log model
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, 'model')
            except Exception as e:
                print(f'⚠️  Failed to log model: {e}')

        # Log artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                try:
                    mlflow.log_artifact(artifact_path, artifact_name)
                except Exception as e:
                    print(f'⚠️  Failed to log artifact {artifact_name}: {e}')

        run_id = run.info.run_id

        print(f'\n✓ Logged run to MLflow:')
        print(f'  Experiment: {experiment_name}')
        print(f'  Run: {run_name}')
        print(f'  Run ID: {run_id}')
        print(f'  Params: {len(params)}')
        print(f'  Metrics: {len(metrics)}')

        return run_id


def log_feature_importance(importance_dict: Dict[str, float], top_n: int = 20):
    """
    Log feature importances to MLflow

    Args:
        importance_dict: Dict of {feature_name: importance_value}
        top_n: Number of top features to log as params
    """
    import pandas as pd

    # Sort by importance
    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Log top N as params
    for i, (feature, importance) in enumerate(sorted_importance[:top_n]):
        mlflow.log_param(f'top_feature_{i+1}', feature)
        mlflow.log_metric(f'importance_{i+1}', importance)

    # Save all importances to CSV and log as artifact
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(sorted_importance, columns=['feature', 'importance'])
        df.to_csv(f.name, index=False)
        mlflow.log_artifact(f.name, 'feature_importance.csv')

    print(f'  ✓ Logged {len(importance_dict)} feature importances')


def compare_runs(experiment_name: str, metric_name: str = 'val_auc', top_n: int = 10):
    """
    Compare runs in an experiment by a specific metric

    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to sort by (default: 'val_auc')
        top_n: Number of top runs to display

    Returns:
        DataFrame of top runs
    """
    import pandas as pd

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f'Experiment "{experiment_name}" not found')
        return None

    # Search runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metrics.{metric_name} DESC']
    )

    if len(runs) == 0:
        print(f'No runs found in experiment "{experiment_name}"')
        return None

    # Select relevant columns
    cols_to_show = [
        'run_id',
        'tags.mlflow.runName',
        f'metrics.{metric_name}',
        'start_time'
    ]

    # Add any other metrics
    metric_cols = [c for c in runs.columns if c.startswith('metrics.')]
    cols_to_show.extend([c for c in metric_cols if c not in cols_to_show])

    # Filter to available columns
    cols_to_show = [c for c in cols_to_show if c in runs.columns]

    top_runs = runs[cols_to_show].head(top_n)

    print(f'\nTop {top_n} runs in "{experiment_name}" by {metric_name}:')
    print(top_runs.to_string(index=False))

    return top_runs
