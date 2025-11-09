#!/usr/bin/env python3
"""
Model Persistence and Artifact Management

Saves trained models, preprocessors, and metadata for deployment and reproducibility.
Integrates with MLflow for experiment tracking.

Usage:
    from ml_engineering.utils.persistence import save_model_artifact, load_model_artifact

    # After training
    artifact_path = save_model_artifact(
        pipeline=trained_pipeline,
        feature_cols=['aadt', 'hour', 'weather_category'],
        metrics={'val_auc': 0.85, 'val_f1': 0.42},
        model_name='random_forest_baseline',
        run_id=mlflow.active_run().info.run_id  # Optional
    )

    # For inference
    pipeline, metadata = load_model_artifact(artifact_path)
    predictions = pipeline.predict(X_new)
"""

import joblib
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List


def save_model_artifact(
    pipeline,
    feature_cols: List[str],
    metrics: Dict[str, float],
    model_name: str,
    run_id: Optional[str] = None,
    output_dir: str = 'models/artifacts',
    log_to_mlflow: bool = True
) -> Path:
    """
    Save model pipeline and metadata together

    Args:
        pipeline: Fitted sklearn Pipeline
        feature_cols: List of feature column names used
        metrics: Dict of metric names and values (e.g., {'val_auc': 0.85})
        model_name: Human-readable model name (e.g., 'random_forest_baseline')
        run_id: MLflow run ID (if using MLflow tracking)
        output_dir: Directory to save artifacts
        log_to_mlflow: Whether to log artifact to MLflow

    Returns:
        Path to saved model artifact directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f'{model_name}_{timestamp}'
    artifact_dir = output_dir / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline (includes preprocessor + model)
    pipeline_path = artifact_dir / 'pipeline.pkl'
    joblib.dump(pipeline, pipeline_path)

    # Extract model type
    if hasattr(pipeline, 'named_steps'):
        if 'classifier' in pipeline.named_steps:
            model_obj = pipeline.named_steps['classifier']
        elif 'regressor' in pipeline.named_steps:
            model_obj = pipeline.named_steps['regressor']
        else:
            model_obj = pipeline
    else:
        model_obj = pipeline

    model_type = type(model_obj).__name__

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'model_type': model_type,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'metrics': metrics,
        'run_id': run_id,
        'sklearn_version': joblib.__version__,
    }

    metadata_path = artifact_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create a simple README
    readme_path = artifact_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"# {model_name}\n\n")
        f.write(f"**Created**: {timestamp}\n\n")
        f.write(f"**Model**: {model_type}\n\n")
        f.write(f"**Features**: {len(feature_cols)}\n\n")
        f.write("## Metrics\n\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"- {metric_name}: {metric_value:.4f}\n")
        f.write("\n## Usage\n\n")
        f.write("```python\n")
        f.write("from ml_engineering.utils.persistence import load_model_artifact\n\n")
        f.write(f"pipeline, metadata = load_model_artifact('{artifact_dir}')\n")
        f.write("predictions = pipeline.predict(X_new)\n")
        f.write("```\n")

    print(f'\n✓ Saved model artifact to {artifact_dir}/')
    print(f'  - pipeline.pkl ({pipeline_path.stat().st_size / 1024:.1f} KB)')
    print(f'  - metadata.json')
    print(f'  - README.md')

    # Log to MLflow if requested
    if log_to_mlflow and mlflow.active_run():
        try:
            mlflow.sklearn.log_model(pipeline, 'model')
            mlflow.log_artifact(str(metadata_path))
            print(f'  ✓ Logged to MLflow run: {mlflow.active_run().info.run_id}')
        except Exception as e:
            print(f'  ⚠️  Failed to log to MLflow: {e}')

    return artifact_dir


def load_model_artifact(artifact_path: Path) -> Tuple[Any, Dict]:
    """
    Load a saved model artifact

    Args:
        artifact_path: Path to artifact directory

    Returns:
        Tuple of (pipeline, metadata_dict)
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    # Load pipeline
    pipeline_path = artifact_path / 'pipeline.pkl'
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)

    # Load metadata
    metadata_path = artifact_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    print(f'✓ Loaded model artifact from {artifact_path}')
    if metadata:
        print(f'  Model: {metadata.get("model_type", "unknown")}')
        print(f'  Features: {metadata.get("n_features", "unknown")}')
        print(f'  Created: {metadata.get("timestamp", "unknown")}')

    return pipeline, metadata


def find_latest_artifact(model_name: str, artifacts_dir: str = 'models/artifacts') -> Optional[Path]:
    """
    Find the most recent artifact for a given model name

    Args:
        model_name: Model name prefix to search for
        artifacts_dir: Directory containing artifacts

    Returns:
        Path to latest artifact, or None if not found
    """
    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.exists():
        return None

    # Find all matching artifacts
    matching = list(artifacts_dir.glob(f'{model_name}_*'))

    if not matching:
        return None

    # Sort by timestamp (embedded in directory name)
    matching.sort(reverse=True)

    return matching[0]


def smoke_test_artifact(artifact_path: Path, X_test, y_test) -> Dict[str, float]:
    """
    Load and test a saved artifact on test data

    Args:
        artifact_path: Path to artifact
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict of test metrics
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    pipeline, metadata = load_model_artifact(artifact_path)

    print('\nRunning smoke test...')

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred)
    }

    # Add AUC if classifier
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics['test_auc'] = roc_auc_score(y_test, y_proba)
        test_metrics['test_f1'] = f1_score(y_test, y_pred)

    print('Test metrics:')
    for metric, value in test_metrics.items():
        print(f'  {metric}: {value:.4f}')

    return test_metrics
