"""
ML Utilities Module

Model persistence, MLflow tracking, and shared utilities for ML tasks
"""

# Make MLflow optional
try:
    from .persistence import (
        save_model_artifact,
        load_model_artifact,
        find_latest_artifact,
        smoke_test_artifact
    )

    from .tracking import (
        start_experiment,
        log_model_run,
        log_feature_importance,
        compare_runs
    )

    __all__ = [
        'save_model_artifact',
        'load_model_artifact',
        'find_latest_artifact',
        'smoke_test_artifact',
        'start_experiment',
        'log_model_run',
        'log_feature_importance',
        'compare_runs',
    ]
except ImportError:
    # MLflow not installed - provide fallback
    print("⚠️  MLflow not installed. Install with: pip install mlflow")
    __all__ = []
