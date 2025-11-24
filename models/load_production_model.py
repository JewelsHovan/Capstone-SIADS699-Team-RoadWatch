#!/usr/bin/env python3
"""
Load Production Models

Simple utility to load production models for crash severity prediction.

Usage:
    from models.load_production_model import load_production_model

    # Load best recall model (for emergency response)
    model, metadata = load_production_model('random_forest')

    # Make predictions
    probabilities = model.predict_proba(X_new)[:, 1]
    threshold = metadata['metrics']['threshold']
    predictions = (probabilities >= threshold).astype(int)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from ml_engineering.utils.persistence import load_model_artifact


def load_production_model(model_type='random_forest'):
    """
    Load a production model

    Args:
        model_type: One of 'random_forest', 'catboost', 'lightgbm'

    Returns:
        (pipeline, metadata) tuple

    Examples:
        # Best recall (emergency response)
        model, metadata = load_production_model('random_forest')

        # Best balanced (resource constrained)
        model, metadata = load_production_model('catboost')

        # Best AUC (high precision validation)
        model, metadata = load_production_model('lightgbm')
    """
    model_paths = {
        'random_forest': 'models/production/random_forest_best_recall',
        'catboost': 'models/production/catboost_best_balanced',
        'lightgbm': 'models/production/lightgbm_best_auc'
    }

    if model_type not in model_paths:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: {list(model_paths.keys())}"
        )

    model_path = Path(model_paths[model_type])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}\n"
            f"Run the training pipeline first."
        )

    # Load model
    pipeline, metadata = load_model_artifact(model_path)

    # Display model info
    print(f"\n{'='*70}")
    print(f"Loaded: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"  Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"  Features: {metadata.get('n_features', 'Unknown')}")
    print(f"  Created: {metadata.get('timestamp', 'Unknown')}")

    if 'metrics' in metadata:
        print(f"\n  Test Metrics:")
        for metric, value in metadata['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")

    print(f"{'='*70}\n")

    return pipeline, metadata


def compare_production_models():
    """
    Compare all production models

    Returns:
        DataFrame with model comparison
    """
    import pandas as pd

    models = ['random_forest', 'catboost', 'lightgbm']
    comparison = []

    for model_type in models:
        try:
            _, metadata = load_production_model(model_type)
            metrics = metadata.get('metrics', {})

            comparison.append({
                'Model': model_type.title(),
                'AUC': metrics.get('auc', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'Threshold': metrics.get('threshold', 0.5)
            })
        except FileNotFoundError:
            print(f"⚠️  {model_type} not found")

    return pd.DataFrame(comparison)


if __name__ == '__main__':
    # Demo: Load and compare all models
    import argparse

    parser = argparse.ArgumentParser(description='Load production crash severity model')
    parser.add_argument('model', choices=['random_forest', 'catboost', 'lightgbm', 'compare'],
                       help='Model to load')
    args = parser.parse_args()

    if args.model == 'compare':
        print("\n=== Production Model Comparison ===")
        df = compare_production_models()
        print(df.to_string(index=False))
    else:
        pipeline, metadata = load_production_model(args.model)
        print(f"\nModel loaded successfully!")
        print(f"Use: pipeline.predict_proba(X_new)[:, 1] for predictions")
