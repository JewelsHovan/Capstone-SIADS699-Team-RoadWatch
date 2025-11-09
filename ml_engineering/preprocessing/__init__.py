"""
ML Preprocessing Module

Provides sklearn Pipelines and feature schema definitions for reproducible
preprocessing across train/val/test splits.
"""

from .pipelines import (
    create_crash_classifier_pipeline,
    create_crash_regressor_pipeline,
    get_feature_names
)

from .feature_lists import (
    CRASH_NUMERIC_FEATURES,
    CRASH_CATEGORICAL_FEATURES,
    CRASH_TARGET,
    SEGMENT_NUMERIC_FEATURES,
    SEGMENT_CATEGORICAL_FEATURES,
    SEGMENT_TARGET_CRASH_RATE,
    validate_features,
    check_for_leakage
)

__all__ = [
    'create_crash_classifier_pipeline',
    'create_crash_regressor_pipeline',
    'get_feature_names',
    'CRASH_NUMERIC_FEATURES',
    'CRASH_CATEGORICAL_FEATURES',
    'CRASH_TARGET',
    'SEGMENT_NUMERIC_FEATURES',
    'SEGMENT_CATEGORICAL_FEATURES',
    'SEGMENT_TARGET_CRASH_RATE',
    'validate_features',
    'check_for_leakage',
]
