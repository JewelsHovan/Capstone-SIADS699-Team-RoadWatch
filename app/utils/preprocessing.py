"""
Feature preprocessing utilities for ML models.

This module contains all feature engineering and preprocessing logic
used by both baseline and trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def safe_numeric_convert(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to numeric, returning default if conversion fails.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Numeric value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def preprocess_crash_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Preprocess crash features for severity prediction.

    Expected features:
    - hour: 0-23
    - day_of_week: 0-6
    - is_weekend: 0/1
    - is_rush_hour: 0/1
    - temperature: Fahrenheit
    - visibility: Miles
    - adverse_weather: 0/1
    - low_visibility: 0/1
    - speed_limit: MPH
    - through_lanes: Count
    - f_system: Functional class
    - aadt: Annual Average Daily Traffic

    Args:
        features: Raw feature dictionary

    Returns:
        Preprocessed feature dictionary with all values as floats
    """
    processed = {}

    # Temporal features
    processed['hour'] = safe_numeric_convert(features.get('hour', 12))
    processed['day_of_week'] = safe_numeric_convert(features.get('day_of_week', 0))
    processed['is_weekend'] = safe_numeric_convert(features.get('is_weekend', 0))
    processed['is_rush_hour'] = safe_numeric_convert(features.get('is_rush_hour', 0))

    # Weather features
    processed['temperature'] = safe_numeric_convert(features.get('temperature', 70))
    processed['visibility'] = safe_numeric_convert(features.get('visibility', 10))
    processed['adverse_weather'] = safe_numeric_convert(features.get('adverse_weather', 0))
    processed['low_visibility'] = safe_numeric_convert(features.get('low_visibility', 0))

    # Roadway features
    processed['speed_limit'] = safe_numeric_convert(features.get('speed_limit', 30))
    processed['through_lanes'] = safe_numeric_convert(features.get('through_lanes', 2))
    processed['f_system'] = safe_numeric_convert(features.get('f_system', 1))
    processed['aadt'] = safe_numeric_convert(features.get('aadt', 10000))

    return processed


def preprocess_segment_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Preprocess segment features for risk prediction.

    Expected features:
    - speed_limit: MPH
    - through_lanes: Count
    - f_system: Functional class (1-7)
    - urban_id: Urban code (0 or 5-digit)
    - aadt: Annual Average Daily Traffic
    - speed_x_aadt: Interaction feature
    - fsystem_x_urban: Interaction feature
    - lanes_x_aadt: Interaction feature

    Args:
        features: Raw feature dictionary

    Returns:
        Preprocessed feature dictionary with all values as floats
    """
    processed = {}

    # Base features
    processed['speed_limit'] = safe_numeric_convert(features.get('speed_limit', 30))
    processed['through_lanes'] = safe_numeric_convert(features.get('through_lanes', 2))
    processed['f_system'] = safe_numeric_convert(features.get('f_system', 1))
    processed['urban_id'] = safe_numeric_convert(features.get('urban_id', 0))
    processed['aadt'] = safe_numeric_convert(features.get('aadt', 10000))

    # Interaction features
    processed['speed_x_aadt'] = processed['speed_limit'] * processed['aadt']
    processed['fsystem_x_urban'] = processed['f_system'] * (1 if processed['urban_id'] > 0 else 0)
    processed['lanes_x_aadt'] = processed['through_lanes'] * processed['aadt']

    return processed


def create_feature_vector(
    features: Dict[str, float],
    feature_order: List[str]
) -> np.ndarray:
    """
    Create ordered feature vector for model input.

    Args:
        features: Preprocessed feature dictionary
        feature_order: List of feature names in expected order

    Returns:
        NumPy array with features in correct order
    """
    return np.array([features[feat] for feat in feature_order])


def get_crash_feature_order() -> List[str]:
    """Get expected feature order for crash severity model."""
    return [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'temperature', 'visibility', 'adverse_weather', 'low_visibility',
        'speed_limit', 'through_lanes', 'f_system', 'aadt'
    ]


def get_segment_feature_order() -> List[str]:
    """Get expected feature order for segment risk model."""
    return [
        'speed_limit', 'through_lanes', 'f_system', 'urban_id', 'aadt',
        'speed_x_aadt', 'fsystem_x_urban', 'lanes_x_aadt'
    ]
