"""
Feature preprocessing utilities for ML models.

This module converts app features to the format expected by trained models.
Matches the 32-feature schema from ml_engineering/preprocessing/feature_lists.py
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


def preprocess_crash_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess crash features for ML model prediction.

    Converts app features (12 simple features) to the 32 features expected by trained models.
    Uses sensible defaults and derives categorical encodings for features not available in real-time.

    Training Feature Schema (32 features):
    - 22 numeric: temporal, weather, location, traffic/roadway
    - 10 categorical: weather categories, location zones, road categories

    Args:
        features: Raw feature dictionary from app

    Returns:
        Feature dictionary matching training schema (will be converted to DataFrame)
    """
    processed = {}

    # ============================================================================
    # TEMPORAL FEATURES (5 numeric)
    # ============================================================================
    hour = safe_numeric_convert(features.get('hour', 12))
    processed['hour'] = hour
    processed['day_of_week'] = safe_numeric_convert(features.get('day_of_week', 0))
    processed['month'] = safe_numeric_convert(features.get('month', 6))
    processed['is_weekend'] = safe_numeric_convert(features.get('is_weekend', 0))
    processed['is_rush_hour'] = safe_numeric_convert(features.get('is_rush_hour', 0))

    # ============================================================================
    # WEATHER FEATURES (7 numeric) - Use training feature names with (units)
    # ============================================================================
    temp_f = safe_numeric_convert(features.get('temperature', 70))
    visibility_mi = safe_numeric_convert(features.get('visibility', 10))

    processed['Temperature(F)'] = temp_f
    processed['Visibility(mi)'] = visibility_mi
    processed['Pressure(in)'] = safe_numeric_convert(features.get('pressure', 30.0))
    processed['Humidity(%)'] = safe_numeric_convert(features.get('humidity', 60))
    processed['Wind_Speed(mph)'] = safe_numeric_convert(features.get('wind_speed', 5))
    processed['adverse_weather'] = safe_numeric_convert(features.get('adverse_weather', 0))
    processed['low_visibility'] = safe_numeric_convert(features.get('low_visibility', 0))

    # ============================================================================
    # LOCATION FEATURES (6 numeric)
    # ============================================================================
    # Coordinates (used by older models, despite overfitting concerns)
    lat = safe_numeric_convert(features.get('lat', 30.27))  # Austin default
    lng = safe_numeric_convert(features.get('lng', -97.74))  # Austin default
    processed['Start_Lat'] = lat
    processed['Start_Lng'] = lng

    # Urban/rural indicator (binary)
    # Derive from urban_id if available, else from city_size_category
    urban_id = features.get('urban_id', 0)
    city_size = features.get('city_size_category', 'medium')
    processed['is_urban'] = 1 if (urban_id and urban_id > 0) or city_size == 'large' else 0

    processed['is_major_city'] = safe_numeric_convert(features.get('is_major_city', 0))
    processed['is_junction'] = safe_numeric_convert(features.get('is_junction', 0))
    processed['Distance(mi)'] = safe_numeric_convert(features.get('distance', 0.5))

    # ============================================================================
    # TRAFFIC & ROADWAY FEATURES (7 numeric) - From HPMS or user input
    # ============================================================================
    # Use HPMS features if available, otherwise use user-provided values or defaults
    processed['aadt'] = safe_numeric_convert(
        features.get('aadt', features.get('hpms_aadt', 15000))
    )
    processed['distance_to_aadt_m'] = safe_numeric_convert(
        features.get('distance_to_aadt_m', 50)
    )

    # HPMS road characteristics (prefixed with hpms_)
    processed['hpms_speed_limit'] = safe_numeric_convert(
        features.get('hpms_speed_limit', features.get('speed_limit', 45))
    )
    processed['hpms_lanes'] = safe_numeric_convert(
        features.get('hpms_lanes', features.get('through_lanes', 2))
    )
    processed['hpms_functional_class'] = safe_numeric_convert(
        features.get('hpms_functional_class', features.get('f_system', 1))
    )
    processed['hpms_aadt'] = safe_numeric_convert(
        features.get('hpms_aadt', processed['aadt'])
    )
    processed['hpms_distance_m'] = safe_numeric_convert(
        features.get('hpms_distance_m', 50)
    )

    # ============================================================================
    # CATEGORICAL FEATURES (10) - Derive from numeric features
    # ============================================================================

    # Weather category (clear, rain, snow, fog, cloudy)
    if processed['adverse_weather'] == 1:
        processed['weather_category'] = 'rain'
    elif processed['low_visibility'] == 1:
        processed['weather_category'] = 'fog'
    else:
        processed['weather_category'] = 'clear'

    # Temperature category (freezing, cold, mild, warm, hot)
    if temp_f < 32:
        processed['temp_category'] = 'freezing'
    elif temp_f < 50:
        processed['temp_category'] = 'cold'
    elif temp_f < 70:
        processed['temp_category'] = 'mild'
    elif temp_f < 85:
        processed['temp_category'] = 'warm'
    else:
        processed['temp_category'] = 'hot'

    # Lat zone (south, south_central, central, north) - using lat/lng from above
    if lat < 29:
        processed['lat_zone'] = 'south'
    elif lat < 31:
        processed['lat_zone'] = 'south_central'
    elif lat < 33:
        processed['lat_zone'] = 'central'
    else:
        processed['lat_zone'] = 'north'

    # Lng zone (west, west_central, central, east)
    if lng < -100:
        processed['lng_zone'] = 'west'
    elif lng < -98:
        processed['lng_zone'] = 'west_central'
    elif lng < -96:
        processed['lng_zone'] = 'central'
    else:
        processed['lng_zone'] = 'east'

    # Region (combined lat_lng zone)
    processed['region'] = f"{processed['lat_zone']}_{processed['lng_zone']}"

    # City size category (large, medium, small)
    processed['city_size_category'] = features.get('city_size_category', 'medium')

    # County (top 20 counties by crash frequency + 'other')
    processed['county_top20'] = features.get('county', 'other')

    # Speed category (low, medium, high, highway)
    speed = processed['hpms_speed_limit']
    if speed < 35:
        processed['speed_category'] = 'low'
    elif speed < 50:
        processed['speed_category'] = 'medium'
    elif speed < 65:
        processed['speed_category'] = 'high'
    else:
        processed['speed_category'] = 'highway'

    # Lane category (narrow, standard, wide)
    lanes = processed['hpms_lanes']
    if lanes <= 1:
        processed['lane_category'] = 'narrow'
    elif lanes <= 3:
        processed['lane_category'] = 'standard'
    else:
        processed['lane_category'] = 'wide'

    # Road class (functional classification)
    func_class = int(processed['hpms_functional_class'])
    if func_class == 1:
        processed['road_class'] = 'interstate'
    elif func_class <= 3:
        processed['road_class'] = 'principal_arterial'
    elif func_class <= 5:
        processed['road_class'] = 'minor_arterial'
    else:
        processed['road_class'] = 'collector'

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
    """
    Get expected feature order for crash severity model.

    NOTE: This is for baseline model compatibility only.
    Trained models use sklearn pipelines that handle feature ordering internally.
    """
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
