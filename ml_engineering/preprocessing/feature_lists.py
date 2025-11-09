#!/usr/bin/env python3
"""
Feature Schema Definitions

Documents expected features for crash-level and segment-level datasets.
Used by pipelines to ensure consistency across train/val/test.

These lists should be updated whenever the data engineering pipeline changes.
"""

# ============================================================================
# CRASH-LEVEL FEATURES (one row per crash)
# ============================================================================

CRASH_NUMERIC_FEATURES = [
    # Temporal
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'is_rush_hour',

    # Weather (numeric)
    'Temperature(F)',
    'Visibility(mi)',
    'Pressure(in)',
    'Humidity(%)',
    'Wind_Speed(mph)',
    'adverse_weather',
    'low_visibility',

    # Location (numeric indicators)
    'is_major_city',
    'is_junction',
    'Distance(mi)',

    # Traffic (if available)
    'aadt',
    'distance_to_aadt_m',

    # HPMS road characteristics (numeric - if available)
    'hpms_speed_limit',
    'hpms_lanes',
    'hpms_functional_class',
    'hpms_aadt',
    'hpms_distance_m',
]

CRASH_CATEGORICAL_FEATURES = [
    # Weather
    'weather_category',  # clear, rain, snow, fog, cloudy
    'temp_category',     # freezing, cold, mild, warm, hot

    # Location (coarse regions - prevent overfitting)
    'lat_zone',          # south, south_central, central, north
    'lng_zone',          # west, west_central, central, east
    'region',            # combined lat_lng zone (e.g., "south_west")
    'city_size_category', # large, medium, small
    'county_top20',      # top 20 counties by crash frequency + 'other'

    # HPMS road characteristics (categorical - if available)
    'speed_category',    # low, medium, high, highway
    'lane_category',     # narrow, standard, wide
    'road_class',        # functional classification
]

# NOTE: Start_Lat and Start_Lng are NO LONGER USED
# They caused severe overfitting (top 2 features, 45% importance combined)
# Replaced with coarse region features that generalize better

CRASH_TARGET = 'high_severity'  # Binary: 0 or 1

# Features that should NEVER be used (data leakage)
CRASH_FORBIDDEN_FEATURES = [
    'Severity',           # Direct target leakage
    'End_Time',           # Post-crash information
    'End_Lat',            # Post-crash information
    'End_Lng',            # Post-crash information
    'Description',        # Contains outcome information
    # Note: 'ID' and 'year' are allowed - ID for tracking, year as drift marker
]


# ============================================================================
# SEGMENT-LEVEL FEATURES (one row per road segment)
# ============================================================================

SEGMENT_NUMERIC_FEATURES = [
    # HPMS road characteristics
    'speed_limit',
    'through_lanes',
    'f_system',
    'urban_id',
    'aadt',
    'length_miles',

    # Derived features
    'speed_x_aadt',
    'lanes_x_aadt',

    # Add more as engineered
]

SEGMENT_CATEGORICAL_FEATURES = [
    # Add categorical road features if available
    # e.g., 'surface_type', 'road_class', etc.
]

# Targets for segment-level prediction
SEGMENT_TARGET_CRASH_COUNT = 'crash_count'  # Raw count (biased toward busy roads)
SEGMENT_TARGET_CRASH_RATE = 'crash_rate_per_100M_vmt'  # Exposure-normalized (preferred)
SEGMENT_TARGET_HIGH_SEVERITY = 'high_severity_count'

# Features that should NEVER be used (data leakage)
SEGMENT_FORBIDDEN_FEATURES = [
    'crash_count',                      # If predicting crash_rate
    'crash_rate_per_100M_vmt',          # If predicting crash_count
    'high_severity_count',              # Always forbidden
    'high_severity_rate_per_100M_vmt',  # Always forbidden
    'segment_id',                       # Not predictive
    'has_crashes',                      # Derived from target
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_features(df, feature_list, dataset_type='crash'):
    """
    Validate that expected features exist in DataFrame

    Args:
        df: pandas DataFrame
        feature_list: List of expected feature names
        dataset_type: 'crash' or 'segment'

    Returns:
        Tuple of (available_features, missing_features)
    """
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]

    print(f'\n{dataset_type.upper()} Features Validation:')
    print(f'  Available: {len(available)}/{len(feature_list)}')
    if missing:
        print(f'  Missing: {missing}')

    return available, missing


def check_for_leakage(df, dataset_type='crash'):
    """
    Check if DataFrame contains forbidden features

    Args:
        df: pandas DataFrame
        dataset_type: 'crash' or 'segment'

    Raises:
        ValueError if leakage features detected
    """
    forbidden = (
        CRASH_FORBIDDEN_FEATURES if dataset_type == 'crash'
        else SEGMENT_FORBIDDEN_FEATURES
    )

    leakage = set(forbidden) & set(df.columns)

    if leakage:
        raise ValueError(
            f'DATA LEAKAGE DETECTED in {dataset_type} dataset: {leakage}\n'
            f'These features must be removed before training.'
        )

    print(f'  ✓ No data leakage detected in {dataset_type} dataset')
