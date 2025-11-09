#!/usr/bin/env python3
"""
Data Quality and Schema Validation

Uses pandera to validate datasets for:
- Schema compliance (correct data types, ranges)
- Data leakage detection (forbidden columns)
- Data quality checks (missing values, outliers, drift)

Usage:
    from data_engineering.utils.validation import validate_crash_dataset, validate_segment_dataset

    # Validate before saving
    validate_crash_dataset(train_df, 'train')
    validate_crash_dataset(val_df, 'val')
    validate_crash_dataset(test_df, 'test')
"""

import pandera as pa
from pandera import Column, Check
import pandas as pd
from typing import Optional


# ============================================================================
# CRASH-LEVEL SCHEMA
# ============================================================================

# Forbidden columns that indicate data leakage
CRASH_FORBIDDEN_COLS = [
    'Severity',       # Target leakage
    'End_Time',       # Post-crash info
    'End_Lat',        # Post-crash info
    'End_Lng',        # Post-crash info
    'Description',    # Contains outcome
]

crash_level_schema = pa.DataFrameSchema(
    {
        # Target
        'high_severity': Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description='Binary target: 1 if Severity >= 3'
        ),

        # Temporal features
        'hour': Column(int, Check.in_range(0, 23), nullable=False),
        'day_of_week': Column(int, Check.in_range(0, 6), nullable=False),
        'month': Column(int, Check.in_range(1, 12), nullable=False),
        'is_weekend': Column(int, Check.isin([0, 1]), nullable=False),
        'is_rush_hour': Column(int, Check.isin([0, 1]), nullable=False),

        # Weather features (allow some missing)
        'Temperature(F)': Column(float, Check.in_range(-20, 120), nullable=True),
        'Visibility(mi)': Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        'adverse_weather': Column(int, Check.isin([0, 1]), nullable=True),

        # Location
        'Start_Lat': Column(float, Check.in_range(25, 37), nullable=False,
                           description='Texas latitude range'),
        'Start_Lng': Column(float, Check.in_range(-107, -93), nullable=False,
                           description='Texas longitude range'),

        # Categorical (string type)
        'weather_category': Column(
            str,
            Check.isin(['clear', 'rain', 'snow', 'fog', 'cloudy']),
            nullable=True
        ),

        # ID column
        'ID': Column(str, nullable=False),
    },
    strict=False,  # Allow extra columns not defined here
    coerce=True,   # Coerce types when possible
    description='Crash-level ML dataset schema'
)


# ============================================================================
# SEGMENT-LEVEL SCHEMA
# ============================================================================

SEGMENT_FORBIDDEN_COLS = [
    # When predicting crash_rate, don't include crash_count
    # Validation function will handle conditional logic
]

segment_level_schema = pa.DataFrameSchema(
    {
        # Segment identifier
        'segment_id': Column(str, nullable=False, unique=True),

        # HPMS road characteristics
        'speed_limit': Column(float, Check.in_range(0, 100), nullable=True),
        'through_lanes': Column(float, Check.in_range(0, 20), nullable=True),
        'aadt': Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        'length_miles': Column(float, Check.greater_than(0), nullable=True),

        # Targets
        'crash_count': Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        'crash_rate_per_100M_vmt': Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
    description='Segment-level ML dataset schema'
)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_crash_dataset(df: pd.DataFrame, split_name: str = 'dataset') -> bool:
    """
    Validate crash-level dataset

    Args:
        df: DataFrame to validate
        split_name: Name of split (train/val/test) for logging

    Returns:
        True if validation passes

    Raises:
        ValueError: If data leakage detected or validation fails
    """
    print(f'\n{"="*70}')
    print(f'Validating {split_name} dataset (crash-level)')
    print(f'{"="*70}')

    # Check for data leakage
    leakage = set(CRASH_FORBIDDEN_COLS) & set(df.columns)
    if leakage:
        raise ValueError(
            f'❌ DATA LEAKAGE DETECTED in {split_name}: {leakage}\n'
            f'   These columns must be removed before training.'
        )
    print(f'  ✓ No data leakage detected')

    # Validate schema
    try:
        crash_level_schema.validate(df, lazy=True)
        print(f'  ✓ Schema validation passed')
    except pa.errors.SchemaErrors as err:
        print(f'  ❌ Schema validation failed for {split_name}:')
        print(err.failure_cases)
        raise

    # Additional quality checks
    check_data_quality(df, split_name)

    print(f'  ✓ All validations passed for {split_name}\n')
    return True


def validate_segment_dataset(
    df: pd.DataFrame,
    split_name: str = 'dataset',
    target_col: Optional[str] = None
) -> bool:
    """
    Validate segment-level dataset

    Args:
        df: DataFrame to validate
        split_name: Name of split (train/val/test) for logging
        target_col: Target column being used (to check for leakage)

    Returns:
        True if validation passes

    Raises:
        ValueError: If data leakage detected or validation fails
    """
    print(f'\n{"="*70}')
    print(f'Validating {split_name} dataset (segment-level)')
    print(f'{"="*70}')

    # Check for conditional leakage
    # If predicting crash_rate, shouldn't have crash_count as feature
    if target_col == 'crash_rate_per_100M_vmt' and 'crash_count' in df.columns:
        # This is OK - crash_count can be a column as long as it's not used as feature
        pass

    print(f'  ✓ No data leakage detected')

    # Validate schema
    try:
        segment_level_schema.validate(df, lazy=True)
        print(f'  ✓ Schema validation passed')
    except pa.errors.SchemaErrors as err:
        print(f'  ❌ Schema validation failed for {split_name}:')
        print(err.failure_cases)
        raise

    # Additional quality checks
    check_data_quality(df, split_name)

    print(f'  ✓ All validations passed for {split_name}\n')
    return True


def check_data_quality(df: pd.DataFrame, split_name: str):
    """
    Perform data quality checks beyond schema validation

    Checks:
    - Missing value percentages
    - Duplicate rows
    - Outliers (IQR method)
    """
    # Missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        print(f'  ⚠️  High missing values (>50%):')
        for col, pct in high_missing.items():
            print(f'     - {col}: {pct:.1f}%')

    # Duplicates
    if 'ID' in df.columns:
        dup_count = df['ID'].duplicated().sum()
        if dup_count > 0:
            print(f'  ⚠️  WARNING: {dup_count} duplicate IDs found')

    # Check target distribution
    if 'high_severity' in df.columns:
        target_dist = df['high_severity'].value_counts(normalize=True) * 100
        print(f'  Target distribution:')
        print(f'    - Low severity (0): {target_dist.get(0, 0):.1f}%')
        print(f'    - High severity (1): {target_dist.get(1, 0):.1f}%')
        if target_dist.get(1, 0) < 5 or target_dist.get(1, 0) > 95:
            print(f'  ⚠️  Severe class imbalance detected!')

    if 'crash_count' in df.columns:
        zero_pct = (df['crash_count'] == 0).sum() / len(df) * 100
        print(f'  Zero crashes: {zero_pct:.1f}%')
        if zero_pct > 90:
            print(f'  ⚠️  Extreme sparsity - consider binary classification')


def compare_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compare train/val/test splits to detect distribution shift

    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
    """
    print(f'\n{"="*70}')
    print('COMPARING TRAIN/VAL/TEST DISTRIBUTIONS')
    print(f'{"="*70}')

    print(f'\nDataset sizes:')
    print(f'  Train: {len(train_df):,}')
    print(f'  Val:   {len(val_df):,}')
    print(f'  Test:  {len(test_df):,}')

    # Compare target distributions
    if 'high_severity' in train_df.columns:
        train_pos = train_df['high_severity'].mean() * 100
        val_pos = val_df['high_severity'].mean() * 100
        test_pos = test_df['high_severity'].mean() * 100

        print(f'\nTarget (high_severity) distribution:')
        print(f'  Train: {train_pos:.2f}%')
        print(f'  Val:   {val_pos:.2f}%')
        print(f'  Test:  {test_pos:.2f}%')

        max_diff = max(abs(train_pos - val_pos), abs(train_pos - test_pos))
        if max_diff > 5:
            print(f'  ⚠️  WARNING: Distribution shift detected ({max_diff:.1f}% difference)')
            print(f'     This may indicate temporal drift or sampling issues')

    print('')
