#!/usr/bin/env python3
"""
Build segment-level dataset for work zone risk prediction

This script aggregates crash-level data into road segment-level features,
suitable for predicting risk scores for active work zones.

Input: Crash-level dataset (data/processed/crash_level/train_latest.csv)
Output: Segment-level dataset where each row = road segment + time period

Target variables:
  - crash_rate: crashes per segment per time period
  - severity_rate: proportion of high-severity crashes
  - traffic_impact: AADT × severity_rate (proxy for congestion risk)

Usage:
    python build_segment_dataset.py
    python build_segment_dataset.py --input data/processed/crash_level/train_latest.csv
    python build_segment_dataset.py --time-window monthly
    python build_segment_dataset.py --min-crashes 5
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_INPUT = "data/processed/crash_level/train_latest.csv"
DEFAULT_OUTPUT_DIR = "data/processed/segment_level"
TEXAS_CRS = "EPSG:3083"  # Texas Centric Mapping System (meters)

def load_crash_data(crash_file, sample_size=None, verbose=True):
    """
    Load crash-level dataset

    Args:
        crash_file: Path to crash-level CSV
        sample_size: Optional sample size for testing
        verbose: Print progress

    Returns:
        DataFrame with crash data
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 1: LOADING CRASH-LEVEL DATA")
        print("="*80 + "\n")
        print(f"Reading {crash_file}...")

    df = pd.read_csv(crash_file)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        if verbose:
            print(f"  Sampled {len(df):,} crashes")

    # Parse timestamps
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['year'] = df['Start_Time'].dt.year
    df['month'] = df['Start_Time'].dt.month
    df['quarter'] = df['Start_Time'].dt.quarter

    if verbose:
        print(f"  Total crashes: {len(df):,}")
        print(f"  Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"\n✓ Loaded {len(df):,} crashes")

    return df

def create_segment_identifiers(df, verbose=True):
    """
    Create unique segment identifiers by grouping crashes on same road segments

    Strategy:
      - Group by: highway_type, num_lanes, City (approximate segment)
      - Include: geographic bin (lat/lon rounded to ~100m precision)

    Args:
        df: Crash DataFrame
        verbose: Print progress

    Returns:
        DataFrame with segment_id column
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 2: CREATING SEGMENT IDENTIFIERS")
        print("="*80 + "\n")

    # Round coordinates to create geographic bins (~0.001 deg ≈ 100m)
    df['lat_bin'] = (df['Start_Lat'] / 0.001).round() * 0.001
    df['lon_bin'] = (df['Start_Lng'] / 0.001).round() * 0.001

    # Create segment identifier from geographic and road characteristics
    df['segment_id'] = (
        df['City'].astype(str) + "_" +
        df['highway_type'].fillna('unknown').astype(str) + "_" +
        df['num_lanes'].fillna(0).astype(int).astype(str) + "_" +
        df['lat_bin'].round(3).astype(str) + "_" +
        df['lon_bin'].round(3).astype(str)
    )

    if verbose:
        n_segments = df['segment_id'].nunique()
        print(f"  Created {n_segments:,} unique road segments")
        print(f"  Mean crashes per segment: {len(df) / n_segments:.1f}")

        # Distribution
        crashes_per_segment = df.groupby('segment_id').size()
        print(f"\n  Crashes per segment distribution:")
        print(f"    Min: {crashes_per_segment.min()}")
        print(f"    25th: {crashes_per_segment.quantile(0.25):.0f}")
        print(f"    Median: {crashes_per_segment.median():.0f}")
        print(f"    75th: {crashes_per_segment.quantile(0.75):.0f}")
        print(f"    Max: {crashes_per_segment.max()}")

        print(f"\n✓ Segment identifiers created")

    return df

def aggregate_by_segment_and_time(df, time_window='quarterly', min_crashes=1, verbose=True):
    """
    Aggregate crashes by segment and time window

    Args:
        df: Crash DataFrame with segment_id
        time_window: 'monthly', 'quarterly', or 'yearly'
        min_crashes: Minimum crashes per segment-time to include
        verbose: Print progress

    Returns:
        Aggregated DataFrame (segment-time level)
    """

    if verbose:
        print("\n" + "="*80)
        print(f"STEP 3: AGGREGATING BY SEGMENT AND TIME ({time_window.upper()})")
        print("="*80 + "\n")

    # Determine time grouping column
    if time_window == 'monthly':
        time_col = 'year_month'
        df['year_month'] = df['year'].astype(str) + "_" + df['month'].astype(str).str.zfill(2)
        group_cols = ['segment_id', 'year_month', 'year', 'month']
    elif time_window == 'quarterly':
        time_col = 'year_quarter'
        df['year_quarter'] = df['year'].astype(str) + "_Q" + df['quarter'].astype(str)
        group_cols = ['segment_id', 'year_quarter', 'year', 'quarter']
    elif time_window == 'yearly':
        time_col = 'year'
        group_cols = ['segment_id', 'year']
    else:
        raise ValueError(f"Invalid time_window: {time_window}")

    if verbose:
        print(f"  Grouping by: {group_cols}")
        print(f"  Time windows: {df[time_col].nunique()}")

    # Build aggregation dict dynamically based on available columns
    agg_dict = {
        # Crash counts
        'ID': 'count',  # total crashes

        # Severity
        'Severity': 'mean',  # mean severity score
        'high_severity': 'sum',  # count of high-severity crashes

        # Location
        'City': 'first',
        'Start_Lat': 'mean',
        'Start_Lng': 'mean',
    }

    # Add optional road features if they exist
    if 'highway_type' in df.columns:
        agg_dict['highway_type'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    if 'num_lanes' in df.columns:
        agg_dict['num_lanes'] = 'mean'
    if 'speed_limit' in df.columns:
        agg_dict['speed_limit'] = 'mean'
    if 'is_bridge' in df.columns:
        agg_dict['is_bridge'] = 'max'
    if 'is_tunnel' in df.columns:
        agg_dict['is_tunnel'] = 'max'

    # Add traffic features if they exist
    if 'aadt' in df.columns:
        agg_dict['aadt'] = 'mean'
    if 'distance_to_aadt_m' in df.columns:
        agg_dict['distance_to_aadt_m'] = 'mean'

    # Add location features if they exist
    if 'is_urban' in df.columns:
        agg_dict['is_urban'] = 'max'

    # Add temporal features if they exist
    if 'is_rush_hour' in df.columns:
        agg_dict['is_rush_hour'] = 'mean'
    if 'is_weekend' in df.columns:
        agg_dict['is_weekend'] = 'mean'
    if 'hour' in df.columns:
        agg_dict['hour'] = 'mean'

    # Add weather features if they exist
    if 'adverse_weather' in df.columns:
        agg_dict['adverse_weather'] = 'mean'
    if 'low_visibility' in df.columns:
        agg_dict['low_visibility'] = 'mean'
    if 'Temperature(F)' in df.columns:
        agg_dict['Temperature(F)'] = 'mean'
    if 'Humidity(%)' in df.columns:
        agg_dict['Humidity(%)'] = 'mean'
    if 'Visibility(mi)' in df.columns:
        agg_dict['Visibility(mi)'] = 'mean'
    if 'Wind_Speed(mph)' in df.columns:
        agg_dict['Wind_Speed(mph)'] = 'mean'

    # Add infrastructure features if they exist
    if 'Junction' in df.columns:
        agg_dict['Junction'] = 'max'
    if 'Traffic_Signal' in df.columns:
        agg_dict['Traffic_Signal'] = 'max'
    if 'Stop' in df.columns:
        agg_dict['Stop'] = 'max'
    if 'Crossing' in df.columns:
        agg_dict['Crossing'] = 'max'

    # Perform aggregation
    if verbose:
        print(f"\n  Aggregating {len(df):,} crashes...")

    agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Rename count column
    agg_df.rename(columns={'ID': 'crash_count'}, inplace=True)

    # Calculate derived metrics
    agg_df['severity_rate'] = agg_df['high_severity'] / agg_df['crash_count']
    agg_df['traffic_impact'] = agg_df['aadt'] * agg_df['severity_rate']
    agg_df['crash_density'] = agg_df['crash_count'] / agg_df['aadt'] * 1000  # crashes per 1000 vehicles

    # Filter by minimum crashes
    if min_crashes > 1:
        before = len(agg_df)
        agg_df = agg_df[agg_df['crash_count'] >= min_crashes]
        if verbose:
            print(f"  Filtered to segments with >= {min_crashes} crashes: {before:,} → {len(agg_df):,}")

    if verbose:
        print(f"\n✓ Created {len(agg_df):,} segment-time records")
        print(f"\n  Summary statistics:")
        print(f"    Mean crash_count: {agg_df['crash_count'].mean():.1f}")
        print(f"    Mean severity_rate: {agg_df['severity_rate'].mean():.2%}")
        print(f"    Mean traffic_impact: {agg_df['traffic_impact'].mean():.0f}")
        print(f"    Mean crash_density: {agg_df['crash_density'].mean():.2f} per 1000 vehicles")

    return agg_df

def engineer_segment_features(df, verbose=True):
    """
    Engineer additional features for segment-level dataset

    Args:
        df: Aggregated segment DataFrame
        verbose: Print progress

    Returns:
        DataFrame with additional features
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 4: ENGINEERING SEGMENT FEATURES")
        print("="*80 + "\n")

    # Historical crash rate for this segment (across all time periods)
    segment_history = df.groupby('segment_id')['crash_count'].agg(['mean', 'std', 'max']).reset_index()
    segment_history.columns = ['segment_id', 'segment_crash_mean', 'segment_crash_std', 'segment_crash_max']

    df = df.merge(segment_history, on='segment_id', how='left')

    # Risk score components
    df['risk_score_simple'] = (
        df['crash_count'] * 0.4 +  # Frequency
        df['severity_rate'] * 100 * 0.3 +  # Severity (scaled)
        df['traffic_impact'] / 1000 * 0.3  # Traffic impact (scaled)
    )

    # Categorical risk bins
    df['risk_category'] = pd.cut(
        df['risk_score_simple'],
        bins=[0, 2, 5, 10, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
    )

    if verbose:
        print("  ✓ Historical segment features")
        print("  ✓ Risk score components")
        print("  ✓ Risk categories")
        print(f"\n  Risk category distribution:")
        for cat, count in df['risk_category'].value_counts().sort_index().items():
            pct = count / len(df) * 100
            print(f"    {cat}: {count:,} ({pct:.1f}%)")

        print(f"\n✓ Feature engineering complete")

    return df

def create_train_val_test_split(df, train_years, val_years, test_years, verbose=True):
    """
    Split segment data by year (temporal validation)

    Args:
        df: Segment DataFrame
        train_years: List of years for training
        val_years: List of years for validation
        test_years: List of years for testing
        verbose: Print progress

    Returns:
        train, val, test DataFrames
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 5: CREATING TRAIN/VAL/TEST SPLITS")
        print("="*80 + "\n")

    train = df[df['year'].isin(train_years)]
    val = df[df['year'].isin(val_years)]
    test = df[df['year'].isin(test_years)]

    if verbose:
        print(f"Train ({min(train_years)}-{max(train_years)}): {len(train):,} samples ({len(train)/len(df)*100:.1f}%)")
        print(f"Val   ({min(val_years)}-{max(val_years)}): {len(val):,} samples ({len(val)/len(df)*100:.1f}%)")
        print(f"Test  ({min(test_years)}-{max(test_years)}): {len(test):,} samples ({len(test)/len(df)*100:.1f}%)")

        print(f"\n  Risk category distribution:")
        for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
            counts = split_df['risk_category'].value_counts()
            print(f"    {split_name}:")
            for cat in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
                if cat in counts.index:
                    print(f"      {cat}: {counts[cat]:,} ({counts[cat]/len(split_df)*100:.1f}%)")

    return train, val, test

def save_datasets(train, val, test, output_dir, verbose=True):
    """
    Save segment-level datasets

    Args:
        train, val, test: DataFrames
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Paths to saved files
    """

    if verbose:
        print("\n" + "="*80)
        print("STEP 6: SAVING DATASETS")
        print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save files
    train_file = output_dir / f"segment_train_{timestamp}.csv"
    val_file = output_dir / f"segment_val_{timestamp}.csv"
    test_file = output_dir / f"segment_test_{timestamp}.csv"

    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)

    if verbose:
        print(f"Saving datasets to {output_dir}...")
        print(f"  ✓ Train: {train_file.name} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  ✓ Val:   {val_file.name} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  ✓ Test:  {test_file.name} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Create symlinks
    for name, path in [('train', train_file), ('val', val_file), ('test', test_file)]:
        link = output_dir / f"segment_{name}_latest.csv"
        if link.exists():
            link.unlink()
        link.symlink_to(path.name)

    if verbose:
        print(f"\n  ✓ Created symlinks: segment_train_latest.csv, segment_val_latest.csv, segment_test_latest.csv")

    return train_file, val_file, test_file

def print_summary(train, val, test):
    """Print final dataset summary"""

    print("\n" + "="*80)
    print("SEGMENT-LEVEL DATASET SUMMARY")
    print("="*80 + "\n")

    print(f"Total samples: {len(train) + len(val) + len(test):,}")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")

    print(f"\nTotal features: {len(train.columns)}")

    print(f"\nKey metrics (train):")
    print(f"  crash_count       : {train['crash_count'].mean():.2f} ± {train['crash_count'].std():.2f}")
    print(f"  severity_rate     : {train['severity_rate'].mean():.2%} ± {train['severity_rate'].std():.2%}")
    print(f"  traffic_impact    : {train['traffic_impact'].mean():.0f} ± {train['traffic_impact'].std():.0f}")
    print(f"  crash_density     : {train['crash_density'].mean():.3f} ± {train['crash_density'].std():.3f}")
    print(f"  risk_score_simple : {train['risk_score_simple'].mean():.2f} ± {train['risk_score_simple'].std():.2f}")

    print(f"\nFeature completeness (train):")
    for col in ['highway_type', 'num_lanes', 'speed_limit', 'aadt', 'Temperature(F)']:
        if col in train.columns:
            completeness = (1 - train[col].isna().sum() / len(train)) * 100
            print(f"  {col:20s}: {completeness:5.1f}%")

    print("\n" + "="*80)
    print("✅ SEGMENT-LEVEL DATASET CREATION COMPLETE!")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Build segment-level dataset for work zone risk prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use latest crash-level training data, quarterly aggregation
  python build_segment_dataset.py

  # Monthly aggregation
  python build_segment_dataset.py --time-window monthly

  # Require minimum 5 crashes per segment-time
  python build_segment_dataset.py --min-crashes 5

  # Test with sample
  python build_segment_dataset.py --sample 10000
        """
    )

    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                       help=f'Input crash-level CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--time-window', type=str, default='quarterly',
                       choices=['monthly', 'quarterly', 'yearly'],
                       help='Time aggregation window (default: quarterly)')
    parser.add_argument('--min-crashes', type=int, default=1,
                       help='Minimum crashes per segment-time (default: 1)')
    parser.add_argument('--sample', type=int,
                       help='Sample size for testing (optional)')
    parser.add_argument('--train-years', type=str, default='2016,2017,2018,2019,2020,2021',
                       help='Comma-separated train years (default: 2016-2021)')
    parser.add_argument('--val-years', type=str, default='2022',
                       help='Comma-separated validation years (default: 2022)')
    parser.add_argument('--test-years', type=str, default='2023',
                       help='Comma-separated test years (default: 2023)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Parse year arguments
    train_years = [int(y) for y in args.train_years.split(',')]
    val_years = [int(y) for y in args.val_years.split(',')]
    test_years = [int(y) for y in args.test_years.split(',')]

    verbose = not args.quiet

    if verbose:
        print("\n" + "="*80)
        print("SEGMENT-LEVEL DATASET BUILDER")
        print("="*80 + "\n")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input file: {args.input}")
        print(f"Output dir: {args.output_dir}")
        print(f"Time window: {args.time_window}")
        print(f"Min crashes: {args.min_crashes}")
        if args.sample:
            print(f"Sample size: {args.sample:,}")

    # Build dataset
    df = load_crash_data(args.input, sample_size=args.sample, verbose=verbose)
    df = create_segment_identifiers(df, verbose=verbose)
    df = aggregate_by_segment_and_time(df, time_window=args.time_window,
                                      min_crashes=args.min_crashes, verbose=verbose)
    df = engineer_segment_features(df, verbose=verbose)

    train, val, test = create_train_val_test_split(
        df, train_years, val_years, test_years, verbose=verbose
    )

    train_file, val_file, test_file = save_datasets(
        train, val, test, args.output_dir, verbose=verbose
    )

    if verbose:
        print_summary(train, val, test)
        print(f"\n✅ Datasets saved:")
        print(f"   {train_file}")
        print(f"   {val_file}")
        print(f"   {test_file}")
        print()

if __name__ == "__main__":
    main()
