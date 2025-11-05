#!/usr/bin/env python3
"""
Crash-Level ML Dataset Builder

Builds crash-level training datasets with high-quality, predictive features.
Uses 2016-2020 data with stratified random split for consistent severity definition.

Features:
- Crash-level predictions (one row per crash)
- HPMS road characteristics integration
- AADT traffic data integration
- Weather and temporal features
- Stratified train/val/test splits (70/15/15)

Output: data/gold/ml_datasets/crash_level/
  - train_latest.csv
  - val_latest.csv
  - test_latest.csv

Usage:
  python build_crash_level_dataset.py
  python build_crash_level_dataset.py --sample 10000

Author: Data Engineering Team
Date: 2025-11-04
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
from datetime import datetime
from shapely.geometry import Point
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Import paths from config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    DEFAULT_CRASH_FILE, DEFAULT_HPMS_FILE, DEFAULT_AADT_FILE,
    CRASH_LEVEL_ML
)

# Configuration
DEFAULT_OUTPUT_DIR = str(CRASH_LEVEL_ML)


def check_duplicates(df, step_name):
    """Check for and report duplicates"""
    if 'ID' in df.columns:
        dup_count = df['ID'].duplicated().sum()
        if dup_count > 0:
            print(f'  ⚠️  WARNING: {dup_count:,} duplicate IDs found after {step_name}')
            print(f'     Removing duplicates...')
            df = df.drop_duplicates(subset='ID', keep='first')
            print(f'     ✓ Removed {dup_count:,} duplicates')
    return df


def load_crashes(crash_file, sample_size=None):
    """Load crash data, filter to 2016-2020, create target"""
    print(f'\n{"="*80}')
    print('STEP 1: LOADING CRASH DATA')
    print(f'{"="*80}')

    print(f'\nReading {crash_file}...')
    df = pd.read_csv(crash_file)
    print(f'  Total crashes in file: {len(df):,}')

    # Filter to 2016-2020 only (consistent severity definition)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['year'] = df['Start_Time'].dt.year
    df = df[(df['year'] >= 2016) & (df['year'] <= 2020)].copy()
    print(f'  After filtering to 2016-2020: {len(df):,}')

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f'  After sampling: {len(df):,}')

    print(f'\nDate range: {df["Start_Time"].min()} to {df["Start_Time"].max()}')

    # Create target variable BEFORE removing Severity
    print(f'\n📊 Creating target variable (high_severity)...')
    if 'Severity' in df.columns:
        df['high_severity'] = (df['Severity'] >= 3).astype(int)
        high_sev_count = df['high_severity'].sum()
        high_sev_pct = high_sev_count / len(df) * 100
        print(f'   ✓ Created target: {high_sev_count:,} high severity ({high_sev_pct:.1f}%)')

    # Remove data leakage features
    print(f'\n🚨 Removing data leakage features...')
    LEAKAGE_FEATURES = ['Severity', 'End_Time', 'End_Lat', 'End_Lng']
    features_to_remove = [f for f in LEAKAGE_FEATURES if f in df.columns]
    if features_to_remove:
        print(f'   Removing: {", ".join(features_to_remove)}')
        df = df.drop(columns=features_to_remove)
        print(f'   ✓ Removed {len(features_to_remove)} leakage features')

    print(f'\n✓ Loaded {len(df):,} crashes')

    df = check_duplicates(df, 'load_crashes')

    return df


def integrate_hpms(crashes_df, hpms_file):
    """Integrate HPMS road characteristics"""
    print(f'\n{"="*80}')
    print('STEP 2: INTEGRATING HPMS ROAD CHARACTERISTICS')
    print(f'{"="*80}')

    try:
        print(f'\nLoading HPMS data from {hpms_file}...')
        hpms_gdf = gpd.read_file(hpms_file)
        print(f'  ✓ Loaded {len(hpms_gdf):,} road segments')

        # Convert crashes to GeoDataFrame
        crashes_gdf = gpd.GeoDataFrame(
            crashes_df,
            geometry=[Point(lon, lat) for lon, lat in zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])],
            crs='EPSG:4326'
        )

        # Convert to UTM for accurate distance
        print(f'  Performing spatial join (nearest road segment within 100m)...')
        crashes_utm = crashes_gdf.to_crs('EPSG:3083')
        hpms_utm = hpms_gdf.to_crs('EPSG:3083')

        # Select HPMS columns we want
        hpms_cols = ['geometry', 'speed_limit', 'through_lanes', 'f_system', 'aadt']
        available_cols = [col for col in hpms_cols if col in hpms_utm.columns]

        # Spatial join
        crashes_with_hpms = crashes_utm.sjoin_nearest(
            hpms_utm[available_cols],
            how='left',
            max_distance=100,
            distance_col='hpms_distance_m'
        )

        matched = crashes_with_hpms['speed_limit'].notna().sum()
        print(f'  ✓ Matched {matched:,} / {len(crashes_with_hpms):,} crashes ({matched/len(crashes_with_hpms)*100:.1f}%)')
        print(f'  Mean distance to road: {crashes_with_hpms["hpms_distance_m"].mean():.1f}m')

        # Rename columns to avoid conflicts
        rename_map = {
            'speed_limit': 'hpms_speed_limit',
            'through_lanes': 'hpms_lanes',
            'f_system': 'hpms_functional_class',
            'aadt': 'hpms_aadt'
        }
        crashes_with_hpms = crashes_with_hpms.rename(columns={k: v for k, v in rename_map.items() if k in crashes_with_hpms.columns})

        print(f'\n  HPMS Feature Completeness:')
        for col in ['hpms_speed_limit', 'hpms_lanes', 'hpms_functional_class', 'hpms_aadt']:
            if col in crashes_with_hpms.columns:
                count = crashes_with_hpms[col].notna().sum()
                pct = count / len(crashes_with_hpms) * 100
                print(f'    {col:25s}: {count:6,} ({pct:5.1f}%)')

        # Convert back to DataFrame
        result_df = pd.DataFrame(crashes_with_hpms.drop(columns=['geometry'], errors='ignore'))

        # Drop spatial join artifacts
        result_df = result_df.drop(columns=['index_right'], errors='ignore')

        result_df = check_duplicates(result_df, 'HPMS integration')

        return result_df

    except Exception as e:
        print(f'\n⚠️  HPMS integration failed: {e}')
        import traceback
        traceback.print_exc()
        print(f'   Continuing without HPMS features...')
        return crashes_df


def attach_aadt(crashes_df, aadt_file):
    """Attach AADT traffic data"""
    print(f'\n{"="*80}')
    print('STEP 3: ATTACHING AADT TRAFFIC DATA')
    print(f'{"="*80}')

    try:
        print(f'\nLoading AADT data from {aadt_file}...')
        aadt_gdf = gpd.read_file(aadt_file)
        print(f'  AADT stations: {len(aadt_gdf):,}')

        # Convert to GeoDataFrame
        crashes_gdf = gpd.GeoDataFrame(
            crashes_df,
            geometry=[Point(lon, lat) for lon, lat in zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])],
            crs='EPSG:4326'
        )

        # Convert to UTM
        crashes_utm = crashes_gdf.to_crs('EPSG:3083')
        aadt_utm = aadt_gdf[['geometry', 'AADT_RPT_QTY']].to_crs('EPSG:3083')

        print(f'  Performing spatial join (nearest station)...')
        crashes_with_aadt = crashes_utm.sjoin_nearest(
            aadt_utm,
            how='left',
            distance_col='distance_to_aadt_m'
        )

        crashes_with_aadt = crashes_with_aadt.rename(columns={'AADT_RPT_QTY': 'aadt'})

        matched = crashes_with_aadt['aadt'].notna().sum()
        print(f'  ✓ Matched {matched:,} / {len(crashes_with_aadt):,} crashes ({matched/len(crashes_with_aadt)*100:.1f}%)')
        print(f'  Mean distance to station: {crashes_with_aadt["distance_to_aadt_m"].mean():.0f}m')

        # Convert back to DataFrame
        result_df = pd.DataFrame(crashes_with_aadt.drop(columns=['geometry'], errors='ignore'))

        # Drop spatial join artifacts
        result_df = result_df.drop(columns=['index_right'], errors='ignore')

        result_df = check_duplicates(result_df, 'AADT integration')

        return result_df

    except Exception as e:
        print(f'\n⚠️  AADT integration failed: {e}')
        print(f'   Continuing without AADT features...')
        return crashes_df


def engineer_features(df):
    """Engineer temporal, weather, and categorical features"""
    print(f'\n{"="*80}')
    print('STEP 4: FEATURE ENGINEERING')
    print(f'{"="*80}')

    print('\nCreating features...')

    # Temporal features (NO YEAR - just drift marker!)
    df['hour'] = pd.to_datetime(df['Start_Time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['Start_Time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['Start_Time']).dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = (
        ((df['hour'] >= 6) & (df['hour'] <= 9)) |
        ((df['hour'] >= 16) & (df['hour'] <= 19))
    ).astype(int)

    print('  ✓ Temporal: hour, day_of_week, month, is_weekend, is_rush_hour')

    # Weather features
    if 'Weather_Condition' in df.columns:
        # Simplify weather to categories
        def categorize_weather(condition):
            if pd.isna(condition):
                return 'clear'
            condition = str(condition).lower()
            if any(x in condition for x in ['rain', 'drizzle', 'shower']):
                return 'rain'
            elif any(x in condition for x in ['snow', 'sleet', 'ice', 'hail']):
                return 'snow'
            elif 'fog' in condition or 'mist' in condition:
                return 'fog'
            elif 'cloud' in condition or 'overcast' in condition:
                return 'cloudy'
            else:
                return 'clear'

        df['weather_category'] = df['Weather_Condition'].apply(categorize_weather)
        df['adverse_weather'] = df['weather_category'].isin(['rain', 'snow', 'fog']).astype(int)

    # Low visibility
    if 'Visibility(mi)' in df.columns:
        df['low_visibility'] = (df['Visibility(mi)'] < 2).astype(int)

    # Temperature categories
    if 'Temperature(F)' in df.columns:
        df['temp_category'] = pd.cut(
            df['Temperature(F)'],
            bins=[-np.inf, 32, 50, 70, 90, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm', 'hot']
        )

    print('  ✓ Weather: weather_category, adverse_weather, low_visibility, temp_category')

    # Location
    if 'City' in df.columns:
        # Urban vs rural (major cities = urban)
        major_cities = ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth', 'El Paso']
        df['is_urban'] = df['City'].isin(major_cities).astype(int)
        print('  ✓ Location: is_urban')

    # Infrastructure
    if 'Junction' in df.columns:
        df['is_junction'] = (df['Junction'].notna()).astype(int)
        print('  ✓ Infrastructure: is_junction')

    print('  ✓ Target: high_severity (created earlier to avoid leakage)')

    return df


def create_stratified_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified random train/val/test split

    This maintains class balance across all splits, unlike temporal split
    which had severe drift (16.3% → 8.4% → 0.4%)
    """
    print(f'\n{"="*80}')
    print('STEP 5: CREATING STRATIFIED TRAIN/VAL/TEST SPLITS')
    print(f'{"="*80}')

    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['high_severity'],
        random_state=random_state
    )

    # Second split: separate train and val
    val_adjusted_size = val_size / (1 - test_size)  # Adjust for already-removed test set
    train, val = train_test_split(
        train_val,
        test_size=val_adjusted_size,
        stratify=train_val['high_severity'],
        random_state=random_state
    )

    print(f'\nSplit sizes:')
    print(f'  Train: {len(train):,} samples ({len(train)/len(df)*100:.1f}%)')
    print(f'  Val:   {len(val):,} samples ({len(val)/len(df)*100:.1f}%)')
    print(f'  Test:  {len(test):,} samples ({len(test)/len(df)*100:.1f}%)')

    # Check class balance
    print('\nClass distribution (high_severity):')
    print(f'  Train: {train["high_severity"].sum():,} / {len(train):,} ({train["high_severity"].mean()*100:.1f}%)')
    print(f'  Val:   {val["high_severity"].sum():,} / {len(val):,} ({val["high_severity"].mean()*100:.1f}%)')
    print(f'  Test:  {test["high_severity"].sum():,} / {len(test):,} ({test["high_severity"].mean()*100:.1f}%)')

    return train, val, test


def save_datasets(train, val, test, output_dir):
    """Save train/val/test datasets"""
    print(f'\n{"="*80}')
    print('STEP 6: SAVING DATASETS')
    print(f'{"="*80}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_path = output_dir / f'train_{timestamp}.csv'
    val_path = output_dir / f'val_{timestamp}.csv'
    test_path = output_dir / f'test_{timestamp}.csv'

    print(f'\nSaving datasets to {output_dir}...')

    train.to_csv(train_path, index=False)
    print(f'  ✓ Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)')

    val.to_csv(val_path, index=False)
    print(f'  ✓ Val:   {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)')

    test.to_csv(test_path, index=False)
    print(f'  ✓ Test:  {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)')

    # Create symlinks to latest
    for name, path in [('train_latest.csv', train_path),
                       ('val_latest.csv', val_path),
                       ('test_latest.csv', test_path)]:
        latest_link = output_dir / name
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.name)

    print(f'\n  ✓ Created symlinks: train_latest.csv, val_latest.csv, test_latest.csv')

    return train_path, val_path, test_path


def print_summary(train, val, test):
    """Print dataset summary"""
    print(f'\n{"="*80}')
    print('DATASET SUMMARY')
    print(f'{"="*80}')

    total = len(train) + len(val) + len(test)
    print(f'\nTotal samples: {total:,}')
    print(f'  Train: {len(train):,}')
    print(f'  Val:   {len(val):,}')
    print(f'  Test:  {len(test):,}')

    print(f'\nTotal features: {len(train.columns)}')

    # Feature completeness
    print(f'\nFeature completeness (train):')
    key_features = [
        'hpms_speed_limit', 'hpms_lanes', 'hpms_functional_class', 'hpms_aadt',
        'aadt', 'Temperature(F)', 'Visibility(mi)', 'weather_category',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour'
    ]

    for feat in key_features:
        if feat in train.columns:
            completeness = train[feat].notna().mean() * 100
            print(f'  {feat:25s}: {completeness:5.1f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Build simplified ML training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset (2016-2020, ~371K crashes)
  python build_ml_dataset_simplified.py

  # Small sample for testing
  python build_ml_dataset_simplified.py --sample 10000

  # Custom output location
  python build_ml_dataset_simplified.py --output-dir /path/to/output
        """
    )

    parser.add_argument('--crash-file', type=str, default=DEFAULT_CRASH_FILE,
                       help='Path to crash data CSV')
    parser.add_argument('--hpms-file', type=str, default=DEFAULT_HPMS_FILE,
                       help='Path to HPMS GeoPackage')
    parser.add_argument('--aadt-file', type=str, default=DEFAULT_AADT_FILE,
                       help='Path to AADT GeoPackage')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (default: use all data)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for datasets')

    args = parser.parse_args()

    print(f'\n{"="*80}')
    print('SIMPLIFIED ML TRAINING DATASET BUILDER')
    print(f'{"="*80}')
    print(f'\nTimestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Data period: 2016-2020 (consistent severity definition)')
    print(f'Split method: Stratified random (70/15/15)')
    print(f'Crash file: {args.crash_file}')
    print(f'Output dir: {args.output_dir}')

    # Pipeline
    crashes_df = load_crashes(args.crash_file, sample_size=args.sample)
    crashes_with_hpms = integrate_hpms(crashes_df, args.hpms_file)
    crashes_with_aadt = attach_aadt(crashes_with_hpms, args.aadt_file)
    ml_dataset = engineer_features(crashes_with_aadt)

    # Stratified split (maintains class balance)
    train, val, test = create_stratified_split(ml_dataset)

    # Save
    train_path, val_path, test_path = save_datasets(train, val, test, args.output_dir)

    # Summary
    print_summary(train, val, test)

    print(f'\n✅ Datasets saved:')
    print(f'   {train_path}')
    print(f'   {val_path}')
    print(f'   {test_path}')
    print()


if __name__ == '__main__':
    main()
