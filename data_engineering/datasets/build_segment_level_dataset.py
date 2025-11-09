#!/usr/bin/env python3
"""
Build Segment-Level ML Dataset

Aggregates crash data onto HPMS road segments for road-level risk prediction.
Creates multiple target variables and aggregates crash statistics by road segment.

Input:
  - HPMS Texas road segments (971K segments)
  - Crash data from 2016-2020

Output: data/gold/ml_datasets/segment_level/
  - train_latest.csv
  - val_latest.csv
  - test_latest.csv

Usage:
  python build_segment_level_dataset.py

Author: Data Engineering Team
Date: 2025-11-04
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import paths from config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    TEXAS_SILVER_ROADWAY, TEXAS_BRONZE_CRASHES,
    SEGMENT_LEVEL_ML, DEFAULT_HPMS_FILE, DEFAULT_CRASH_FILE
)

print('='*80)
print('SEGMENT-LEVEL ML DATASET BUILDER')
print('='*80)
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ============================================================================
# 1. LOAD HPMS ROAD SEGMENTS
# ============================================================================
print('\n' + '='*80)
print('1. LOADING HPMS ROAD SEGMENTS')
print('='*80)

hpms_file = DEFAULT_HPMS_FILE
if not hpms_file.exists():
    raise FileNotFoundError(f"HPMS file not found: {hpms_file}")

print(f'\nLoading {hpms_file}...')
hpms = gpd.read_file(hpms_file)
print(f'  ✓ Loaded {len(hpms):,} road segments')
print(f'  ✓ CRS: {hpms.crs}')

# Check key columns (note: HPMS has 'urban_id' not 'urban_code')
required_cols = ['speed_limit', 'through_lanes', 'f_system', 'urban_id', 'aadt']
available_cols = [c for c in required_cols if c in hpms.columns]
print(f'  ✓ Available HPMS features: {len(available_cols)}/{len(required_cols)}')

# Convert HPMS features to numeric (they're stored as strings with "NULL" for missing)
print('\nConverting HPMS features to numeric types...')
for col in available_cols:
    if col in hpms.columns:
        # Replace "NULL" string with NaN, then convert to numeric
        hpms[col] = pd.to_numeric(hpms[col].replace('NULL', np.nan), errors='coerce')
        completeness = hpms[col].notna().mean() * 100
        print(f'    - {col}: {completeness:.1f}% complete')

# Create unique segment ID
hpms['segment_id'] = hpms.index.astype(str)

# Convert to UTM for spatial operations
print('\nConverting to UTM Zone 14N...')
hpms_utm = hpms.to_crs('EPSG:32614')
print(f'  ✓ Converted to {hpms_utm.crs}')

# ============================================================================
# 2. LOAD CRASH DATA
# ============================================================================
print('\n' + '='*80)
print('2. LOADING CRASH DATA')
print('='*80)

crashes_file = DEFAULT_CRASH_FILE
print(f'\nLoading {crashes_file}...')
crashes = pd.read_csv(crashes_file)
print(f'  ✓ Loaded {len(crashes):,} crashes')

# Filter to 2016-2020 (consistent severity definition)
crashes['Start_Time'] = pd.to_datetime(crashes['Start_Time'])
crashes['year'] = crashes['Start_Time'].dt.year
crashes_filtered = crashes[(crashes['year'] >= 2016) & (crashes['year'] <= 2020)].copy()
print(f'  ✓ Filtered to 2016-2020: {len(crashes_filtered):,} crashes')

# Create high_severity target
crashes_filtered['high_severity'] = (crashes_filtered['Severity'] >= 3).astype(int)
print(f'  ✓ High severity: {crashes_filtered["high_severity"].sum():,} / {len(crashes_filtered):,} ({crashes_filtered["high_severity"].mean()*100:.1f}%)')

# Convert to GeoDataFrame
print('\nConverting crashes to GeoDataFrame...')
crashes_gdf = gpd.GeoDataFrame(
    crashes_filtered,
    geometry=gpd.points_from_xy(crashes_filtered['Start_Lng'], crashes_filtered['Start_Lat']),
    crs='EPSG:4326'
)
print(f'  ✓ Created GeoDataFrame with {len(crashes_gdf):,} crashes')

# Convert to UTM
print('Converting to UTM Zone 14N...')
crashes_utm = crashes_gdf.to_crs('EPSG:32614')
print(f'  ✓ Converted to {crashes_utm.crs}')

# ============================================================================
# 3. SPATIAL JOIN: CRASHES TO ROAD SEGMENTS
# ============================================================================
print('\n' + '='*80)
print('3. SPATIAL JOIN: CRASHES TO ROAD SEGMENTS')
print('='*80)

print('\nJoining crashes to nearest road segments (max distance: 100m)...')
print('This may take several minutes...')

# Perform spatial join
crashes_with_segments = crashes_utm.sjoin_nearest(
    hpms_utm[['segment_id', 'geometry']],
    how='left',
    max_distance=100,
    distance_col='segment_distance_m'
)

# Check join quality
joined_count = crashes_with_segments['segment_id'].notna().sum()
join_rate = joined_count / len(crashes_with_segments) * 100
print(f'  ✓ Joined {joined_count:,} / {len(crashes_with_segments):,} crashes ({join_rate:.1f}%)')

# Distance statistics
matched = crashes_with_segments[crashes_with_segments['segment_id'].notna()]
print(f'  ✓ Average distance to segment: {matched["segment_distance_m"].mean():.1f}m')
print(f'  ✓ Median distance: {matched["segment_distance_m"].median():.1f}m')
print(f'  ✓ Max distance: {matched["segment_distance_m"].max():.1f}m')

# Filter to matched crashes only
crashes_matched = crashes_with_segments[crashes_with_segments['segment_id'].notna()].copy()
print(f'\n  ✓ Using {len(crashes_matched):,} crashes matched to segments')

# ============================================================================
# 4. AGGREGATE CRASHES BY SEGMENT
# ============================================================================
print('\n' + '='*80)
print('4. AGGREGATING CRASHES BY SEGMENT')
print('='*80)

print('\nAggregating crash statistics by segment...')

# Prepare aggregation features (TARGETS ONLY - no weather/temporal to avoid leakage)
agg_features = {}

# Count-based targets
agg_features['crash_count'] = ('ID', 'count')
agg_features['high_severity_count'] = ('high_severity', 'sum')

# NOTE: Removed temporal and weather aggregates to avoid data leakage
# These were calculated FROM crashes, creating circular dependency with target

# Aggregate
segment_stats = crashes_matched.groupby('segment_id').agg(**agg_features).reset_index()

print(f'  ✓ Aggregated {len(segment_stats):,} segments with crashes')

# NOTE: Removed derived targets (crash_rate, high_severity_rate, risk_score) to avoid leakage
# These are directly derived from crash_count and create data leakage

print(f'  ✓ Created 2 target variables:')
print(f'    - crash_count: {segment_stats["crash_count"].describe()[["mean", "50%", "max"]]}'.replace('\n', '\n      '))
print(f'    - high_severity_count: {segment_stats["high_severity_count"].describe()[["mean", "50%", "max"]]}'.replace('\n', '\n      '))

# ============================================================================
# 5. JOIN SEGMENT FEATURES
# ============================================================================
print('\n' + '='*80)
print('5. JOINING SEGMENT FEATURES')
print('='*80)

# Select HPMS features to include
hpms_features = hpms[['segment_id'] + available_cols].copy()

# Merge with crash aggregations
print('\nMerging segment stats with HPMS features...')
segment_data = hpms_features.merge(segment_stats, on='segment_id', how='left')

# Fill missing crash counts with 0 (segments with no crashes)
crash_cols = ['crash_count', 'high_severity_count']
for col in crash_cols:
    if col in segment_data.columns:
        segment_data[col] = segment_data[col].fillna(0)

print(f'  ✓ Total segments in dataset: {len(segment_data):,}')
print(f'  ✓ Segments with crashes: {(segment_data["crash_count"] > 0).sum():,} ({(segment_data["crash_count"] > 0).mean()*100:.1f}%)')
print(f'  ✓ Segments without crashes: {(segment_data["crash_count"] == 0).sum():,} ({(segment_data["crash_count"] == 0).mean()*100:.1f}%)')

# ============================================================================
# EXPOSURE NORMALIZATION (NOT LEAKAGE - uses independent road characteristics)
# ============================================================================
print('\n  Computing exposure-normalized crash rate...')

# Get geometry from original HPMS to calculate segment length
segment_data = segment_data.merge(hpms[['segment_id', 'geometry']], on='segment_id', how='left')

# Calculate segment length in miles (geometry is in UTM meters)
if 'geometry' in segment_data.columns:
    segment_data_gdf = gpd.GeoDataFrame(segment_data, geometry='geometry', crs='EPSG:3083')
    segment_data['length_miles'] = segment_data_gdf.geometry.length / 1609.34  # meters to miles

    # Calculate Vehicle Miles Traveled (VMT) per year
    # VMT = AADT × length_miles × 365 days
    if 'aadt' in segment_data.columns:
        segment_data['vmt_annual'] = (
            segment_data['aadt'] * segment_data['length_miles'] * 365
        )

        # Crash rate per 100 million VMT (standard safety metric)
        # This is NOT leakage - it normalizes crashes by exposure
        segment_data['crash_rate_per_100M_vmt'] = (
            segment_data['crash_count'] / (segment_data['vmt_annual'] / 1e8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        segment_data['high_severity_rate_per_100M_vmt'] = (
            segment_data['high_severity_count'] / (segment_data['vmt_annual'] / 1e8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        valid_rate = (segment_data['crash_rate_per_100M_vmt'] > 0).sum()
        print(f'  ✓ Created exposure-normalized crash_rate ({valid_rate:,} non-zero values)')
        print(f'    Mean crash rate: {segment_data["crash_rate_per_100M_vmt"].mean():.2f} crashes per 100M VMT')

        # Drop VMT column (intermediate calculation)
        segment_data = segment_data.drop(columns=['vmt_annual', 'geometry'], errors='ignore')
    else:
        print(f'  ⚠️  AADT not available - skipping exposure normalization')
        segment_data = segment_data.drop(columns=['geometry'], errors='ignore')
else:
    print(f'  ⚠️  Geometry not available - skipping exposure normalization')

# ============================================================================
# 6. DATA QUALITY AND FILTERING
# ============================================================================
print('\n' + '='*80)
print('6. DATA QUALITY AND FILTERING')
print('='*80)

# Keep only segments with sufficient data quality
print('\nFiltering segments for data quality...')
initial_count = len(segment_data)

# Option 1: Keep all segments (including those with 0 crashes)
# Option 2: Keep only segments with crashes
# For now, let's keep all segments but flag those with crashes

segment_data['has_crashes'] = (segment_data['crash_count'] > 0).astype(int)

# Remove duplicates if any
dup_count = segment_data['segment_id'].duplicated().sum()
if dup_count > 0:
    print(f'  ⚠️  Removing {dup_count:,} duplicate segments...')
    segment_data = segment_data.drop_duplicates(subset='segment_id', keep='first')
    print(f'  ✓ Kept {len(segment_data):,} unique segments')

print(f'  ✓ Final dataset: {len(segment_data):,} segments')

# ============================================================================
# 7. FEATURE ENGINEERING
# ============================================================================
print('\n' + '='*80)
print('7. FEATURE ENGINEERING')
print('='*80)

print('\nCreating interaction features...')

# Interaction features (address Gemini's recommendation)
# Handle NaN values properly - interaction is NaN if either component is NaN
if 'speed_limit' in segment_data.columns and 'aadt' in segment_data.columns:
    segment_data['speed_x_aadt'] = segment_data['speed_limit'] * segment_data['aadt']
    valid_count = segment_data['speed_x_aadt'].notna().sum()
    print(f'  ✓ Created speed_limit × aadt ({valid_count:,} / {len(segment_data):,} non-null)')

if 'f_system' in segment_data.columns and 'urban_id' in segment_data.columns:
    segment_data['fsystem_x_urban'] = segment_data['f_system'] * segment_data['urban_id']
    valid_count = segment_data['fsystem_x_urban'].notna().sum()
    print(f'  ✓ Created f_system × urban_id ({valid_count:,} / {len(segment_data):,} non-null)')

if 'through_lanes' in segment_data.columns and 'aadt' in segment_data.columns:
    segment_data['lanes_x_aadt'] = segment_data['through_lanes'] * segment_data['aadt']
    valid_count = segment_data['lanes_x_aadt'].notna().sum()
    print(f'  ✓ Created through_lanes × aadt ({valid_count:,} / {len(segment_data):,} non-null)')

print(f'\n  ✓ Total features: {len(segment_data.columns)}')

# ============================================================================
# 8. CREATE TRAIN/VAL/TEST SPLITS
# ============================================================================
print('\n' + '='*80)
print('8. CREATING TRAIN/VAL/TEST SPLITS')
print('='*80)

# Use stratified random split based on crash presence and crash count bins
from sklearn.model_selection import train_test_split

# Create stratification variable
# Stratify by: has_crashes + crash_count_bin
# Since 95.6% of segments have crash_count=0, use custom binning
segment_data['crash_bin'] = 'zero'
mask_nonzero = segment_data['crash_count'] > 0
if mask_nonzero.sum() > 0:
    # Only bin non-zero crash counts
    try:
        segment_data.loc[mask_nonzero, 'crash_bin'] = pd.qcut(
            segment_data.loc[mask_nonzero, 'crash_count'],
            q=min(5, mask_nonzero.sum()),  # Use min to avoid errors with small samples
            labels=False,
            duplicates='drop'
        ).astype(str)
    except:
        # If qcut fails, use simple binning
        segment_data.loc[mask_nonzero, 'crash_bin'] = 'nonzero'

segment_data['strata'] = (
    segment_data['has_crashes'].astype(str) + '_' + segment_data['crash_bin'].astype(str)
)

print(f'\nStratification groups:')
print(segment_data['strata'].value_counts().sort_index())

# Split: 70% train, 15% val, 15% test
print('\nSplitting into train/val/test (70/15/15)...')

train_val, test = train_test_split(
    segment_data,
    test_size=0.15,
    random_state=42,
    stratify=segment_data['strata']
)

train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,  # 15% of original = 17.65% of train_val
    random_state=42,
    stratify=train_val['strata']
)

print(f'  ✓ Train: {len(train):,} segments ({len(train)/len(segment_data)*100:.1f}%)')
print(f'  ✓ Val:   {len(val):,} segments ({len(val)/len(segment_data)*100:.1f}%)')
print(f'  ✓ Test:  {len(test):,} segments ({len(test)/len(segment_data)*100:.1f}%)')

# Check target distributions
print(f'\nTarget distributions:')
print(f'  Average crash_count:')
print(f'    Train: {train["crash_count"].mean():.2f}')
print(f'    Val:   {val["crash_count"].mean():.2f}')
print(f'    Test:  {test["crash_count"].mean():.2f}')

print(f'  Average high_severity_count:')
print(f'    Train: {train["high_severity_count"].mean():.2f}')
print(f'    Val:   {val["high_severity_count"].mean():.2f}')
print(f'    Test:  {test["high_severity_count"].mean():.2f}')

print(f'  Segments with crashes:')
print(f'    Train: {train["has_crashes"].mean()*100:.1f}%')
print(f'    Val:   {val["has_crashes"].mean()*100:.1f}%')
print(f'    Test:  {test["has_crashes"].mean()*100:.1f}%')

# Drop temporary columns
cols_to_drop = ['strata', 'crash_bin', 'has_crashes']
train = train.drop(columns=cols_to_drop)
val = val.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

# ============================================================================
# 9. SAVE DATASETS
# ============================================================================
print('\n' + '='*80)
print('9. SAVING DATASETS')
print('='*80)

output_dir = SEGMENT_LEVEL_ML
output_dir.mkdir(parents=True, exist_ok=True)

# Save datasets
print(f'\nSaving to {output_dir}/')

train_file = output_dir / 'train_latest.csv'
val_file = output_dir / 'val_latest.csv'
test_file = output_dir / 'test_latest.csv'

train.to_csv(train_file, index=False)
val.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)

print(f'  ✓ Train: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)')
print(f'  ✓ Val:   {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)')
print(f'  ✓ Test:  {test_file} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)')

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print('\n' + '='*80)
print('10. SUMMARY STATISTICS')
print('='*80)

print(f'\nDataset shape:')
print(f'  Features: {len(train.columns)}')
print(f'  Train samples: {len(train):,}')
print(f'  Val samples: {len(val):,}')
print(f'  Test samples: {len(test):,}')

print(f'\nTarget variables:')
for target in ['crash_count', 'high_severity_count']:
    if target in train.columns:
        print(f'  {target}:')
        print(f'    Mean: {train[target].mean():.2f}')
        print(f'    Median: {train[target].median():.2f}')
        print(f'    Max: {train[target].max():.0f}')

print(f'\nFeature completeness:')
for col in available_cols:
    if col in train.columns:
        completeness = train[col].notna().mean() * 100
        print(f'  {col}: {completeness:.1f}%')

print('\n' + '='*80)
print('SEGMENT-LEVEL DATASET BUILD COMPLETE!')
print('='*80)
print(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
