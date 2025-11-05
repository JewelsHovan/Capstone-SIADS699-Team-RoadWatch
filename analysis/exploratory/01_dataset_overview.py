#!/usr/bin/env python3
"""
Dataset Overview Analysis

Comprehensive analysis of the ML training dataset:
- Data quality and completeness
- Target variable distribution
- Feature statistics
- Missing value patterns
- Temporal patterns
- Geographic distribution

Author: Data Engineering Team
Date: 2025-11-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import paths from config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import CRASH_LEVEL_ML

# Styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print('='*80)
print('DATASET OVERVIEW ANALYSIS')
print('='*80)

# Load datasets
print('\nLoading datasets...')
data_dir = CRASH_LEVEL_ML

train = pd.read_csv(data_dir / 'train_latest.csv')
val = pd.read_csv(data_dir / 'val_latest.csv')
test = pd.read_csv(data_dir / 'test_latest.csv')

print(f'  ✓ Train: {len(train):,} samples')
print(f'  ✓ Val:   {len(val):,} samples')
print(f'  ✓ Test:  {len(test):,} samples')

# Basic stats
print('\n' + '='*80)
print('1. DATASET STATISTICS')
print('='*80)

print(f'\nShape:')
print(f'  Train: {train.shape}')
print(f'  Val:   {val.shape}')
print(f'  Test:  {test.shape}')

print(f'\nMemory usage:')
print(f'  Train: {train.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB')
print(f'  Val:   {val.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB')
print(f'  Test:  {test.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB')

# Target distribution
print('\n' + '='*80)
print('2. TARGET VARIABLE DISTRIBUTION')
print('='*80)

print(f'\nHigh Severity Rate:')
print(f'  Train: {train["high_severity"].mean()*100:.2f}% ({train["high_severity"].sum():,} / {len(train):,})')
print(f'  Val:   {val["high_severity"].mean()*100:.2f}% ({val["high_severity"].sum():,} / {len(val):,})')
print(f'  Test:  {test["high_severity"].mean()*100:.2f}% ({test["high_severity"].sum():,} / {len(test):,})')

# Temporal distribution
print('\n' + '='*80)
print('3. TEMPORAL DISTRIBUTION')
print('='*80)

train['Start_Time'] = pd.to_datetime(train['Start_Time'])
train['date'] = train['Start_Time'].dt.date

print(f'\nDate range:')
print(f'  Start: {train["Start_Time"].min()}')
print(f'  End:   {train["Start_Time"].max()}')
print(f'  Days:  {(train["Start_Time"].max() - train["Start_Time"].min()).days}')

print(f'\nCrashes by year:')
year_counts = train.groupby('year').size().sort_index()
for year, count in year_counts.items():
    pct = count / len(train) * 100
    print(f'  {year}: {count:6,} ({pct:5.1f}%)')

# Geographic distribution
print('\n' + '='*80)
print('4. GEOGRAPHIC DISTRIBUTION')
print('='*80)

if 'City' in train.columns:
    print(f'\nTop 10 cities by crash count:')
    city_counts = train['City'].value_counts().head(10)
    for city, count in city_counts.items():
        pct = count / len(train) * 100
        high_sev_rate = train[train['City'] == city]['high_severity'].mean() * 100
        print(f'  {city:20s}: {count:6,} ({pct:5.1f}%) | High severity: {high_sev_rate:5.1f}%')

# Feature completeness
print('\n' + '='*80)
print('5. FEATURE COMPLETENESS')
print('='*80)

# Categorize features
feature_categories = {
    'HPMS Road': [c for c in train.columns if c.startswith('hpms_')],
    'Traffic': [c for c in train.columns if 'aadt' in c.lower()],
    'Weather': [c for c in train.columns if any(x in c.lower() for x in ['weather', 'temp', 'visibility', 'wind', 'precip'])],
    'Temporal': [c for c in train.columns if any(x in c.lower() for x in ['hour', 'day', 'month', 'weekend', 'rush'])],
    'Location': [c for c in train.columns if any(x in c.lower() for x in ['city', 'county', 'state', 'lat', 'lng', 'urban'])],
}

print(f'\nCompleteness by category:')
for category, features in feature_categories.items():
    if features:
        completeness = train[features].notna().mean().mean() * 100
        print(f'  {category:15s}: {completeness:5.1f}% ({len(features)} features)')

# Missing values
print('\n' + '='*80)
print('6. MISSING VALUE ANALYSIS')
print('='*80)

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print(f'\nTop 10 features with missing values:')
    for col, count in missing.head(10).items():
        pct = count / len(train) * 100
        print(f'  {col:30s}: {count:6,} ({pct:5.1f}%)')
else:
    print(f'\n✅ No missing values!')

# Feature distributions
print('\n' + '='*80)
print('7. KEY FEATURE STATISTICS')
print('='*80)

# Numeric features
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['ID', 'high_severity', 'year']]

print(f'\nNumeric features ({len(numeric_cols)}):')
for col in ['hpms_speed_limit', 'hpms_lanes', 'aadt', 'Temperature(F)', 'Visibility(mi)']:
    if col in train.columns:
        data = train[col].dropna()
        print(f'\n  {col}:')
        print(f'    Mean:   {data.mean():.2f}')
        print(f'    Median: {data.median():.2f}')
        print(f'    Std:    {data.std():.2f}')
        print(f'    Range:  [{data.min():.2f}, {data.max():.2f}]')

# Data quality checks
print('\n' + '='*80)
print('8. DATA QUALITY CHECKS')
print('='*80)

# Check for duplicates
dup_count = train['ID'].duplicated().sum()
print(f'\nDuplicate IDs: {dup_count:,}')

# Check for outliers in key features
print(f'\nOutlier detection (values outside 1st-99th percentile):')
for col in ['hpms_speed_limit', 'Temperature(F)', 'Visibility(mi)']:
    if col in train.columns:
        p1, p99 = train[col].quantile([0.01, 0.99])
        outliers = ((train[col] < p1) | (train[col] > p99)).sum()
        print(f'  {col:25s}: {outliers:6,} ({outliers/len(train)*100:4.1f}%)')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
