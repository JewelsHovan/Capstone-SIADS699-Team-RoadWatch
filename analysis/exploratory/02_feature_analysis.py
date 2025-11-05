#!/usr/bin/env python3
"""
Feature Analysis and Visualization

Creates comprehensive visualizations of features vs target variable:
- Target distribution
- Temporal patterns (hour, day, month)
- Weather conditions
- Road characteristics (speed, lanes, AADT)
- Geographic patterns
- Feature correlations

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

# Styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

# Create output directory
output_dir = Path('analysis/exploratory/figures')
output_dir.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FEATURE ANALYSIS AND VISUALIZATION')
print('='*80)

# Load training data
print('\nLoading training dataset...')
train = pd.read_csv('data/processed/crash_level/train_latest.csv')
print(f'  ✓ Loaded {len(train):,} samples')

# Parse datetime
train['Start_Time'] = pd.to_datetime(train['Start_Time'])

print('\nGenerating visualizations...')

# ============================================================================
# 1. TARGET DISTRIBUTION
# ============================================================================
print('  [1/8] Target distribution...')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
train['high_severity'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('High Severity')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

# Add value labels
for i, v in enumerate(train['high_severity'].value_counts().values):
    pct = v / len(train) * 100
    axes[0].text(i, v + 1000, f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Year trend
if 'year' in train.columns:
    yearly_severity = train.groupby('year')['high_severity'].mean() * 100
    yearly_severity.plot(kind='line', marker='o', ax=axes[1], linewidth=2, markersize=8, color='steelblue')
    axes[1].set_title('High Severity Rate by Year (2016-2020)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('High Severity Rate (%)')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. TEMPORAL PATTERNS
# ============================================================================
print('  [2/8] Temporal patterns...')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Hour of day
if 'hour' in train.columns:
    hourly_data = train.groupby('hour')['high_severity'].agg(['mean', 'count'])

    # Severity rate
    ax1 = axes[0, 0]
    (hourly_data['mean'] * 100).plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('High Severity Rate by Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('High Severity Rate (%)')
    ax1.set_xticklabels(range(24), rotation=0)
    ax1.axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5, label='Overall avg')
    ax1.legend()

    # Crash count
    ax2 = axes[0, 1]
    hourly_data['count'].plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Crash Count by Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Number of Crashes')
    ax2.set_xticklabels(range(24), rotation=0)

# Day of week
if 'day_of_week' in train.columns:
    daily_data = train.groupby('day_of_week')['high_severity'].agg(['mean', 'count'])

    # Severity rate
    ax3 = axes[1, 0]
    (daily_data['mean'] * 100).plot(kind='bar', ax=ax3, color='steelblue')
    ax3.set_title('High Severity Rate by Day of Week', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('High Severity Rate (%)')
    ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    ax3.axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5, label='Overall avg')
    ax3.legend()

# Month
if 'month' in train.columns:
    monthly_data = train.groupby('month')['high_severity'].agg(['mean', 'count'])

    # Severity rate
    ax4 = axes[1, 1]
    (monthly_data['mean'] * 100).plot(kind='bar', ax=ax4, color='steelblue')
    ax4.set_title('High Severity Rate by Month', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('High Severity Rate (%)')
    ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    ax4.axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5, label='Overall avg')
    ax4.legend()

plt.tight_layout()
plt.savefig(output_dir / '02_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. WEATHER CONDITIONS
# ============================================================================
print('  [3/8] Weather conditions...')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Weather category
if 'weather_category' in train.columns:
    weather_data = train.groupby('weather_category')['high_severity'].agg(['mean', 'count'])
    weather_data = weather_data.sort_values('mean', ascending=False)

    # Severity rate
    (weather_data['mean'] * 100).plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('High Severity Rate by Weather', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('High Severity Rate (%)')
    axes[0, 0].axvline(x=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

    # Count
    weather_data['count'].sort_values().plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Crash Count by Weather', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Crashes')

# Temperature
if 'Temperature(F)' in train.columns:
    temp_bins = pd.cut(train['Temperature(F)'], bins=10)
    temp_data = train.groupby(temp_bins)['high_severity'].agg(['mean', 'count'])

    # Severity by temp
    axes[1, 0].plot(range(len(temp_data)), temp_data['mean'] * 100, marker='o', linewidth=2, markersize=6)
    axes[1, 0].set_title('High Severity Rate by Temperature', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Temperature Range')
    axes[1, 0].set_ylabel('High Severity Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

# Visibility
if 'Visibility(mi)' in train.columns:
    vis_bins = pd.cut(train['Visibility(mi)'], bins=[0, 1, 2, 5, 10, 20])
    vis_data = train.groupby(vis_bins)['high_severity'].agg(['mean', 'count'])

    (vis_data['mean'] * 100).plot(kind='bar', ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('High Severity Rate by Visibility', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Visibility (miles)')
    axes[1, 1].set_ylabel('High Severity Rate (%)')
    axes[1, 1].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / '03_weather_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. ROAD CHARACTERISTICS
# ============================================================================
print('  [4/8] Road characteristics...')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Speed limit
if 'hpms_speed_limit' in train.columns:
    speed_bins = pd.cut(train['hpms_speed_limit'], bins=[0, 30, 45, 55, 65, 75, 100])
    speed_data = train.groupby(speed_bins)['high_severity'].agg(['mean', 'count'])

    (speed_data['mean'] * 100).plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('High Severity Rate by Speed Limit', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Speed Limit (mph)')
    axes[0, 0].set_ylabel('High Severity Rate (%)')
    axes[0, 0].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

    # Add counts
    for i, (idx, row) in enumerate(speed_data.iterrows()):
        axes[0, 0].text(i, row['mean']*100 + 0.5, f"n={int(row['count']):,}",
                       ha='center', va='bottom', fontsize=8)

# Number of lanes
if 'hpms_lanes' in train.columns:
    lanes_data = train.groupby('hpms_lanes')['high_severity'].agg(['mean', 'count'])
    lanes_data = lanes_data[lanes_data.index <= 8]  # Exclude outliers

    (lanes_data['mean'] * 100).plot(kind='bar', ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title('High Severity Rate by Number of Lanes', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Lanes')
    axes[0, 1].set_ylabel('High Severity Rate (%)')
    axes[0, 1].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

# AADT (traffic volume)
if 'aadt' in train.columns:
    aadt_bins = pd.qcut(train['aadt'].dropna(), q=10, duplicates='drop')
    aadt_data = train.groupby(aadt_bins)['high_severity'].agg(['mean', 'count'])

    axes[1, 0].plot(range(len(aadt_data)), aadt_data['mean'] * 100, marker='o', linewidth=2, markersize=6)
    axes[1, 0].set_title('High Severity Rate by Traffic Volume (AADT)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('AADT Decile (Low → High)')
    axes[1, 0].set_ylabel('High Severity Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

# Functional class
if 'hpms_functional_class' in train.columns:
    func_data = train.groupby('hpms_functional_class')['high_severity'].agg(['mean', 'count'])
    func_data = func_data.sort_values('mean', ascending=False).head(10)

    (func_data['mean'] * 100).plot(kind='barh', ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('High Severity Rate by Road Functional Class', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('High Severity Rate (%)')
    axes[1, 1].axvline(x=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / '04_road_characteristics.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. GEOGRAPHIC PATTERNS
# ============================================================================
print('  [5/8] Geographic patterns...')

if 'City' in train.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top cities by crash count
    city_data = train.groupby('City')['high_severity'].agg(['mean', 'count'])
    top_cities = city_data.nlargest(15, 'count')

    # Count
    top_cities['count'].sort_values().plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_title('Top 15 Cities by Crash Count', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Crashes')

    # Severity rate
    (top_cities['mean'] * 100).sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('High Severity Rate in Top 15 Cities', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('High Severity Rate (%)')
    axes[1].axvline(x=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5, label='Overall avg')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / '05_geographic_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 6. FEATURE CORRELATIONS
# ============================================================================
print('  [6/8] Feature correlations...')

# Select numeric features
numeric_features = [
    'high_severity', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hpms_speed_limit', 'hpms_lanes', 'aadt', 'Temperature(F)', 'Visibility(mi)',
    'adverse_weather', 'low_visibility'
]
numeric_features = [f for f in numeric_features if f in train.columns]

# Compute correlation matrix
corr = train[numeric_features].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '06_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. FEATURE IMPORTANCE (Simple)
# ============================================================================
print('  [7/8] Feature importance (correlation with target)...')

# Correlations with target
target_corr = train[numeric_features].corr()['high_severity'].drop('high_severity').abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
target_corr.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Correlation with High Severity (Absolute)', fontsize=14, fontweight='bold')
ax.set_xlabel('Absolute Correlation')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. COMBINED EFFECTS
# ============================================================================
print('  [8/8] Combined effects...')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Rush hour + weather
if 'is_rush_hour' in train.columns and 'weather_category' in train.columns:
    combined = train.groupby(['is_rush_hour', 'weather_category'])['high_severity'].mean().unstack()
    combined_pct = combined * 100

    combined_pct.T.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Severity Rate: Rush Hour × Weather', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Weather Category')
    axes[0].set_ylabel('High Severity Rate (%)')
    axes[0].legend(['Non-rush hour', 'Rush hour'])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

# Weekend + hour
if 'is_weekend' in train.columns and 'hour' in train.columns:
    combined2 = train.groupby(['is_weekend', 'hour'])['high_severity'].mean().unstack()
    combined2_pct = combined2.T * 100

    combined2_pct.plot(kind='line', ax=axes[1], marker='o', linewidth=2)
    axes[1].set_title('Severity Rate: Weekend × Hour', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('High Severity Rate (%)')
    axes[1].legend(['Weekday', 'Weekend'])
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=train['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / '08_combined_effects.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n' + '='*80)
print('VISUALIZATION COMPLETE')
print('='*80)
print(f'\n✅ Generated 8 visualizations in {output_dir}/')
print('\nFiles created:')
for i, name in enumerate([
    '01_target_distribution.png',
    '02_temporal_patterns.png',
    '03_weather_patterns.png',
    '04_road_characteristics.png',
    '05_geographic_patterns.png',
    '06_correlations.png',
    '07_feature_importance.png',
    '08_combined_effects.png'
], 1):
    print(f'  [{i}] {name}')
