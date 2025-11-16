#!/usr/bin/env python3
"""
Figure 1: Geospatial Distribution of Crash Data (Simplified)

Creates geospatial visualizations using county-level aggregations
since the ML dataset has discretized coordinates.

Author: Capstone Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CRASH_LEVEL_ML = PROJECT_ROOT / 'data' / 'gold' / 'ml_datasets' / 'crash_level'

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 1: GEOSPATIAL DISTRIBUTION OF CRASH DATA')
print('='*80)

# Load data
print('\nLoading crash dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
train_path = CRASH_LEVEL_ML / 'train_latest.csv'

# Load with relevant columns
train = pd.read_csv(train_path, usecols=[
    'ID', 'high_severity', 'City', 'County', 'region', 'is_major_city'
])
test = pd.read_csv(test_path, usecols=[
    'ID', 'high_severity', 'City', 'County', 'region', 'is_major_city'
])

# Combine
all_crashes = pd.concat([train, test], ignore_index=True)
print(f'  ✓ Loaded {len(all_crashes):,} total crashes')
print(f'  ✓ High severity: {all_crashes["high_severity"].sum():,} ({all_crashes["high_severity"].mean()*100:.1f}%)')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel A: Top counties by crash count
ax1 = axes[0, 0]
county_stats = all_crashes.groupby('County').agg({
    'ID': 'count',
    'high_severity': 'mean'
}).rename(columns={'ID': 'crash_count'}).sort_values('crash_count', ascending=False).head(15)

county_stats['crash_count'].plot(kind='barh', ax=ax1, color='steelblue', alpha=0.7)
ax1.set_title('Panel A: Top 15 Counties by Crash Count', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Number of Crashes', fontsize=11)
ax1.set_ylabel('County', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# Add counts
for i, (idx, row) in enumerate(county_stats.iterrows()):
    ax1.text(row['crash_count'] + 500, i, f"{int(row['crash_count']):,}",
            va='center', fontsize=9)

# Panel B: Severity rate by top counties
ax2 = axes[0, 1]
top_counties = county_stats.index[:15]
severity_by_county = all_crashes[all_crashes['County'].isin(top_counties)].groupby('County')['high_severity'].mean() * 100
severity_by_county = severity_by_county.loc[top_counties]

severity_by_county.plot(kind='barh', ax=ax2, color='coral', alpha=0.7)
ax2.set_title('Panel B: High Severity Rate - Top 15 Counties', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('High Severity Rate (%)', fontsize=11)
ax2.set_ylabel('County', fontsize=11)
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=all_crashes['high_severity'].mean()*100, color='red', linestyle='--', alpha=0.5, label='Overall avg')
ax2.legend()

# Panel C: Top cities
ax3 = axes[1, 0]
city_stats = all_crashes.groupby('City').agg({
    'ID': 'count',
    'high_severity': 'mean'
}).rename(columns={'ID': 'crash_count'}).sort_values('crash_count', ascending=False).head(12)

city_stats['crash_count'].plot(kind='barh', ax=ax3, color='lightgreen', alpha=0.7)
ax3.set_title('Panel C: Top 12 Cities by Crash Count', fontsize=13, fontweight='bold', pad=15)
ax3.set_xlabel('Number of Crashes', fontsize=11)
ax3.set_ylabel('City', fontsize=11)
ax3.grid(axis='x', alpha=0.3)

# Panel D: Region distribution
ax4 = axes[1, 1]
region_stats = all_crashes.groupby('region').agg({
    'ID': 'count',
    'high_severity': ['mean', 'sum']
})
region_stats.columns = ['crash_count', 'severity_rate', 'severe_count']
region_stats = region_stats.sort_values('crash_count', ascending=False)

x = np.arange(len(region_stats))
width = 0.35

bars1 = ax4.bar(x - width/2, region_stats['crash_count']/1000, width,
               label='Total Crashes (000s)', color='steelblue', alpha=0.7)
bars2 = ax4.bar(x + width/2, region_stats['severe_count']/1000, width,
               label='Severe Crashes (000s)', color='red', alpha=0.7)

ax4.set_title('Panel D: Crash Distribution by Texas Region', fontsize=13, fontweight='bold', pad=15)
ax4.set_xlabel('Region', fontsize=11)
ax4.set_ylabel('Number of Crashes (thousands)', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(region_stats.index, rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Geospatial Distribution of Texas Crash Data (2016-2020)',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_1_geospatial.png', dpi=300, bbox_inches='tight')
print(f'\n  ✓ Saved: {OUTPUT_DIR / "figure_1_geospatial.png"}')
plt.close()

# Summary stats
print('\n' + '='*80)
print('SUMMARY STATISTICS')
print('='*80)
print(f'\nTotal crashes: {len(all_crashes):,}')
print(f'High severity: {all_crashes["high_severity"].sum():,} ({all_crashes["high_severity"].mean()*100:.1f}%)')
print(f'Low severity:  {(~all_crashes["high_severity"].astype(bool)).sum():,} ({(1-all_crashes["high_severity"].mean())*100:.1f}%)')
print(f'\nUnique counties: {all_crashes["County"].nunique()}')
print(f'Unique cities: {all_crashes["City"].nunique()}')
print(f'Unique regions: {all_crashes["region"].nunique()}')

print('\n✅ Figure 1 complete!')
