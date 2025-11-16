#!/usr/bin/env python3
"""
Figure 1: Geospatial Distribution (IMPROVED VERSION)

Improvements:
- Better spacing between panels
- Improved label positioning
- Better number formatting
- Cleaner x-axis labels

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
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 1: GEOSPATIAL DISTRIBUTION (IMPROVED)')
print('='*80)

# Load data
print('\nLoading crash dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
train_path = CRASH_LEVEL_ML / 'train_latest.csv'

train = pd.read_csv(train_path, usecols=[
    'ID', 'high_severity', 'City', 'County', 'region', 'is_major_city'
])
test = pd.read_csv(test_path, usecols=[
    'ID', 'high_severity', 'City', 'County', 'region', 'is_major_city'
])

all_crashes = pd.concat([train, test], ignore_index=True)
print(f'  ✓ Loaded {len(all_crashes):,} total crashes')

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

# Create visualization with better spacing
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Panel A: Top counties by crash count
ax1 = fig.add_subplot(gs[0, 0])
county_stats = all_crashes.groupby('County').agg({
    'ID': 'count',
    'high_severity': 'mean'
}).rename(columns={'ID': 'crash_count'}).sort_values('crash_count', ascending=False).head(15)

bars = ax1.barh(range(len(county_stats)), county_stats['crash_count'], color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(county_stats)))
ax1.set_yticklabels(county_stats.index, fontsize=11)
ax1.set_xlabel('Number of Crashes', fontsize=12, fontweight='bold')
ax1.set_ylabel('County', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Top 15 Counties by Crash Count', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add counts outside bars
for i, (idx, row) in enumerate(county_stats.iterrows()):
    ax1.text(row['crash_count'] + 1000, i, f"{int(row['crash_count']):,}",
            va='center', ha='left', fontsize=10, fontweight='bold')

# Extend x-axis to fit labels
ax1.set_xlim(0, county_stats['crash_count'].max() * 1.15)

# Panel B: Severity rate by top counties
ax2 = fig.add_subplot(gs[0, 1])
top_counties = county_stats.index[:15]
severity_by_county = all_crashes[all_crashes['County'].isin(top_counties)].groupby('County')['high_severity'].mean() * 100
severity_by_county = severity_by_county.loc[top_counties]

bars = ax2.barh(range(len(severity_by_county)), severity_by_county, color='coral', alpha=0.7)
ax2.set_yticks(range(len(severity_by_county)))
ax2.set_yticklabels(severity_by_county.index, fontsize=11)
ax2.set_xlabel('High Severity Rate (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('County', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: High Severity Rate - Top 15 Counties', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

overall_avg = all_crashes['high_severity'].mean() * 100
ax2.axvline(x=overall_avg, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall avg ({overall_avg:.1f}%)')
ax2.legend(fontsize=10)

# Panel C: Top cities
ax3 = fig.add_subplot(gs[1, 0])
city_stats = all_crashes.groupby('City').agg({
    'ID': 'count',
    'high_severity': 'mean'
}).rename(columns={'ID': 'crash_count'}).sort_values('crash_count', ascending=False).head(12)

bars = ax3.barh(range(len(city_stats)), city_stats['crash_count'], color='lightgreen', alpha=0.7, edgecolor='darkgreen')
ax3.set_yticks(range(len(city_stats)))
ax3.set_yticklabels(city_stats.index, fontsize=11)
ax3.set_xlabel('Number of Crashes', fontsize=12, fontweight='bold')
ax3.set_ylabel('City', fontsize=12, fontweight='bold')
ax3.set_title('Panel C: Top 12 Cities by Crash Count', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# Add counts
for i, (idx, row) in enumerate(city_stats.iterrows()):
    ax3.text(row['crash_count'] + 500, i, f"{int(row['crash_count']):,}",
            va='center', ha='left', fontsize=10, fontweight='bold')

ax3.set_xlim(0, city_stats['crash_count'].max() * 1.12)

# Panel D: Region distribution - IMPROVED
ax4 = fig.add_subplot(gs[1, 1])
region_stats = all_crashes.groupby('region').agg({
    'ID': 'count',
    'high_severity': ['mean', 'sum']
})
region_stats.columns = ['crash_count', 'severity_rate', 'severe_count']
region_stats = region_stats.sort_values('crash_count', ascending=False).head(12)  # Top 12 regions

x = np.arange(len(region_stats))
width = 0.35

bars1 = ax4.bar(x - width/2, region_stats['crash_count']/1000, width,
               label='Total Crashes (000s)', color='steelblue', alpha=0.7)
bars2 = ax4.bar(x + width/2, region_stats['severe_count']/1000, width,
               label='Severe Crashes (000s)', color='red', alpha=0.7)

ax4.set_xlabel('Region', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Crashes (thousands)', fontsize=12, fontweight='bold')
ax4.set_title('Panel D: Crash Distribution by Texas Region (Top 12)', fontsize=13, fontweight='bold', pad=15)
ax4.set_xticks(x)
# Shorten region names if needed and rotate less
region_labels = [r[:20] + '...' if len(r) > 20 else r for r in region_stats.index]
ax4.set_xticklabels(region_labels, rotation=35, ha='right', fontsize=9)
ax4.legend(fontsize=11, loc='upper right')
ax4.grid(axis='y', alpha=0.3)

# Overall title
fig.suptitle('Geospatial Distribution of Texas Crash Data (2016-2020)',
            fontsize=17, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / 'figure_1_geospatial.png', dpi=300, bbox_inches='tight')
print(f'\n  ✓ Saved: {OUTPUT_DIR / "figure_1_geospatial.png"}')

print('\n✅ Figure 1 (improved) complete!')
plt.close()
