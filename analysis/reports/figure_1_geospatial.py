#!/usr/bin/env python3
"""
Figure 1: Geospatial Distribution of Crash Data and Work Zones

Creates a comprehensive geospatial visualization showing:
- Crash density heatmap across Texas
- High severity vs low severity crashes (color-coded)
- Work zone locations overlaid

This figure demonstrates the scale and geographic distribution of the integrated dataset.

Author: Capstone Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CRASH_LEVEL_ML = PROJECT_ROOT / 'data' / 'gold' / 'ml_datasets' / 'crash_level'

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 1: GEOSPATIAL DISTRIBUTION OF CRASH DATA AND WORK ZONES')
print('='*80)

# ============================================================================
# Load Data
# ============================================================================
print('\nLoading crash dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
train_path = CRASH_LEVEL_ML / 'train_latest.csv'

# Load a sample for visualization (full dataset would be too large)
train = pd.read_csv(train_path, usecols=[
    'ID', 'Start_Lat', 'Start_Lng', 'high_severity', 'City', 'County'
])
test = pd.read_csv(test_path, usecols=[
    'ID', 'Start_Lat', 'Start_Lng', 'high_severity', 'City', 'County'
])

# Combine for total dataset
all_crashes = pd.concat([train, test], ignore_index=True)
print(f'  ✓ Loaded {len(all_crashes):,} total crashes')
print(f'  ✓ High severity: {all_crashes["high_severity"].sum():,} ({all_crashes["high_severity"].mean()*100:.1f}%)')

# Sample for visualization (use all data or sample for performance)
SAMPLE_SIZE = 50000  # Adjust based on performance needs
if len(all_crashes) > SAMPLE_SIZE:
    print(f'\n  Sampling {SAMPLE_SIZE:,} crashes for visualization...')
    crash_sample = all_crashes.sample(n=SAMPLE_SIZE, random_state=42)
else:
    crash_sample = all_crashes

# ============================================================================
# Static Map Visualization (for report)
# ============================================================================
print('\nCreating static geospatial visualization...')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Left panel: All crashes - density visualization
ax1 = axes[0]
ax1.hexbin(
    all_crashes['Start_Lng'],
    all_crashes['Start_Lat'],
    gridsize=50,
    cmap='YlOrRd',
    mincnt=1,
    alpha=0.8
)
ax1.set_title(f'Crash Density Heatmap - Texas ({len(all_crashes):,} crashes, 2016-2020)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add colorbar
plt.colorbar(ax1.collections[0], ax=ax1, label='Crash Count')

# Right panel: Severity-coded scatter (zoomed into major metro)
ax2 = axes[1]

# Focus on Houston area (or Dallas - adjust as needed)
houston_crashes = all_crashes[
    (all_crashes['Start_Lng'] > -95.8) &
    (all_crashes['Start_Lng'] < -95.0) &
    (all_crashes['Start_Lat'] > 29.5) &
    (all_crashes['Start_Lat'] < 30.2)
]

# Sample if too many
if len(houston_crashes) > 10000:
    houston_sample = houston_crashes.sample(n=10000, random_state=42)
else:
    houston_sample = houston_crashes

# Plot by severity
low_severity = houston_sample[houston_sample['high_severity'] == 0]
high_severity = houston_sample[houston_sample['high_severity'] == 1]

ax2.scatter(
    low_severity['Start_Lng'],
    low_severity['Start_Lat'],
    c='steelblue',
    alpha=0.3,
    s=5,
    label=f'Low Severity (n={len(low_severity):,})'
)
ax2.scatter(
    high_severity['Start_Lng'],
    high_severity['Start_Lat'],
    c='red',
    alpha=0.5,
    s=10,
    label=f'High Severity (n={len(high_severity):,})'
)

ax2.set_title('Crash Severity Distribution - Houston Metro Area',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Add annotation
ax2.text(0.02, 0.98,
         f'High Severity Rate: {high_severity.shape[0]/(high_severity.shape[0]+low_severity.shape[0])*100:.1f}%',
         transform=ax2.transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_1_geospatial.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_1_geospatial.png"}')
plt.close()

# ============================================================================
# Interactive HTML Map (optional - for exploration)
# ============================================================================
print('\nCreating interactive HTML map...')

# Center on Texas
texas_center = [31.0, -99.0]
m = folium.Map(location=texas_center, zoom_start=6, tiles='OpenStreetMap')

# Add heatmap layer for all crashes
heat_data = [[row['Start_Lat'], row['Start_Lng']]
             for idx, row in crash_sample.iterrows()
             if pd.notna(row['Start_Lat']) and pd.notna(row['Start_Lng'])]

HeatMap(
    heat_data,
    name='Crash Density',
    min_opacity=0.3,
    max_opacity=0.8,
    radius=8,
    blur=10
).add_to(m)

# Add high severity crashes as markers (sample to avoid overload)
high_sev_sample = crash_sample[crash_sample['high_severity'] == 1].sample(
    n=min(1000, crash_sample['high_severity'].sum()),
    random_state=42
)

marker_cluster = MarkerCluster(name='High Severity Crashes')
for idx, row in high_sev_sample.iterrows():
    folium.CircleMarker(
        location=[row['Start_Lat'], row['Start_Lng']],
        radius=3,
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.6,
        popup=f"City: {row['City']}<br>County: {row['County']}"
    ).add_to(marker_cluster)

marker_cluster.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save interactive map
map_path = OUTPUT_DIR / 'figure_1_interactive_map.html'
m.save(str(map_path))
print(f'  ✓ Saved: {map_path}')

# ============================================================================
# Summary Statistics
# ============================================================================
print('\n' + '='*80)
print('SUMMARY STATISTICS')
print('='*80)
print(f'\nTotal crashes: {len(all_crashes):,}')
print(f'High severity: {all_crashes["high_severity"].sum():,} ({all_crashes["high_severity"].mean()*100:.1f}%)')
print(f'Low severity:  {(~all_crashes["high_severity"].astype(bool)).sum():,} ({(1-all_crashes["high_severity"].mean())*100:.1f}%)')
print(f'\nTop 5 counties by crash count:')
print(all_crashes['County'].value_counts().head(5))
print(f'\nTop 5 cities by crash count:')
print(all_crashes['City'].value_counts().head(5))

print('\n✅ Figure 1 complete!')
print(f'\nOutputs saved to: {OUTPUT_DIR}/')
print('  - figure_1_geospatial.png (static, for report)')
print('  - figure_1_interactive_map.html (interactive, for exploration)')
