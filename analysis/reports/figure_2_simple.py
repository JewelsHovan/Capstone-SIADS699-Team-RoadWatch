#!/usr/bin/env python3
"""
Figure 2: Feature Importance (SIMPLIFIED - RELIABLE VERSION)

Uses correlation-based feature importance which works reliably
without needing to extract from complex pipeline.

Author: Capstone Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CRASH_LEVEL_ML = PROJECT_ROOT / 'data' / 'gold' / 'ml_datasets' / 'crash_level'
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 2: FEATURE IMPORTANCE (SIMPLIFIED VERSION)')
print('='*80)

# Load data
print('\nLoading dataset...')
train_path = CRASH_LEVEL_ML / 'train_latest.csv'
train = pd.read_csv(train_path)
print(f'  ✓ Loaded {len(train):,} training samples')

# Define features (numeric only for simple analysis)
numeric_features = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'Temperature(F)', 'Visibility(mi)', 'Pressure(in)', 'Humidity(%)', 'Wind_Speed(mph)',
    'adverse_weather', 'low_visibility', 'is_major_city', 'is_junction',
    'aadt', 'distance_to_aadt_m', 'hpms_speed_limit', 'hpms_lanes', 'hpms_aadt'
]

# Filter to features that exist
numeric_features = [f for f in numeric_features if f in train.columns]

X = train[numeric_features]
y = train['high_severity']

print(f'  ✓ Using {len(numeric_features)} numeric features')

# Train a simple Random Forest for feature importance
print('\nTraining Random Forest for feature importance...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)
print('  ✓ Training complete')

# Get feature importances
importance_df = pd.DataFrame({
    'feature': numeric_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f'  ✓ Extracted {len(importance_df)} feature importances')

# Compute correlations for context
correlations = train[numeric_features + ['high_severity']].corr()['high_severity'].drop('high_severity').abs()

# Create figure
print('\nCreating figure...')

fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.25)

# ============================================================================
# Panel A: Feature Importance
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

top_features = importance_df.head(20)

# Color by category
colors = []
for feat in top_features['feature']:
    feat_lower = feat.lower()
    if 'aadt' in feat_lower:
        colors.append('#4472C4')  # Blue - Traffic
    elif any(x in feat_lower for x in ['weather', 'temp', 'visibility', 'wind', 'humidity', 'pressure']):
        colors.append('#ED7D31')  # Orange - Weather
    elif any(x in feat_lower for x in ['speed', 'lane', 'junction']):
        colors.append('#70AD47')  # Green - Road
    elif any(x in feat_lower for x in ['hour', 'day', 'month', 'weekend', 'rush']):
        colors.append('#9E67AB')  # Purple - Temporal
    else:
        colors.append('#C0C0C0')  # Gray - Other

# Create bars
y_pos = np.arange(len(top_features))
bars = ax1.barh(y_pos, top_features['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_features['feature'], fontsize=11)
ax1.set_xlabel('Importance Score (Random Forest)', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Feature Importance Rankings (Top 20)',
             fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax1.text(row['importance'] + 0.005, i, f"{row['importance']:.3f}",
            va='center', ha='left', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4472C4', alpha=0.8, label='Traffic (AADT)'),
    Patch(facecolor='#ED7D31', alpha=0.8, label='Weather'),
    Patch(facecolor='#70AD47', alpha=0.8, label='Road Characteristics'),
    Patch(facecolor='#9E67AB', alpha=0.8, label='Temporal'),
    Patch(facecolor='#C0C0C0', alpha=0.8, label='Location/Other')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

# ============================================================================
# Panel B: Correlation Heatmap
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Get top features for correlation
top_feature_names = importance_df.head(12)['feature'].tolist()
features_for_corr = top_feature_names + ['high_severity']

# Compute correlation
corr_matrix = train[features_for_corr].corr()

# Plot
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    ax=ax2,
    vmin=-1,
    vmax=1
)
ax2.set_title('Panel B: Feature Correlations (Top Features)',
             fontsize=14, fontweight='bold', pad=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)

# Overall title
fig.suptitle('Feature Importance and Impact Analysis',
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / 'figure_2_combined.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_2_combined.png"}')

print('\n✅ Figure 2 complete!')

# Print top 10
print('\n' + '='*80)
print('TOP 10 MOST IMPORTANT FEATURES')
print('='*80)
for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
    corr = correlations.get(row['feature'], 0)
    print(f'{i:2d}. {row["feature"]:<25} Importance: {row["importance"]:.4f}  |  Correlation: {corr:.4f}')

plt.close()
