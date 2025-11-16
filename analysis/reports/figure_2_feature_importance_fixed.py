#!/usr/bin/env python3
"""
Figure 2: Feature Importance (FIXED VERSION)

Critical fix: Ensure Panel A actually displays the feature importance bars

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

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engineering.utils.persistence import load_model_artifact

OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 2: FEATURE IMPORTANCE (FIXED VERSION)')
print('='*80)

# Load test data
print('\nLoading test dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
test = pd.read_csv(test_path)
print(f'  ✓ Loaded {len(test):,} test samples')

# Load model
print('\nLoading Random Forest model...')
model_path = 'models/artifacts/random_forest_calibrated_optimized_20251110_183042'
pipeline, metadata = load_model_artifact(model_path)
print(f'  ✓ Model loaded: {metadata["model_name"]}')

feature_cols = metadata['feature_cols']
X_test = test[feature_cols].copy()

# Get feature importances
print('\nExtracting feature importance...')

# CalibratedClassifierCV has .calibrated_classifiers_ attribute
# Each calibrated classifier has .estimator attribute (the base model)
base_model = None
if hasattr(pipeline, 'calibrated_classifiers_'):
    # Get the first calibrated classifier's base estimator
    base_model = pipeline.calibrated_classifiers_[0].estimator
    print('  ✓ Extracted base model from CalibratedClassifierCV')

    # If it's a Pipeline, get the last step (the actual classifier)
    if hasattr(base_model, 'steps'):
        base_model = base_model.steps[-1][1]
        print('  ✓ Extracted classifier from Pipeline')
elif hasattr(pipeline, 'base_estimator'):
    base_model = pipeline.base_estimator
elif hasattr(pipeline, 'estimator'):
    base_model = pipeline.estimator
else:
    base_model = pipeline

# If still a pipeline, get the last step
if hasattr(base_model, 'steps'):
    base_model = base_model.steps[-1][1]
    print('  ✓ Extracted final estimator from Pipeline')

if base_model is not None and hasattr(base_model, 'feature_importances_'):
    importances = base_model.feature_importances_

    # Check if lengths match (might have feature selection in pipeline)
    if len(importances) != len(feature_cols):
        print(f'  ⚠ Length mismatch: {len(importances)} importances vs {len(feature_cols)} features')
        # Use generic feature names if mismatch
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    else:
        feature_names = feature_cols

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f'  ✓ Extracted importance for {len(importance_df)} features')
else:
    print('  ✗ Could not extract feature importances')
    print(f'  Model type: {type(base_model)}')
    if base_model is not None:
        print(f'  Has feature_importances_: {hasattr(base_model, "feature_importances_")}')
        print(f'  Available attributes: {[a for a in dir(base_model) if not a.startswith("_")][:10]}')
    importance_df = None

# Create combined figure
print('\nCreating combined figure...')

fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.25)

# ============================================================================
# Panel A: Feature Importance (LEFT)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

if importance_df is not None:
    top_features = importance_df.head(20)

    # Color by category
    colors = []
    for feat in top_features['feature']:
        feat_lower = feat.lower()
        if 'aadt' in feat_lower or 'distance_to_aadt' in feat_lower:
            colors.append('#4472C4')  # Blue - Traffic
        elif any(x in feat_lower for x in ['weather', 'temp', 'visibility', 'wind', 'humidity', 'pressure', 'precip']):
            colors.append('#ED7D31')  # Orange - Weather
        elif any(x in feat_lower for x in ['speed', 'lane', 'functional', 'road', 'junction']):
            colors.append('#70AD47')  # Green - Road
        elif any(x in feat_lower for x in ['hour', 'day', 'month', 'weekend', 'rush']):
            colors.append('#9E67AB')  # Purple - Temporal
        else:
            colors.append('#C0C0C0')  # Gray - Other

    # Create horizontal bars
    y_pos = np.arange(len(top_features))
    bars = ax1.barh(y_pos, top_features['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features['feature'], fontsize=11)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Panel A: Feature Importance Rankings (Top 20)',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax1.text(row['importance'] + 0.003, i, f"{row['importance']:.3f}",
                va='center', ha='left', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', alpha=0.8, label='Traffic'),
        Patch(facecolor='#ED7D31', alpha=0.8, label='Weather'),
        Patch(facecolor='#70AD47', alpha=0.8, label='Road Characteristics'),
        Patch(facecolor='#9E67AB', alpha=0.8, label='Temporal'),
        Patch(facecolor='#C0C0C0', alpha=0.8, label='Location/Other')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

else:
    ax1.text(0.5, 0.5, 'Feature importance not available',
            ha='center', va='center', fontsize=14, color='red')

# ============================================================================
# Panel B: Correlation Heatmap (RIGHT)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Get top numeric features for correlation
if importance_df is not None:
    top_feature_names = importance_df.head(12)['feature'].tolist()
else:
    top_feature_names = [f for f in feature_cols if any(x in f.lower() for x in
                        ['aadt', 'speed', 'hour', 'visibility', 'temp', 'lane'])][:12]

features_for_corr = top_feature_names + ['high_severity']
features_for_corr = [f for f in features_for_corr if f in test.columns]

# Filter to numeric only
numeric_features_for_corr = test[features_for_corr].select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = test[numeric_features_for_corr].corr()

# Plot heatmap
sns.heatmap(
    corr_matrix,
    annot=False,  # No annotations for cleaner look
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

print('\n✅ Figure 2 (fixed) complete!')

# Print top 10 features
print('\n' + '='*80)
print('TOP 10 MOST IMPORTANT FEATURES')
print('='*80)
if importance_df is not None:
    for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f'{i:2d}. {row["feature"]:<30} {row["importance"]:.4f}')

plt.close()
