#!/usr/bin/env python3
"""
Figure 2: Feature Importance and Impact Analysis

Creates comprehensive feature importance visualizations:
- Panel A: SHAP summary plot showing feature importance and directionality
- Panel B: Correlation heatmap of top features

This figure identifies which factors most strongly predict crash severity.

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

# Add project root to path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engineering.utils.persistence import load_model_artifact

# Try to import SHAP (install if needed: pip install shap)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("WARNING: SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 2: FEATURE IMPORTANCE AND IMPACT ANALYSIS')
print('='*80)

# ============================================================================
# Load Data and Model
# ============================================================================
print('\nLoading test dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
test = pd.read_csv(test_path)
print(f'  ✓ Loaded {len(test):,} test samples')

print('\nLoading best model (Random Forest)...')
model_path = 'models/artifacts/random_forest_calibrated_optimized_20251110_183042'
pipeline, metadata = load_model_artifact(model_path)
print(f'  ✓ Loaded model: {metadata["model_name"]}')
print(f'  ✓ Test AUC: {metadata["metrics"]["auc"]:.4f}')
print(f'  ✓ Test F1: {metadata["metrics"]["f1"]:.4f}')

# Get feature columns
feature_cols = metadata['feature_cols']
print(f'  ✓ Features: {len(feature_cols)}')

# Prepare data
X_test = test[feature_cols].copy()
y_test = test['high_severity'].copy()

# ============================================================================
# Method 1: SHAP Values (if available)
# ============================================================================
if SHAP_AVAILABLE:
    print('\nComputing SHAP values...')
    print('  (This may take several minutes for large datasets)')

    # Get the base estimator (before calibration)
    if hasattr(pipeline, 'base_estimator'):
        base_model = pipeline.base_estimator
    else:
        # Try to get the last step that's not a calibrator
        base_model = pipeline

    # Sample data for SHAP (use subset for speed)
    SHAP_SAMPLE = min(1000, len(X_test))
    X_shap = X_test.sample(n=SHAP_SAMPLE, random_state=42)

    print(f'  Computing SHAP on {SHAP_SAMPLE} samples...')

    try:
        # Create explainer
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_shap)

        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        print('  ✓ SHAP computation complete')

        # Create SHAP summary plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        shap.summary_plot(
            shap_values,
            X_shap,
            plot_type="dot",
            show=False,
            max_display=20
        )

        plt.title('Feature Impact on Crash Severity Prediction (SHAP Analysis)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'figure_2a_shap_importance.png', dpi=300, bbox_inches='tight')
        print(f'  ✓ Saved: {OUTPUT_DIR / "figure_2a_shap_importance.png"}')
        plt.close()

        # Get mean absolute SHAP values for ranking
        shap_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print('\nTop 15 Features by SHAP Importance:')
        print(shap_importance.head(15).to_string(index=False))

    except Exception as e:
        print(f'  ⚠ SHAP computation failed: {e}')
        print('  Falling back to Random Forest feature importance')
        SHAP_AVAILABLE = False

# ============================================================================
# Method 2: Random Forest Feature Importance (fallback or supplement)
# ============================================================================
if not SHAP_AVAILABLE or True:  # Always compute as backup
    print('\nComputing Random Forest feature importance...')

    # Get feature importances from the model
    if hasattr(pipeline, 'base_estimator'):
        base_model = pipeline.base_estimator
    elif hasattr(pipeline, 'estimator'):
        base_model = pipeline.estimator
    else:
        base_model = pipeline

    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Plot top 20 features
        fig, ax = plt.subplots(figsize=(10, 12))
        top_features = importance_df.head(20)

        # Color by category
        colors = []
        for feat in top_features['feature']:
            if 'aadt' in feat.lower() or 'distance' in feat.lower():
                colors.append('steelblue')  # Traffic
            elif 'weather' in feat.lower() or 'temp' in feat.lower() or 'visibility' in feat.lower() or 'wind' in feat.lower() or 'humidity' in feat.lower() or 'pressure' in feat.lower():
                colors.append('coral')  # Weather
            elif 'speed' in feat.lower() or 'lane' in feat.lower() or 'functional' in feat.lower() or 'road' in feat.lower():
                colors.append('lightgreen')  # Road
            elif 'hour' in feat.lower() or 'day' in feat.lower() or 'month' in feat.lower() or 'weekend' in feat.lower() or 'rush' in feat.lower():
                colors.append('mediumpurple')  # Temporal
            else:
                colors.append('lightgray')  # Location/Other

        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 20 Features by Random Forest Importance',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='Traffic'),
            Patch(facecolor='coral', label='Weather'),
            Patch(facecolor='lightgreen', label='Road Characteristics'),
            Patch(facecolor='mediumpurple', label='Temporal'),
            Patch(facecolor='lightgray', label='Location/Other')
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'figure_2b_rf_importance.png', dpi=300, bbox_inches='tight')
        print(f'  ✓ Saved: {OUTPUT_DIR / "figure_2b_rf_importance.png"}')
        plt.close()

        print('\nTop 15 Features by RF Importance:')
        print(importance_df.head(15).to_string(index=False))

# ============================================================================
# Correlation Heatmap of Top Features
# ============================================================================
print('\nCreating correlation heatmap...')

# Select top features (from RF importance or SHAP)
if 'importance_df' in locals():
    top_feature_names = importance_df.head(15)['feature'].tolist()
else:
    # Use default important features
    top_feature_names = [f for f in feature_cols if any(x in f.lower() for x in
                        ['aadt', 'speed', 'hour', 'weather', 'visibility', 'temp', 'lane'])][:15]

# Add target
features_for_corr = top_feature_names + ['high_severity']
features_for_corr = [f for f in features_for_corr if f in test.columns]

# Filter to only numeric columns for correlation
numeric_features_for_corr = test[features_for_corr].select_dtypes(include=[np.number]).columns.tolist()

# Compute correlation matrix (only numeric features)
corr_matrix = test[numeric_features_for_corr].corr()

# Plot
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax,
    vmin=-1,
    vmax=1
)
ax.set_title('Feature Correlation Matrix (Top Features)',
            fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_2c_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_2c_correlation_heatmap.png"}')
plt.close()

# ============================================================================
# Combined Figure (for report)
# ============================================================================
print('\nCreating combined figure...')

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

# Left: Feature importance
ax1 = fig.add_subplot(gs[0, 0])
if 'importance_df' in locals():
    top_features = importance_df.head(15)

    # Same color coding as before
    colors = []
    for feat in top_features['feature']:
        if 'aadt' in feat.lower() or 'distance' in feat.lower():
            colors.append('steelblue')
        elif 'weather' in feat.lower() or 'temp' in feat.lower() or 'visibility' in feat.lower() or 'wind' in feat.lower():
            colors.append('coral')
        elif 'speed' in feat.lower() or 'lane' in feat.lower() or 'functional' in feat.lower() or 'road' in feat.lower():
            colors.append('lightgreen')
        elif 'hour' in feat.lower() or 'day' in feat.lower() or 'month' in feat.lower() or 'weekend' in feat.lower() or 'rush' in feat.lower():
            colors.append('mediumpurple')
        else:
            colors.append('lightgray')

    ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Importance Score', fontsize=11)
    ax1.set_title('Panel A: Feature Importance Rankings',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

# Right: Correlation heatmap (smaller version)
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(
    corr_matrix,
    annot=False,  # Skip annotations for cleaner look
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax2,
    vmin=-1,
    vmax=1
)
ax2.set_title('Panel B: Feature Correlations',
             fontsize=13, fontweight='bold', pad=15)

plt.suptitle('Feature Importance and Impact Analysis',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_2_combined.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_2_combined.png"}')
plt.close()

print('\n✅ Figure 2 complete!')
print(f'\nOutputs saved to: {OUTPUT_DIR}/')
if SHAP_AVAILABLE:
    print('  - figure_2a_shap_importance.png')
print('  - figure_2b_rf_importance.png')
print('  - figure_2c_correlation_heatmap.png')
print('  - figure_2_combined.png (recommended for report)')
