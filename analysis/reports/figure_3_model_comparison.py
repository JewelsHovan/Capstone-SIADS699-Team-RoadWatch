#!/usr/bin/env python3
"""
Figure 3: Model Performance Comparison

Creates comprehensive model comparison visualizations:
- Panel A: Grouped bar chart of AUC-ROC and F1 scores
- Panel B: ROC curves for all models overlaid
- Panel C: Precision-Recall curves (optional)

Demonstrates that ML models successfully predict crash severity.

Author: Capstone Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CRASH_LEVEL_ML = PROJECT_ROOT / 'data' / 'gold' / 'ml_datasets' / 'crash_level'

# Add project root to path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engineering.utils.persistence import load_model_artifact

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 3: MODEL PERFORMANCE COMPARISON')
print('='*80)

# ============================================================================
# Load Models and Metrics
# ============================================================================
print('\nLoading trained models...')

# Model paths
models = {
    'Logistic Regression': 'models/artifacts/logistic_regression_calibrated_20251110_183023',
    'Random Forest': 'models/artifacts/random_forest_calibrated_optimized_20251110_183042',
    'XGBoost': 'models/artifacts/xgboost_tuned_calibrated_20251109_190155'
}

# Load metadata
model_metrics = {}
for name, path in models.items():
    metadata_path = Path(path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    model_metrics[name] = metadata['metrics']
    print(f'  ✓ {name}:')
    print(f'    - AUC: {metadata["metrics"]["auc"]:.4f}')
    print(f'    - F1:  {metadata["metrics"]["f1"]:.4f}')

# ============================================================================
# Load Test Data
# ============================================================================
print('\nLoading test dataset...')
test_path = CRASH_LEVEL_ML / 'test_latest.csv'
test = pd.read_csv(test_path)
print(f'  ✓ Loaded {len(test):,} test samples')

# Get features
with open(Path(models['Random Forest']) / 'metadata.json', 'r') as f:
    metadata = json.load(f)
feature_cols = metadata['feature_cols']

X_test = test[feature_cols].copy()
y_test = test['high_severity'].copy()

# ============================================================================
# Generate Predictions for All Models
# ============================================================================
print('\nGenerating predictions...')

predictions = {}
for name, path in models.items():
    print(f'  Predicting with {name}...')
    pipeline, _ = load_model_artifact(path)

    # Get probability predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    predictions[name] = y_pred_proba
    print(f'    ✓ Complete')

# ============================================================================
# Panel A: Grouped Bar Chart (AUC and F1)
# ============================================================================
print('\nCreating bar chart comparison...')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data
models_list = list(model_metrics.keys())
auc_scores = [model_metrics[m]['auc'] for m in models_list]
f1_scores = [model_metrics[m]['f1'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35

# Panel A: Grouped bars
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, auc_scores, width, label='AUC-ROC', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral', alpha=0.8)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Model Performance Metrics', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, fontsize=11)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.0])

# Add horizontal line at 0.5 (random baseline for AUC)
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Random baseline')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel B: ROC Curves
ax2 = axes[1]

# Plot ROC curve for each model
colors = ['green', 'blue', 'purple']
for (name, y_pred_proba), color in zip(predictions.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, color=color, lw=2.5, alpha=0.8,
            label=f'{name} (AUC = {roc_auc:.3f})')

# Plot random classifier
ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')

ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: ROC Curves Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax2.grid(alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.suptitle('Model Performance Comparison on Test Set (2020)',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_3_model_comparison.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_3_model_comparison.png"}')
plt.close()

# ============================================================================
# Additional: Precision-Recall Curves (Supplementary)
# ============================================================================
print('\nCreating Precision-Recall curves...')

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['green', 'blue', 'purple']
for (name, y_pred_proba), color in zip(predictions.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    ax.plot(recall, precision, color=color, lw=2.5, alpha=0.8,
           label=f'{name}')

# Baseline (no skill)
baseline = y_test.mean()
ax.plot([0, 1], [baseline, baseline], 'k--', lw=1.5, alpha=0.5,
       label=f'No Skill (baseline = {baseline:.3f})')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves (Critical for Imbalanced Data)',
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure_3_pr_curves.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {OUTPUT_DIR / "figure_3_pr_curves.png"}')
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print('\n' + '='*80)
print('MODEL PERFORMANCE SUMMARY')
print('='*80)

summary_df = pd.DataFrame({
    'Model': models_list,
    'AUC-ROC': auc_scores,
    'F1 Score': f1_scores,
    'Accuracy': [model_metrics[m]['accuracy'] for m in models_list],
    'Precision': [model_metrics[m]['precision'] for m in models_list],
    'Recall': [model_metrics[m]['recall'] for m in models_list]
})

print('\n' + summary_df.to_string(index=False))

# Find best model
best_auc_model = summary_df.loc[summary_df['AUC-ROC'].idxmax(), 'Model']
best_f1_model = summary_df.loc[summary_df['F1 Score'].idxmax(), 'Model']

print(f'\n📊 Best AUC-ROC: {best_auc_model} ({summary_df["AUC-ROC"].max():.4f})')
print(f'📊 Best F1 Score: {best_f1_model} ({summary_df["F1 Score"].max():.4f})')

# Calculate improvement over baseline
lr_auc = model_metrics['Logistic Regression']['auc']
rf_auc = model_metrics['Random Forest']['auc']
improvement = ((rf_auc - lr_auc) / lr_auc) * 100

print(f'\n📈 Random Forest improves AUC by {improvement:.1f}% over Logistic Regression baseline')

print('\n✅ Figure 3 complete!')
print(f'\nOutputs saved to: {OUTPUT_DIR}/')
print('  - figure_3_model_comparison.png (recommended for report)')
print('  - figure_3_pr_curves.png (supplementary)')
