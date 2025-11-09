#!/usr/bin/env python3
"""
Dataset Analysis - EDA and Feature Analysis

Analyzes both crash-level and segment-level datasets:
- Basic statistics and distributions
- Feature correlations
- Missing value analysis
- Class balance
- Feature importance (using RandomForest)

Usage:
  python -m analysis.dataset_analysis --dataset crash
  python -m analysis.dataset_analysis --dataset segment
  python -m analysis.dataset_analysis --both
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import CRASH_LEVEL_ML, SEGMENT_LEVEL_ML

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_dataset(dataset_type):
    """Load train dataset"""
    if dataset_type == 'crash':
        data_dir = CRASH_LEVEL_ML
        target = 'high_severity'
    else:
        data_dir = SEGMENT_LEVEL_ML
        target = 'crash_count'

    train_file = data_dir / 'train_latest.csv'

    if not train_file.exists():
        print(f"❌ File not found: {train_file}")
        return None, None

    print(f"Loading {dataset_type} dataset from {train_file}...")
    df = pd.read_csv(train_file)
    print(f"✓ Loaded {len(df):,} samples")

    return df, target

def basic_stats(df, dataset_type):
    """Print basic statistics"""
    print(f"\n{'='*70}")
    print(f"BASIC STATISTICS - {dataset_type.upper()}")
    print(f"{'='*70}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")

    # Missing values
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percent': missing_pct[missing > 0]
    }).sort_values('Percent', ascending=False)

    if len(missing_df) > 0:
        print(missing_df.head(10))
    else:
        print("  No missing values!")

def target_distribution(df, target, dataset_type):
    """Analyze target variable distribution"""
    print(f"\n{'='*70}")
    print(f"TARGET DISTRIBUTION - {target}")
    print(f"{'='*70}")

    if target == 'high_severity':
        # Binary classification
        counts = df[target].value_counts()
        print(f"\nClass balance:")
        print(f"  Low severity (0):  {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
        print(f"  High severity (1): {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
    else:
        # Regression (crash counts)
        print(f"\nCrash count statistics:")
        print(df[target].describe())
        print(f"\nZero crashes: {(df[target] == 0).sum():,} ({(df[target] == 0).sum()/len(df)*100:.1f}%)")
        print(f"With crashes:  {(df[target] > 0).sum():,} ({(df[target] > 0).sum()/len(df)*100:.1f}%)")

def feature_correlations(df, target, dataset_type, top_n=15):
    """Analyze feature correlations with target"""
    print(f"\n{'='*70}")
    print(f"TOP {top_n} FEATURES CORRELATED WITH {target.upper()}")
    print(f"{'='*70}")

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target]

    if len(numeric_cols) == 0:
        print("No numeric features found!")
        return

    # Calculate correlations
    correlations = df[numeric_cols + [target]].corr()[target].drop(target)
    correlations = correlations.abs().sort_values(ascending=False)

    print(f"\nTop {top_n} features:")
    for i, (feat, corr) in enumerate(correlations.head(top_n).items(), 1):
        print(f"{i:2d}. {feat:35s} {corr:.4f}")

    return correlations.head(top_n)

def feature_importance_analysis(df, target, dataset_type, top_n=15):
    """Calculate feature importance using RandomForest"""
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE - Random Forest")
    print(f"{'='*70}")

    # Prepare data
    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    # Handle missing values (simple median imputation)
    X = X.fillna(X.median())

    print(f"\nTraining Random Forest on {len(X):,} samples with {X.shape[1]} features...")

    # Train model
    if target == 'high_severity':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X, y)

    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\nTop {top_n} most important features:")
    for i, (feat, imp) in enumerate(importances.head(top_n).items(), 1):
        print(f"{i:2d}. {feat:35s} {imp:.4f}")

    return importances.head(top_n)

def analyze_dataset(dataset_type):
    """Run full analysis on dataset"""
    print(f"\n{'#'*70}")
    print(f"# DATASET ANALYSIS: {dataset_type.upper()}")
    print(f"{'#'*70}")

    # Load data
    df, target = load_dataset(dataset_type)
    if df is None:
        return

    # Run analyses
    basic_stats(df, dataset_type)
    target_distribution(df, target, dataset_type)
    feature_correlations(df, target, dataset_type)
    feature_importance_analysis(df, target, dataset_type)

    print(f"\n{'='*70}")
    print(f"✓ Analysis complete for {dataset_type} dataset")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze ML datasets")
    parser.add_argument('--dataset', choices=['crash', 'segment'],
                       help='Dataset to analyze')
    parser.add_argument('--both', action='store_true',
                       help='Analyze both datasets')

    args = parser.parse_args()

    if args.both:
        analyze_dataset('crash')
        analyze_dataset('segment')
    elif args.dataset:
        analyze_dataset(args.dataset)
    else:
        print("Please specify --dataset crash, --dataset segment, or --both")
        parser.print_help()

if __name__ == "__main__":
    main()
