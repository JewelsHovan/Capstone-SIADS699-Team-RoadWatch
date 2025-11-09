#!/usr/bin/env python3
"""
ML Pipelines for Crash Prediction

Provides reproducible sklearn Pipelines that combine preprocessing and models.
Ensures train/val/test transformations are identical and prevents data leakage.

Usage:
    from ml_engineering.preprocessing.pipelines import create_crash_classifier_pipeline

    pipeline = create_crash_classifier_pipeline(
        numeric_features=['aadt', 'hour'],
        categorical_features=['weather_category']
    )
    pipeline.set_params(classifier=LogisticRegression())
    pipeline.fit(X_train, y_train)
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np


def create_crash_classifier_pipeline(numeric_features, categorical_features, model=None):
    """
    Create full preprocessing + classification pipeline for crash-level data

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        model: sklearn classifier (default: None, must be set later with set_params)

    Returns:
        sklearn Pipeline with 'preprocessor' and 'classifier' steps
    """
    # Numeric transformer: median imputation + standard scaling
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer: constant imputation + one-hot encoding
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)  # Can be None initially, set with set_params()
    ])

    return pipeline


def create_crash_regressor_pipeline(numeric_features, categorical_features, model=None):
    """
    Create full preprocessing + regression pipeline for segment-level data

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        model: sklearn regressor (default: None, must be set later with set_params)

    Returns:
        sklearn Pipeline with 'preprocessor' and 'regressor' steps
    """
    # Same preprocessing as classifier
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline


def get_feature_names(pipeline):
    """
    Extract feature names from a fitted pipeline

    Args:
        pipeline: Fitted sklearn Pipeline with ColumnTransformer

    Returns:
        List of feature names after transformation
    """
    # Get the preprocessor step
    preprocessor = pipeline.named_steps['preprocessor']

    feature_names = []

    # Iterate through transformers
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
        elif name == 'num':
            # Numeric features keep their names
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            onehot = transformer.named_steps['onehot']
            cat_features = onehot.get_feature_names_out(columns)
            feature_names.extend(cat_features)

    return feature_names
