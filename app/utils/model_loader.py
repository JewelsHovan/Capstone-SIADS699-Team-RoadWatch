"""
ML Model Loading Utilities
Handles loading and caching of trained models for predictions
"""

import streamlit as st
from pathlib import Path
import pickle
import joblib
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_crash_severity_model():
    """
    Load trained crash-level severity prediction model

    Returns:
        Trained model object (sklearn/xgboost/etc.)
    """
    model_path = MODELS_DIR / "crash_severity_model.pkl"

    if not model_path.exists():
        st.warning(f"⚠️ Trained model not found at {model_path}. Using baseline model.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading crash severity model: {e}")
        return None


@st.cache_resource
def load_segment_risk_model():
    """
    Load trained segment-level risk prediction model

    Returns:
        Trained model object (sklearn/xgboost/statsmodels/etc.)
    """
    model_path = MODELS_DIR / "segment_risk_model.pkl"

    if not model_path.exists():
        st.warning(f"⚠️ Trained model not found at {model_path}. Using baseline model.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading segment risk model: {e}")
        return None


def predict_crash_severity(model, features: Dict[str, Any]) -> Dict[str, float]:
    """
    Predict crash severity using trained model

    Args:
        model: Trained model object (or None for baseline)
        features: Dictionary of feature values

    Returns:
        Dictionary with prediction results
    """
    if model is None:
        # Baseline heuristic prediction
        risk_score = baseline_crash_severity_prediction(features)
        return {
            'probability_high_severity': risk_score,
            'probability_low_severity': 1 - risk_score,
            'prediction': 'HIGH' if risk_score > 0.5 else 'LOW',
            'model_type': 'baseline'
        }

    # Use trained model
    try:
        # Convert features to DataFrame
        feature_cols = [
            'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'temperature', 'visibility', 'adverse_weather', 'low_visibility',
            'speed_limit', 'through_lanes', 'f_system', 'aadt'
        ]

        X = pd.DataFrame([features])[feature_cols]

        # Handle missing values
        X = X.fillna(X.median())

        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            prob_low = proba[0]
            prob_high = proba[1]
        else:
            # Model doesn't support probabilities, use decision function or predict
            pred = model.predict(X)[0]
            prob_high = pred if pred <= 1.0 else 1.0
            prob_low = 1 - prob_high

        return {
            'probability_high_severity': prob_high,
            'probability_low_severity': prob_low,
            'prediction': 'HIGH' if prob_high > 0.5 else 'LOW',
            'model_type': 'trained'
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fall back to baseline
        risk_score = baseline_crash_severity_prediction(features)
        return {
            'probability_high_severity': risk_score,
            'probability_low_severity': 1 - risk_score,
            'prediction': 'HIGH' if risk_score > 0.5 else 'LOW',
            'model_type': 'baseline_fallback'
        }


def predict_segment_risk(model, segment_features: pd.DataFrame) -> np.ndarray:
    """
    Predict crash risk for road segments using trained model

    Args:
        model: Trained model object (or None for baseline)
        segment_features: DataFrame with segment features

    Returns:
        Array of predicted crash counts
    """
    if model is None:
        # Baseline heuristic prediction
        return baseline_segment_risk_prediction(segment_features)

    # Use trained model
    try:
        feature_cols = [
            'speed_limit', 'through_lanes', 'f_system', 'urban_id', 'aadt',
            'speed_x_aadt', 'fsystem_x_urban', 'lanes_x_aadt'
        ]

        X = segment_features[feature_cols]

        # Handle missing values
        X = X.fillna(X.median())

        # Predict
        predictions = model.predict(X)

        # Ensure predictions are non-negative for count data
        predictions = np.maximum(predictions, 0)

        return predictions

    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fall back to baseline
        return baseline_segment_risk_prediction(segment_features)


def baseline_crash_severity_prediction(features: Dict[str, Any]) -> float:
    """
    Simple baseline heuristic for crash severity prediction

    Args:
        features: Dictionary of feature values

    Returns:
        Probability of high severity (0-1)
    """
    risk_score = 0.0

    # Weather risk
    if features.get('adverse_weather', 0) == 1:
        risk_score += 0.25
    if features.get('low_visibility', 0) == 1:
        risk_score += 0.20

    # Time risk
    hour = features.get('hour', 12)
    if hour >= 22 or hour <= 4:  # Night
        risk_score += 0.15
    if features.get('is_rush_hour', 0) == 1:
        risk_score += 0.10
    if features.get('is_weekend', 0) == 1 and (hour >= 22 or hour <= 4):  # Weekend night
        risk_score += 0.15

    # Road risk
    if features.get('speed_limit', 0) > 65:
        risk_score += 0.15
    if features.get('aadt', 0) > 50000:  # High traffic
        risk_score += 0.10

    # Add small random noise for realism
    risk_score += np.random.normal(0, 0.05)

    # Clip to valid probability range
    risk_score = np.clip(risk_score, 0, 1)

    return risk_score


def baseline_segment_risk_prediction(segment_features: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline heuristic for segment risk prediction

    Args:
        segment_features: DataFrame with segment features

    Returns:
        Array of predicted crash counts
    """
    # Baseline: Higher risk = high speed + high AADT + many lanes
    predictions = (
        segment_features['speed_limit'].fillna(0) / 10 * 0.3 +
        segment_features['aadt'].fillna(0) / 10000 * 0.4 +
        segment_features['through_lanes'].fillna(0) * 0.3 +
        np.random.normal(0, 0.5, len(segment_features))  # Add noise
    ).clip(lower=0)

    return predictions.values


def save_model(model, model_name: str):
    """
    Save a trained model to disk

    Args:
        model: Trained model object
        model_name: Name for the model file (without extension)
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"

    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a trained model

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata or None if not found
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    if not model_path.exists():
        return None

    info = {
        'model_path': str(model_path),
        'exists': True,
        'size_mb': model_path.stat().st_size / (1024 * 1024)
    }

    # Load metadata if available
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        info.update(metadata)

    return info


# Feature preprocessing utilities
def preprocess_crash_features(features: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess features for crash severity prediction

    Args:
        features: Raw feature dictionary

    Returns:
        Preprocessed DataFrame ready for model input
    """
    # Define feature columns in expected order
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
        'temperature', 'visibility', 'adverse_weather', 'low_visibility',
        'speed_limit', 'through_lanes', 'f_system', 'aadt'
    ]

    # Create DataFrame
    df = pd.DataFrame([features])[feature_cols]

    # Handle missing values
    df = df.fillna({
        'hour': 12,
        'day_of_week': 2,
        'is_weekend': 0,
        'is_rush_hour': 0,
        'temperature': 70,
        'visibility': 10,
        'adverse_weather': 0,
        'low_visibility': 0,
        'speed_limit': 60,
        'through_lanes': 2,
        'f_system': 4,
        'aadt': 10000
    })

    return df


def preprocess_segment_features(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features for segment risk prediction

    Args:
        segments: DataFrame with segment features

    Returns:
        Preprocessed DataFrame ready for model input
    """
    # Ensure required columns exist
    required_cols = ['speed_limit', 'through_lanes', 'f_system', 'urban_id', 'aadt']

    for col in required_cols:
        if col not in segments.columns:
            segments[col] = np.nan

    # Create interaction features if not present
    if 'speed_x_aadt' not in segments.columns:
        segments['speed_x_aadt'] = segments['speed_limit'] * segments['aadt']

    if 'fsystem_x_urban' not in segments.columns:
        segments['fsystem_x_urban'] = segments['f_system'] * segments['urban_id']

    if 'lanes_x_aadt' not in segments.columns:
        segments['lanes_x_aadt'] = segments['through_lanes'] * segments['aadt']

    return segments
