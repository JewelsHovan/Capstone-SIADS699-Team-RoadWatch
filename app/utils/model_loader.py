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
from .preprocessing import (
    preprocess_crash_features,
    preprocess_segment_features,
    create_feature_vector,
    get_crash_feature_order,
    get_segment_feature_order
)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_crash_severity_model(model_name: str = "catboost_best_balanced"):
    """
    Load trained crash-level severity prediction model from production artifacts

    Args:
        model_name: Name of production model to load. Options:
            - "catboost_best_balanced" (default, best F1)
            - "random_forest_best_recall" (best recall for emergency response)
            - "lightgbm_best_auc" (best AUC)

    Returns:
        Tuple of (model, metadata) or (None, None) if not found
    """
    # Try production symlink first
    production_path = MODELS_DIR / "production" / model_name / "pipeline.pkl"
    metadata_path = MODELS_DIR / "production" / model_name / "metadata.json"

    # Fallback to legacy path
    legacy_path = MODELS_DIR / "crash_severity_model.pkl"

    if production_path.exists():
        try:
            model = joblib.load(production_path)
            metadata = None
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            return model, metadata
        except Exception as e:
            st.error(f"Error loading crash severity model: {e}")
            return None, None
    elif legacy_path.exists():
        try:
            with open(legacy_path, 'rb') as f:
                model = pickle.load(f)
            return model, None
        except Exception as e:
            st.error(f"Error loading crash severity model: {e}")
            return None, None
    else:
        st.warning(f"⚠️ Trained model not found. Using baseline model.")
        return None, None


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


def predict_crash_severity(model, features: Dict[str, Any], metadata: Optional[Dict] = None) -> Dict[str, float]:
    """
    Predict crash severity using trained model

    Args:
        model: Trained model object (sklearn pipeline) or None for baseline
        features: Dictionary of feature values
        metadata: Optional model metadata (contains threshold, metrics, etc.)

    Returns:
        Dictionary with prediction results
    """
    if model is None:
        # Baseline heuristic prediction
        processed = preprocess_crash_features(features)
        risk_score = baseline_crash_severity_prediction(processed)
        return {
            'probability_high_severity': risk_score,
            'probability_low_severity': 1 - risk_score,
            'prediction': 'HIGH' if risk_score > 0.5 else 'LOW',
            'model_type': 'baseline',
            'threshold': 0.5
        }

    # Get threshold from metadata if available
    threshold = 0.5
    if metadata and 'metrics' in metadata:
        threshold = metadata['metrics'].get('threshold', 0.5)

    # Use trained model (sklearn pipeline with preprocessing built-in)
    try:
        # Preprocess features to expand from 12 simple features to 32 model features
        # Pipeline then handles categorical encoding and scaling
        processed = preprocess_crash_features(features)
        X = pd.DataFrame([processed])

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

        model_name = metadata.get('model_name', 'trained') if metadata else 'trained'

        return {
            'probability_high_severity': prob_high,
            'probability_low_severity': prob_low,
            'prediction': 'HIGH' if prob_high >= threshold else 'LOW',
            'model_type': model_name,
            'threshold': threshold
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fall back to baseline
        processed = preprocess_crash_features(features)
        risk_score = baseline_crash_severity_prediction(processed)
        return {
            'probability_high_severity': risk_score,
            'probability_low_severity': 1 - risk_score,
            'prediction': 'HIGH' if risk_score > 0.5 else 'LOW',
            'model_type': 'baseline_fallback',
            'threshold': 0.5
        }


def predict_segment_risk(model, segment_features: pd.DataFrame) -> np.ndarray:
    """
    Predict crash risk for road segments using trained model

    Args:
        model: Trained model object (sklearn pipeline) or None for baseline
        segment_features: DataFrame with segment features

    Returns:
        Array of predicted crash counts
    """
    if model is None:
        # Baseline heuristic prediction
        return baseline_segment_risk_prediction(segment_features)

    # Use trained model (sklearn pipeline with preprocessing built-in)
    try:
        # Pipeline expects DataFrame input and handles preprocessing internally
        predictions = model.predict(segment_features)

        # Ensure predictions are non-negative for count data
        predictions = np.maximum(predictions, 0)

        return predictions

    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fall back to baseline
        return baseline_segment_risk_prediction(segment_features)


def baseline_crash_severity_prediction(features: Dict[str, float]) -> float:
    """
    Simple baseline heuristic for crash severity prediction (Deterministic)

    Baseline Rule-Based Model:
    - Weather conditions: adverse weather and low visibility increase risk
    - Time factors: night driving and rush hour increase risk
    - Road factors: high speed limits and high traffic increase risk

    Note: This is a deterministic baseline model. For production, replace with
    a trained ML model that has been validated on historical crash data.

    Args:
        features: Dictionary of preprocessed feature values (already floats)

    Returns:
        Probability of high severity (0-1)
    """
    risk_score = 0.0

    # Weather risk
    if features['adverse_weather'] == 1:
        risk_score += 0.25
    if features['low_visibility'] == 1:
        risk_score += 0.20

    # Time risk
    hour = features['hour']
    if hour >= 22 or hour <= 4:  # Night
        risk_score += 0.15
    if features['is_rush_hour'] == 1:
        risk_score += 0.10
    if features['is_weekend'] == 1 and (hour >= 22 or hour <= 4):  # Weekend night
        risk_score += 0.15

    # Road risk (using preprocessed feature names)
    if features.get('hpms_speed_limit', 0) > 65:
        risk_score += 0.15
    if features.get('aadt', 0) > 50000:  # High traffic
        risk_score += 0.10

    # Clip to valid probability range (no random noise for reproducibility)
    risk_score = np.clip(risk_score, 0, 1)

    return risk_score


def baseline_segment_risk_prediction(segment_features: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline heuristic for segment risk prediction (Deterministic)

    Baseline Rule-Based Model:
    - Speed limit: 30% weight (normalized to 0-10 scale)
    - Traffic volume (AADT): 40% weight (normalized to 0-10K scale)
    - Number of lanes: 30% weight

    Note: This is a deterministic baseline model. For production, replace with
    a trained ML model that has been validated on historical crash data.

    Args:
        segment_features: DataFrame with segment features

    Returns:
        Array of predicted crash counts
    """
    # Baseline: Higher risk = high speed + high AADT + many lanes (no random noise)
    predictions = (
        segment_features['speed_limit'].fillna(0) / 10 * 0.3 +
        segment_features['aadt'].fillna(0) / 10000 * 0.4 +
        segment_features['through_lanes'].fillna(0) * 0.3
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
