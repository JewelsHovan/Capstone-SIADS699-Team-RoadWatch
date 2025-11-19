#!/usr/bin/env python3
"""
Quick test script to verify ML model integration with Streamlit app

This script tests that:
1. Models can be loaded
2. Predictions work correctly
3. Feature formatting is compatible

Run: python test_model_integration.py
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def test_model_loading():
    """Test that models load correctly"""
    print("\n" + "="*70)
    print("TEST 1: Model Loading")
    print("="*70)

    from utils.model_loader import load_crash_severity_model, load_segment_risk_model

    # Test crash severity model
    print("\nLoading crash severity model...")
    crash_model = load_crash_severity_model()
    if crash_model is None:
        print("⚠️  No trained model found - will use baseline")
        print("   Expected location: models/crash_severity_model.pkl")
    else:
        print(f"✅ Crash model loaded: {type(crash_model)}")

    # Test segment risk model
    print("\nLoading segment risk model...")
    segment_model = load_segment_risk_model()
    if segment_model is None:
        print("⚠️  No trained model found - will use baseline")
        print("   Expected location: models/segment_risk_model.pkl")
    else:
        print(f"✅ Segment model loaded: {type(segment_model)}")

    return crash_model, segment_model


def test_crash_prediction(model):
    """Test crash severity predictions"""
    print("\n" + "="*70)
    print("TEST 2: Crash Severity Predictions")
    print("="*70)

    from utils.model_loader import predict_crash_severity

    # Test case 1: Low risk scenario
    print("\nTest Case 1: Low Risk (day, clear weather, low speed)")
    features_low = {
        'hour': 14,
        'day_of_week': 2,
        'is_weekend': 0,
        'is_rush_hour': 0,
        'temperature': 75,
        'visibility': 10,
        'adverse_weather': 0,
        'low_visibility': 0,
        'speed_limit': 35,
        'through_lanes': 2,
        'f_system': 1,
        'aadt': 10000
    }

    result_low = predict_crash_severity(model, features_low)
    print(f"  Prediction: {result_low['prediction']}")
    print(f"  High Severity Probability: {result_low['probability_high_severity']:.3f}")
    print(f"  Model Type: {result_low['model_type']}")

    # Test case 2: High risk scenario
    print("\nTest Case 2: High Risk (night, bad weather, high speed)")
    features_high = {
        'hour': 2,
        'day_of_week': 5,
        'is_weekend': 1,
        'is_rush_hour': 0,
        'temperature': 32,
        'visibility': 2,
        'adverse_weather': 1,
        'low_visibility': 1,
        'speed_limit': 75,
        'through_lanes': 4,
        'f_system': 1,
        'aadt': 75000
    }

    result_high = predict_crash_severity(model, features_high)
    print(f"  Prediction: {result_high['prediction']}")
    print(f"  High Severity Probability: {result_high['probability_high_severity']:.3f}")
    print(f"  Model Type: {result_high['model_type']}")

    # Sanity check
    if result_high['probability_high_severity'] > result_low['probability_high_severity']:
        print("\n✅ Sanity check passed: High-risk scenario has higher probability")
    else:
        print("\n⚠️  Warning: Risk scores seem inverted")

    return result_low, result_high


def test_segment_prediction(model):
    """Test segment risk predictions"""
    print("\n" + "="*70)
    print("TEST 3: Segment Risk Predictions")
    print("="*70)

    import pandas as pd
    import numpy as np
    from utils.model_loader import predict_segment_risk

    # Create test segments
    test_segments = pd.DataFrame([
        {
            'speed_limit': 35,
            'through_lanes': 2,
            'f_system': 1,
            'urban_id': 12345,
            'aadt': 15000
        },
        {
            'speed_limit': 75,
            'through_lanes': 4,
            'f_system': 1,
            'urban_id': 12345,
            'aadt': 75000
        }
    ])

    print(f"\nPredicting risk for {len(test_segments)} segments...")
    predictions = predict_segment_risk(model, test_segments)

    for i, (idx, row) in enumerate(test_segments.iterrows()):
        print(f"\nSegment {i+1}:")
        print(f"  Speed: {row['speed_limit']} mph, Lanes: {row['through_lanes']}, AADT: {row['aadt']:,}")
        print(f"  Predicted Risk: {predictions[i]:.2f}")

    return predictions


def test_feature_compatibility():
    """Test that app features match training features"""
    print("\n" + "="*70)
    print("TEST 4: Feature Compatibility Check")
    print("="*70)

    try:
        # Try to import training features
        sys.path.insert(0, str(Path(__file__).parent))
        from ml_engineering.preprocessing import (
            CRASH_NUMERIC_FEATURES,
            CRASH_CATEGORICAL_FEATURES
        )

        training_features = set(CRASH_NUMERIC_FEATURES + CRASH_CATEGORICAL_FEATURES)

        # Get app features
        from utils.preprocessing import get_crash_feature_order
        app_features = set(get_crash_feature_order())

        print(f"\nTraining features: {len(training_features)}")
        print(f"App features: {len(app_features)}")

        # Check for mismatches
        only_in_training = training_features - app_features
        only_in_app = app_features - training_features

        if only_in_training:
            print(f"\n⚠️  Features in training but not in app:")
            for feat in sorted(only_in_training):
                print(f"     - {feat}")

        if only_in_app:
            print(f"\n⚠️  Features in app but not in training:")
            for feat in sorted(only_in_app):
                print(f"     - {feat}")

        if not only_in_training and not only_in_app:
            print("\n✅ Feature sets match perfectly!")
        else:
            print("\n⚠️  Feature mismatch detected - predictions may be incorrect")
            print("    Solution: Update app/utils/preprocessing.py to match training features")

    except ImportError as e:
        print(f"\n⚠️  Could not import training features: {e}")
        print("    This is okay if you're using the pipeline's built-in preprocessing")


def main():
    print("\n" + "="*70)
    print(" MODEL INTEGRATION TEST")
    print("="*70)
    print("\nThis script tests the integration between trained ML models")
    print("and the Streamlit app's prediction functions.")

    # Test 1: Model loading
    crash_model, segment_model = test_model_loading()

    # Test 2: Crash predictions
    test_crash_prediction(crash_model)

    # Test 3: Segment predictions
    test_segment_prediction(segment_model)

    # Test 4: Feature compatibility
    test_feature_compatibility()

    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)

    if crash_model is None:
        print("\n📝 Next Steps:")
        print("   1. Train a model: python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost")
        print("   2. Copy to app: cp models/artifacts/xgboost_calibrated.pkl models/crash_severity_model.pkl")
        print("   3. Test app: streamlit run app/app.py")
    else:
        print("\n✅ Model integration working!")
        print("   Models are loaded and predictions are functioning.")
        print("   Run the Streamlit app to see them in action:")
        print("   streamlit run app/app.py")

    print()


if __name__ == '__main__':
    main()
