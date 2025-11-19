# Model Integration Guide: Connect ML Training to Streamlit App

**Goal:** Replace baseline heuristics in the Streamlit app with trained ML models from `ml_engineering/train_with_mlflow.py`

---

## Current Situation

**App (Streamlit):**
- Uses `app/utils/model_loader.py` to load models
- Expects models at:
  - `models/crash_severity_model.pkl` (crash severity prediction)
  - `models/segment_risk_model.pkl` (segment risk prediction)
- Falls back to baseline heuristics if models not found
- Uses `app/utils/preprocessing.py` for feature preprocessing

**Training Pipeline:**
- Located in `ml_engineering/train_with_mlflow.py`
- Saves models to `models/artifacts/` with descriptive names
- Uses `ml_engineering/preprocessing/` for pipeline creation
- Tracks experiments in MLflow

---

## Integration Steps

### Step 1: Train Your Models

Run the training script to create production-ready models:

```bash
cd /Users/julienh/Desktop/MADS/Capstone

# Option A: Train baseline models (Logistic Regression + Random Forest)
python -m ml_engineering.train_with_mlflow --dataset crash --model baseline

# Option B: Train XGBoost (recommended for best performance)
python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost

# Option C: Train XGBoost with hyperparameter tuning (best quality, takes longer)
python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost --tune

# Option D: Train all models for comparison
python -m ml_engineering.train_with_mlflow --dataset crash --model all
```

**What happens:**
- Models are trained on crash-level dataset
- Calibrated for better probability estimates
- Evaluated on test set
- Saved to `models/artifacts/` directory
- Logged to MLflow for tracking

**Expected output files:**
- `models/artifacts/logistic_regression_calibrated.pkl`
- `models/artifacts/random_forest_calibrated_optimized.pkl`
- `models/artifacts/xgboost_calibrated.pkl`
- `models/artifacts/xgboost_tuned_calibrated.pkl` (if using --tune)

---

### Step 2: Choose Best Model and Copy to App Location

After training, review the results and select the best model:

```bash
# View MLflow UI to compare models
mlflow ui
# Open http://localhost:5000 in browser
```

**Metrics to compare:**
- ROC-AUC (higher is better, target: >0.75)
- Precision/Recall balance
- Calibration quality

**Copy your best model to the app:**

```bash
# Example: Using tuned XGBoost (usually best performance)
cp models/artifacts/xgboost_tuned_calibrated.pkl models/crash_severity_model.pkl

# Or if you prefer Random Forest:
cp models/artifacts/random_forest_calibrated_optimized.pkl models/crash_severity_model.pkl
```

**What this does:**
- Renames the trained model to the expected filename
- Places it in the location where the app looks for models
- App will automatically load this instead of using baseline

---

### Step 3: Verify Model Integration

Test that the app loads the trained model:

```bash
# Run a quick Python test
python -c "
import sys
sys.path.insert(0, '/Users/julienh/Desktop/MADS/Capstone/app')
from utils.model_loader import load_crash_severity_model
model = load_crash_severity_model()
print('✅ Model loaded successfully!' if model else '❌ Model not found')
print(f'Model type: {type(model)}')
"
```

Expected output:
```
✅ Model loaded successfully!
Model type: <class 'sklearn.calibration.CalibratedClassifierCV'>
```

---

### Step 4: Test Predictions in App

Run the Streamlit app and test predictions:

```bash
cd /Users/julienh/Desktop/MADS/Capstone
streamlit run app/app.py
```

**Pages to test:**
1. **Page 7: Crash Severity Predictor**
   - Should now show trained model predictions instead of baseline
   - Probabilities should be well-calibrated (not all 0.5 or 1.0)
   - Should display "Model: trained" or similar indicator

2. **Check for warnings:**
   - Look for "⚠️ Using baseline model" warnings
   - Should be gone if model loaded correctly

---

## Feature Alignment (Critical!)

### Current Issue
Your training pipeline uses `ml_engineering/preprocessing/` but the app uses `app/utils/preprocessing.py`.

**They MUST be aligned** or predictions will be wrong!

### Verify Alignment

Check that both use the same features in the same order:

```bash
# Check training features
python -c "
from ml_engineering.preprocessing import CRASH_NUMERIC_FEATURES, CRASH_CATEGORICAL_FEATURES
print('Training numeric features:', len(CRASH_NUMERIC_FEATURES))
print('Training categorical features:', len(CRASH_CATEGORICAL_FEATURES))
print(CRASH_NUMERIC_FEATURES[:5])
"

# Check app features
python -c "
import sys
sys.path.insert(0, '/Users/julienh/Desktop/MADS/Capstone/app')
from utils.preprocessing import get_crash_feature_order
features = get_crash_feature_order()
print('App features:', len(features))
print(features[:5])
"
```

**If they don't match**, you have two options:

**Option A: Update app preprocessing to match training** (Recommended)
- Modify `app/utils/preprocessing.py` to use the exact same features as training
- Update `get_crash_feature_order()` to match `CRASH_NUMERIC_FEATURES + CRASH_CATEGORICAL_FEATURES`

**Option B: Retrain models with app features**
- Modify `ml_engineering/preprocessing/` to match app expectations
- Retrain models

---

## Troubleshooting

### Issue: "Model not found" warning in app

**Solution:**
```bash
# Check model file exists
ls -lh models/crash_severity_model.pkl

# If not, copy from artifacts:
cp models/artifacts/xgboost_calibrated.pkl models/crash_severity_model.pkl
```

### Issue: "Error loading model" or pickle error

**Causes:**
- Model trained with different scikit-learn version
- Incompatible dependencies

**Solution:**
```bash
# Check versions match between training and app
pip list | grep -E "scikit-learn|xgboost|numpy|pandas"

# If different, recreate environment or retrain
```

### Issue: Predictions are all 0.5 or seem wrong

**Causes:**
- Feature order mismatch
- Preprocessing mismatch

**Solution:**
1. Verify feature alignment (see above)
2. Check preprocessing steps match exactly
3. Test with known inputs and compare to MLflow logged predictions

### Issue: Model performs worse in app than in training

**Causes:**
- Different preprocessing in app vs training
- Data drift (app data different from training data)
- Missing calibration

**Solution:**
1. Ensure preprocessing is identical
2. Check model includes calibration wrapper
3. Verify using the `*_calibrated.pkl` version of the model

---

## Advanced: Model Update Workflow

For production deployments, establish this workflow:

### 1. Training Phase
```bash
# Retrain models monthly or when new data available
python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost --tune

# Review results in MLflow
mlflow ui
```

### 2. Model Selection
- Compare new model to current production model
- Check for performance improvements
- Verify no degradation on key metrics

### 3. Deployment
```bash
# Backup current production model
cp models/crash_severity_model.pkl models/crash_severity_model_backup_$(date +%Y%m%d).pkl

# Deploy new model
cp models/artifacts/xgboost_tuned_calibrated.pkl models/crash_severity_model.pkl

# Test in app
streamlit run app/app.py
```

### 4. Rollback (if needed)
```bash
# If new model has issues, rollback
cp models/crash_severity_model_backup_20250115.pkl models/crash_severity_model.pkl
```

---

## Segment Risk Model (Future)

Currently, only crash severity model is trained. For segment risk prediction:

### Option A: Keep using baseline
- Current baseline in Page 6 is reasonable
- Shows predicted crashes based on AADT, speed, lanes
- Good for demo purposes

### Option B: Train segment risk model
1. Create training script similar to crash model
2. Use `SEGMENT_LEVEL_ML` dataset
3. Train regression or classification model
4. Save to `models/segment_risk_model.pkl`

---

## Quick Reference

### File Locations
| Component | Location |
|-----------|----------|
| Training script | `ml_engineering/train_with_mlflow.py` |
| Trained models (artifacts) | `models/artifacts/*.pkl` |
| Production crash model | `models/crash_severity_model.pkl` |
| Production segment model | `models/segment_risk_model.pkl` |
| App model loader | `app/utils/model_loader.py` |
| App preprocessing | `app/utils/preprocessing.py` |
| MLflow experiments | `mlruns/` |

### Commands Cheat Sheet
```bash
# Train best model
python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost --tune

# View results
mlflow ui

# Deploy to app
cp models/artifacts/xgboost_tuned_calibrated.pkl models/crash_severity_model.pkl

# Test app
streamlit run app/app.py

# Check model loaded
python -c "import sys; sys.path.insert(0, 'app'); from utils.model_loader import load_crash_severity_model; print('✅' if load_crash_severity_model() else '❌')"
```

---

## Next Steps

1. **Train your first model** - Start with `--model baseline` for quick results
2. **Verify it works** - Copy to app location and test in Streamlit
3. **Iterate** - Try `--model xgboost --tune` for best performance
4. **Update app UI** - Show model type and performance metrics in Page 7
5. **Document** - Add model version and training date to app

**Questions?** Check the MLflow UI for detailed training logs and experiment tracking!
