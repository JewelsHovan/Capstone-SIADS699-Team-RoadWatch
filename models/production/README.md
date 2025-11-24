# Production Models

This directory contains the best-performing crash severity prediction models for deployment.

## Models

### 1. Random Forest (Best Recall - Emergency Response)
- **Path:** `random_forest_best_recall/`
- **AUC:** 0.9309
- **Recall:** 81.1% (catches 8,194 / 10,128 severe crashes)
- **Precision:** 57.3%
- **F1:** 0.6714
- **Use Case:** Emergency response prioritization (maximize lives saved)
- **Decision Threshold:** 0.45 (optimized)

### 2. CatBoost (Best Balanced - Resource Constrained)
- **Path:** `catboost_best_balanced/`
- **AUC:** 0.9364
- **Recall:** 59.9% (catches 6,068 / 10,128 severe crashes)
- **Precision:** 80.7%
- **F1:** 0.6878 (BEST)
- **Use Case:** Balanced precision/recall, minimal false alarms
- **Training:** Fastest (26 iterations)

### 3. LightGBM (Best AUC - High Precision)
- **Path:** `lightgbm_best_auc/`
- **AUC:** 0.9364 (TIED BEST)
- **Recall:** 22.5%
- **Precision:** 78.5%
- **Use Case:** High-confidence validation, second opinion

## Usage

### Load a Model

```python
from ml_engineering.utils.persistence import load_model_artifact

# Load Random Forest (best recall)
pipeline, metadata = load_model_artifact('models/production/random_forest_best_recall')

# Predict with optimized threshold
threshold = metadata['metrics']['threshold']  # 0.45
proba = pipeline.predict_proba(X_new)[:, 1]
predictions = (proba >= threshold).astype(int)
```

## Recommendation

**For emergency response:** Use **Random Forest** (catches 81% of severe crashes)
**For balanced operation:** Use **CatBoost** (best F1, minimal false alarms)

---
**Generated:** November 2025
**Dataset:** Crash-Level (2016-2022)
**MLflow Experiment:** crash_severity_prediction
