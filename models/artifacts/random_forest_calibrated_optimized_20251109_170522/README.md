# random_forest_calibrated_optimized

**Created**: 20251109_170522

**Model**: CalibratedClassifierCV

**Features**: 21

## Metrics

- accuracy: 0.5608
- precision: 0.3305
- recall: 0.7095
- f1: 0.4510
- auc: 0.6134
- brier_score: 0.1979
- threshold: 0.3000

## Usage

```python
from ml_engineering.utils.persistence import load_model_artifact

pipeline, metadata = load_model_artifact('models/artifacts/random_forest_calibrated_optimized_20251109_170522')
predictions = pipeline.predict(X_new)
```
