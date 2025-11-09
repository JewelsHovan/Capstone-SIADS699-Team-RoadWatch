# logistic_regression_calibrated

**Created**: 20251109_170511

**Model**: CalibratedClassifierCV

**Features**: 21

## Metrics

- accuracy: 0.7193
- precision: 0.3875
- recall: 0.1792
- f1: 0.2451
- auc: 0.6594
- brier_score: 0.1854
- threshold: 0.5000

## Usage

```python
from ml_engineering.utils.persistence import load_model_artifact

pipeline, metadata = load_model_artifact('models/artifacts/logistic_regression_calibrated_20251109_170511')
predictions = pipeline.predict(X_new)
```
