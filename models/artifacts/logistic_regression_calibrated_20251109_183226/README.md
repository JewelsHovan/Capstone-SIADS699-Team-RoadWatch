# logistic_regression_calibrated

**Created**: 20251109_183226

**Model**: CalibratedClassifierCV

**Features**: 32

## Metrics

- accuracy: 0.7517
- precision: 0.5090
- recall: 0.6590
- f1: 0.5744
- auc: 0.7998
- brier_score: 0.1567
- threshold: 0.5000

## Usage

```python
from ml_engineering.utils.persistence import load_model_artifact

pipeline, metadata = load_model_artifact('models/artifacts/logistic_regression_calibrated_20251109_183226')
predictions = pipeline.predict(X_new)
```
