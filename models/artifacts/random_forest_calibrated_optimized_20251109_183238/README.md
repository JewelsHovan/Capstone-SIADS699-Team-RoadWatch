# random_forest_calibrated_optimized

**Created**: 20251109_183238

**Model**: CalibratedClassifierCV

**Features**: 32

## Metrics

- accuracy: 0.6635
- precision: 0.4199
- recall: 0.8481
- f1: 0.5617
- auc: 0.7968
- brier_score: 0.1609
- threshold: 0.3000

## Usage

```python
from ml_engineering.utils.persistence import load_model_artifact

pipeline, metadata = load_model_artifact('models/artifacts/random_forest_calibrated_optimized_20251109_183238')
predictions = pipeline.predict(X_new)
```
