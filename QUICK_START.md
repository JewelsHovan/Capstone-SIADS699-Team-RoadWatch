# Quick Start Guide

## For Data Engineers

### Build crash-level dataset:
```bash
python data_engineering/datasets/build_ml_training_dataset.py \
    --sample 10000 \
    --output-dir data/processed/crash_level
```

### Build segment-level dataset:
```bash
python data_engineering/datasets/build_segment_dataset.py \
    --input data/processed/crash_level/train_latest.csv \
    --output-dir data/processed/segment_level
```

### Download new data:
```bash
python data_engineering/download/download_texas_feed.py
python data_engineering/download/download_austin_crashes.py
```

---

## For ML Engineers

### Use centralized paths:
```python
from config.paths import CRASH_LEVEL_DATA, CRASH_LEVEL_ML

# Load training data
import pandas as pd
train = pd.read_csv(CRASH_LEVEL_DATA / "train_latest.csv")

# Your preprocessing here
# balanced_train = apply_smote(train)

# Save ML-ready data
balanced_train.to_csv(CRASH_LEVEL_ML / "train_balanced.csv")
```

### Create new models:
Create files in `ml_engineering/models/`:
```python
# ml_engineering/models/xgboost_crash_model.py
from config.paths import CRASH_LEVEL_ML, MODELS
import xgboost as xgb

def train_crash_severity_model():
    # Your model code here
    pass
```

---

## For Analysts

### Use shared utilities:
```python
from config.paths import CRASH_LEVEL_DATA, VISUALIZATIONS
from analysis.visualization import create_crash_map

# Load data
import pandas as pd
crashes = pd.read_csv(CRASH_LEVEL_DATA / "train_latest.csv")

# Create visualization
create_crash_map(crashes, save_path=VISUALIZATIONS / "crash_density.html")
```

### Create notebooks:
Place Jupyter notebooks in `analysis/exploratory/`:
- `analysis/exploratory/01_crash_eda.ipynb`
- `analysis/exploratory/02_workzone_analysis.ipynb`
- `analysis/exploratory/03_feature_importance.ipynb`

---

## Testing

Run scripts from project root:
```bash
# Test data engineering pipeline
python data_engineering/datasets/build_ml_training_dataset.py --sample 1000

# Test imports
python -c "from config.paths import PROJECT_ROOT; print(PROJECT_ROOT)"
python -c "from data_engineering.integrate import integrate_workzones"
```
