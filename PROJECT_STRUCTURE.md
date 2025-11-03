# Project Structure

Organized for collaborative development between Data Engineering, ML Engineering, and Analysis teams.

## Directory Organization

```
Capstone/
├── data/                          # Data storage (gitignored)
│   ├── raw/                       # Raw downloaded data
│   ├── processed/                 # Cleaned, integrated data
│   └── ml_ready/                  # Balanced datasets for ML
│
├── data_engineering/              # Data Engineering Team
│   ├── download/                  # Data acquisition scripts
│   ├── clean/                     # Data cleaning
│   ├── integrate/                 # Data integration (crashes + workzones + AADT)
│   ├── features/                  # Feature engineering
│   ├── datasets/                  # Final dataset creation
│   └── utils/                     # Shared utilities
│
├── ml_engineering/                # ML Engineering Team
│   ├── preprocessing/             # Data preprocessing, SMOTE, feature selection
│   ├── models/                    # Model implementations
│   ├── evaluation/                # Model evaluation, metrics
│   └── utils/                     # ML utilities
│
├── analysis/                      # Analysis Team
│   ├── exploratory/               # Jupyter notebooks, EDA
│   ├── reports/                   # Data quality reports
│   └── visualization/             # Plotting and mapping utilities
│
├── config/                        # Shared configuration
│   └── paths.py                   # Centralized path definitions
│
├── app/                           # Streamlit dashboard
├── scripts/                       # Legacy scripts (to be phased out)
└── outputs/                       # Models, visualizations, maps
```

## Usage

### Data Engineering
```python
from data_engineering.integrate import integrate_workzones
from data_engineering.datasets import build_ml_training_dataset
from config.paths import TEXAS_CRASHES, CRASH_LEVEL_DATA
```

### ML Engineering
```python
from ml_engineering.preprocessing import balance_classes
from ml_engineering.models import train_xgboost
from config.paths import CRASH_LEVEL_ML
```

### Analysis
```python
from analysis.visualization import create_crash_map
from config.paths import VISUALIZATIONS
```

## Migration Status

**Migrated:**
- ✓ Data download scripts → `data_engineering/download/`
- ✓ Dataset building → `data_engineering/datasets/`
- ✓ Work zone integration → `data_engineering/integrate/`
- ✓ Shared paths → `config/paths.py`

**To Do:**
- Move analysis notebooks to `analysis/exploratory/`
- Implement SMOTE pipeline in `ml_engineering/preprocessing/`
- Create baseline models in `ml_engineering/models/`
- Phase out `scripts/` directory once all scripts migrated
