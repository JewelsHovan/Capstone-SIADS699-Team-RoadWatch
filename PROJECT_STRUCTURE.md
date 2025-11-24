# Project Structure

Technical organization for crash severity prediction system development.

## Core Architecture

### Data Pipeline (Medallion Architecture)
```
data/
├── bronze/          # Raw immutable data (Texas, California, New York crashes)
├── silver/          # Cleaned and validated data
└── gold/            # ML-ready datasets with engineered features
    ├── ml_datasets/ # Training/test splits (crash_level, segment_level)
    └── analytics/   # Analytics-ready aggregations
```

### Processing Modules

**data_engineering/** - ETL pipeline
- `download/` - Data acquisition scripts
- `clean/` - Data cleaning and validation
- `integrate/` - Spatial integration (HPMS, AADT joins)
- `features/` - Feature engineering
- `datasets/` - ML dataset builders
- `utils/` - Shared utilities and validation

**ml_engineering/** - Model training pipeline
- `preprocessing/` - Data preprocessing and feature selection
- `models/` - Model training implementations
- `evaluation/` - Model evaluation and metrics
- `utils/` - ML utilities

**analysis/** - Reporting and visualization
- `exploratory/` - Exploratory data analysis scripts
- `reports/` - Publication-ready figures
- `visualization/` - Plotting and mapping utilities

### Application

**app/** - Streamlit dashboard
- `Home.py` - Application entry point
- `pages/` - Interactive pages (7 modules)
- `utils/` - Visualization helpers
- `config.py` - App configuration

**models/** - Trained model artifacts
- `artifacts/` - Model files with metadata (Random Forest, XGBoost, etc.)
- `production/` - Production model symlinks

### Configuration

**config/** - Centralized configuration
- `paths.py` - Path definitions for data pipeline

## Key Files

- `README.md` - Project overview and team contributions
- `requirements.txt` - Python dependencies
- `CLAUDE.md` - Project-specific development guidelines

