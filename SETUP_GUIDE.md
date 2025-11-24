# Texas Crash Prediction - Setup Guide

**Author:** Julien Hovan
**Project:** MADS Capstone - Crash Severity Prediction System

## Quick Start

For instructors with provided data files:

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# 2. Setup directories
PYTHONPATH=. python scripts/setup_data_directories.py

# 3. Place provided data files
# Copy hpms_texas_2023.gpkg to data/silver/texas/roadway/
# Copy kaggle_us_accidents_texas.csv to data/bronze/texas/crashes/

# 4. Verify data
PYTHONPATH=. python scripts/verify_data.py

# 5. Run pipeline (sample mode)
PYTHONPATH=. python scripts/run_pipeline.py --sample 10000

# 6. Train models
PYTHONPATH=. python -m ml_engineering.train_with_mlflow --dataset crash --model all

# 7. Launch application
streamlit run app/Home.py
```

**Time:** 15 minutes (sample mode) or 2 hours (full dataset)

## Infrastructure Created

### 1. Package Installation (setup.py)

Makes project pip-installable:

```bash
pip install -e .
```

Benefits:
- No sys.path.append() hacks required
- Import from anywhere: `from config.paths import BRONZE`
- Automatic dependency installation
- Console scripts: crash-pipeline, crash-train

### 2. Data Directory Setup (scripts/setup_data_directories.py)

Creates Medallion Architecture structure:

```
data/
├── bronze/texas/       # Raw data
├── silver/texas/       # Cleaned data
└── gold/ml_datasets/   # ML-ready datasets
```

Usage:
```bash
PYTHONPATH=. python scripts/setup_data_directories.py
```

Output:
- Creates all necessary directories
- Generates README files in Bronze/Silver/Gold layers
- Shows directory tree structure

### 3. Data Verification (scripts/verify_data.py)

Validates data before running pipeline:

```bash
PYTHONPATH=. python scripts/verify_data.py
```

Checks:
- File existence and size
- Required columns present
- Data integrity (row counts, geometry types)
- CRS information

### 4. Master Pipeline (scripts/run_pipeline.py)

Orchestrates complete data engineering pipeline:

```bash
# Full pipeline
PYTHONPATH=. python scripts/run_pipeline.py

# Sample mode (testing)
PYTHONPATH=. python scripts/run_pipeline.py --sample 10000

# Crash-level only
PYTHONPATH=. python scripts/run_pipeline.py --crash-only
```

Pipeline stages:
1. Data verification
2. Build crash-level dataset (370K samples, 67 features)
3. Build segment-level dataset (971K segments, 11 features)

Features:
- Error handling with user prompts
- Progress tracking
- Duration reporting

## Data Sources

### Required for ML Pipeline

| Data Source | Location | Size | Description |
|-------------|----------|------|-------------|
| Kaggle US Accidents | bronze/texas/crashes/ | 2.8GB | 466K Texas crashes (2016-2023) |
| HPMS Texas 2023 | silver/texas/roadway/ | 1.1GB | 971K road segments with AADT |

HPMS file includes AADT traffic data. Separate AADT files in bronze/traffic/ are optional.

### File to Provide Instructors

**Primary file:** `data/silver/texas/roadway/hpms_texas_2023.gpkg` (1.1GB)

Contents:
- Pre-filtered from 14GB national HPMS dataset
- All road characteristics needed for ML pipeline
- AADT traffic counts integrated
- Ready to use without manual HPMS download

## Project Structure

```
Capstone/
├── config/                  # Centralized configuration
│   └── paths.py            # Medallion architecture paths
│
├── data/                    # Medallion data architecture
│   ├── bronze/             # Raw data
│   ├── silver/             # Cleaned data
│   └── gold/               # ML-ready datasets
│
├── data_engineering/        # ETL pipeline
│   ├── download/           # Data acquisition scripts
│   ├── integrate/          # Spatial joins
│   ├── datasets/           # Dataset builders
│   └── utils/              # Validation, helpers
│
├── ml_engineering/          # ML pipeline
│   ├── preprocessing/      # Feature engineering
│   ├── models/             # Model implementations
│   ├── evaluation/         # Metrics, calibration
│   └── utils/              # MLflow, persistence
│
├── app/                     # Streamlit dashboard (7 pages)
│   ├── Home.py
│   ├── pages/
│   └── utils/
│
├── scripts/                 # Orchestration scripts
│   ├── setup_data_directories.py
│   ├── verify_data.py
│   └── run_pipeline.py
│
└── setup.py                 # Package installation
```

## Complete Workflow

### Phase 1: Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install project
pip install -e .

# Setup data directories
PYTHONPATH=. python scripts/setup_data_directories.py
```

### Phase 2: Data Acquisition

**Option A: Use Provided Files**

```bash
# Place files in correct locations
cp hpms_texas_2023.gpkg data/silver/texas/roadway/
cp kaggle_us_accidents_texas.csv data/bronze/texas/crashes/
```

**Option B: Download from Scratch**

See DATA_ACQUISITION.md for instructions.

### Phase 3: Data Verification

```bash
PYTHONPATH=. python scripts/verify_data.py
```

Expected output:
```
Kaggle crashes found: 466,190 rows
HPMS Texas found: 971,244 road segments
Ready to run data pipeline
```

### Phase 4: Build ML Datasets

```bash
# Sample mode (10K crashes, 2 minutes)
PYTHONPATH=. python scripts/run_pipeline.py --sample 10000

# Full dataset (466K crashes, 30 minutes)
PYTHONPATH=. python scripts/run_pipeline.py
```

Output:
- data/gold/ml_datasets/crash_level/train_latest.csv (370K rows, 67 features)
- data/gold/ml_datasets/crash_level/val_latest.csv (79K rows)
- data/gold/ml_datasets/crash_level/test_latest.csv (95K rows)

### Phase 5: Train Models

```bash
# Train all models (LR, RF, XGBoost, CatBoost, LightGBM)
PYTHONPATH=. python -m ml_engineering.train_with_mlflow --dataset crash --model all

# Train specific model
PYTHONPATH=. python -m ml_engineering.train_with_mlflow --dataset crash --model random_forest --tune
```

Output:
- Trained models in models/artifacts/
- MLflow tracking in mlruns/
- Performance metrics logged

### Phase 6: Launch Dashboard

```bash
streamlit run app/Home.py
```

Features:
- 7 interactive pages
- Crash explorer (466K crashes)
- Real-time crash severity prediction
- Segment risk heatmaps
- Model performance dashboards

## Performance Metrics

### Pipeline Execution Time

| Mode | Sample Size | Time | Output Size |
|------|-------------|------|-------------|
| Sample | 10,000 | 2 min | 20MB |
| Small | 50,000 | 5 min | 100MB |
| Full | 466,000 | 30 min | 800MB |

### Model Training Time

| Model | Training Time | AUC | Status |
|-------|---------------|-----|--------|
| Logistic Regression | 30 sec | 0.61 | Baseline |
| Random Forest | 2 min | 0.93 | Production |
| XGBoost | 5 min | 0.92 | Alternate |
| CatBoost | 8 min | 0.91 | Experimental |
| LightGBM | 3 min | 0.92 | Experimental |

## Validation Checklist

Before submitting to instructors:

- [ ] setup.py installs successfully: `pip install -e .`
- [ ] Data directories created: `PYTHONPATH=. python scripts/setup_data_directories.py`
- [ ] Data files placed correctly (crashes + HPMS)
- [ ] Data verification passes: `PYTHONPATH=. python scripts/verify_data.py`
- [ ] Pipeline runs on sample: `PYTHONPATH=. python scripts/run_pipeline.py --sample 10000`
- [ ] Models train successfully: `PYTHONPATH=. python -m ml_engineering.train_with_mlflow --dataset crash --model baseline`
- [ ] Streamlit app launches: `streamlit run app/Home.py`
- [ ] README updated with correct instructions
- [ ] hpms_texas_2023.gpkg included in submission

## Directory Structure After Setup

```
data/
├── bronze/              # Raw data (immutable)
│   └── texas/
│       └── crashes/
│           └── kaggle_us_accidents_texas.csv  (2.8GB)
│
├── silver/              # Cleaned data
│   └── texas/
│       └── roadway/
│           └── hpms_texas_2023.gpkg  (1.1GB)
│
└── gold/                # ML-ready (generated by pipeline)
    └── ml_datasets/
        └── crash_level/
            ├── train_latest.csv
            ├── val_latest.csv
            └── test_latest.csv
```

## Repository Improvements

### What Was Created

1. Package installation system (setup.py)
2. Data directory setup script
3. Data verification script
4. Master pipeline orchestration script
5. Data acquisition guide
6. Complete setup guide

### Repository Cleanup

Removed:
- California data (1.5MB)
- New York data (2.1GB)
- Updated config/paths.py (Texas-only)

Space saved: 2.1GB

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Time to Reproduce | 4-6 hours | 15-30 min |
| Reproducibility Score | 3/10 | 8/10 |
| Manual Steps | 11+ scripts | 1 command |
