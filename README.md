# Work Zone Safety & Traffic Impact Prediction

**Team IntelliZone** - MADS Capstone Fall 2025

Real-time crash severity prediction system for emergency response prioritization using machine learning.

---

## Overview

This project develops a machine learning system to predict crash severity in real-time using readily available information (location, time, weather, road characteristics) to support emergency response decision-making. Our Random Forest model achieves 93% AUC-ROC and 81% recall on severe crashes, enabling data-driven resource allocation for emergency dispatch.

---

## Project Goals

1. Integrate multi-source transportation data (crashes, traffic, road infrastructure, weather)
2. Engineer predictive features from spatial and temporal data
3. Train and evaluate machine learning models for crash severity prediction
4. Deploy an interactive decision-support application for emergency response

---

## Key Results

**Model Performance** (Test Set, 2022 data):
- **AUC-ROC**: 0.93
- **F1 Score**: 0.67
- **Recall**: 0.81 (catches 81% of severe crashes)
- **Precision**: 0.57
- **Accuracy**: 0.92

**Top Risk Factors**:
1. Traffic volume (AADT): 53% of model importance
2. Road characteristics (lanes, speed limit): 30% of importance
3. Temporal factors (hour, rush hour): 10% of importance
4. Weather conditions: 7% of importance

**Dataset Scale**:
- 466,190 Texas crashes (2016-2022)
- 236 counties, 849 cities, 15 regions
- 5 integrated data sources
- 67 engineered features

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone repository
cd /path/to/Capstone

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, sklearn; print('Installation successful')"
```

---

## Usage

### Run Streamlit Application

```bash
# Start the interactive dashboard
streamlit run app/Home.py

# Application opens at http://localhost:8501
```

### Application Features

The Streamlit application provides 7 interactive pages:

1. **Crashes Explorer**: Exploratory data analysis on 466K crashes
2. **Work Zones**: TxDOT work zone feed visualization
3. **Crash ML Dataset**: Dataset viewer and statistics
4. **Segment ML Dataset**: Road segment-level data exploration
5. **Area Selector**: Geographic region selection tool
6. **Segment Risk Map**: Predicted crash risk heat map by road segment
7. **Crash Severity Predictor**: Real-time severity prediction (primary feature)

### Key Functionality: Real-Time Crash Severity Predictor

**Input**:
- Crash location (interactive map selection)
- Current time and weather conditions
- Road characteristics (auto-populated from HPMS)

**Output**:
- Severity probability (0-100%)
- Risk classification (Low/Medium/High)
- Emergency response recommendation
- Contributing risk factors

**Prediction Latency**: <200ms

---

## Project Structure

```
Capstone/
├── README.md                           # This file
├── FINAL_REPORT_OUTLINE.md            # Comprehensive project report
├── requirements.txt                   # Python dependencies
│
├── app/                               # Streamlit application
│   ├── Home.py                        # Application entry point
│   ├── config.py                      # Application configuration
│   ├── pages/                         # Interactive pages (7 files)
│   └── utils/                         # Visualization utilities
│
├── data/                              # Data pipeline (medallion architecture)
│   ├── bronze/                        # Raw data (crashes, AADT, HPMS)
│   ├── silver/                        # Cleaned and validated data
│   └── gold/                          # ML-ready datasets
│       └── ml_datasets/
│           └── crash_level/           # 466K crashes, train/test splits
│
├── data_engineering/                  # Data processing pipeline
│   ├── download/                      # Data collection scripts
│   ├── clean/                         # Data cleaning
│   ├── features/                      # Feature engineering
│   ├── datasets/                      # ML dataset builders
│   └── integrate/                     # Spatial integration (HPMS, AADT)
│
├── ml_engineering/                    # Machine learning pipeline
│   ├── models/                        # Model training scripts
│   ├── preprocessing/                 # Data preprocessing
│   ├── evaluation/                    # Model evaluation metrics
│   └── utils/                         # ML utilities
│
├── models/                            # Trained models
│   └── artifacts/                     # Model artifacts with metadata
│       ├── logistic_regression_*/     # Baseline model
│       ├── random_forest_*/           # Best model (AUC=0.93)
│       └── xgboost_*/                 # Gradient boosting model
│
├── analysis/                          # Analysis and reporting
│   ├── exploratory/                   # EDA scripts and figures
│   └── reports/                       # Report figures (publication-ready)
│       └── figures/                   # 5 final figures
│
├── mlruns/                            # MLflow experiment tracking
├── outputs/                           # Generated outputs
│   ├── maps/                          # Interactive HTML maps
│   └── visualizations/                # Analysis plots
│
└── config/                            # Project configuration
    └── paths.py                       # Centralized path management
```

---

## Data Sources

### 1. Crash Data
- **Source**: Kaggle US Accidents Dataset
- **Coverage**: 466,190 Texas crashes (2016-2022)
- **Features**: Location, timestamp, severity, weather, road conditions

### 2. Traffic Volume
- **Source**: Texas Department of Transportation (TxDOT) AADT Stations
- **Coverage**: 44,160 traffic monitoring stations statewide
- **Integration**: Nearest-neighbor spatial join (100% crash coverage)

### 3. Road Infrastructure
- **Source**: Highway Performance Monitoring System (HPMS) 2023
- **Features**: Speed limits, lane counts, functional classification
- **Integration**: Spatial join with 100m buffer (12.8-58.6% match rates)

### 4. Weather Data
- **Source**: NOAA and crowdsourced APIs
- **Features**: Temperature, visibility, precipitation, wind speed, pressure
- **Temporal Resolution**: Nearest observation to crash timestamp

### 5. Work Zones (Context)
- **Source**: TxDOT Real-Time Work Zone Feed (WZDx format)
- **Status**: Used for context and visualization (not directly integrated due to temporal mismatch)

---

## Machine Learning Pipeline

### Data Processing

**Medallion Architecture**:
1. **Bronze Layer**: Raw, immutable data as downloaded
2. **Silver Layer**: Cleaned, validated, standardized
3. **Gold Layer**: ML-ready datasets with engineered features

**Temporal Split** (prevents data leakage):
- **Training**: 2016-2020 (370,971 crashes)
- **Validation**: 2021 (not used in final evaluation)
- **Test**: 2022 (95,219 crashes)

### Models Trained

1. **Logistic Regression** (Baseline)
   - Linear model with L2 regularization
   - AUC: 0.84, F1: 0.11

2. **Random Forest** (Best Model)
   - 100 decision trees with class balancing
   - Threshold optimized from 0.5 to 0.45
   - Isotonic calibration for probability reliability
   - AUC: 0.93, F1: 0.67

3. **XGBoost**
   - Gradient boosting with early stopping
   - AUC: 0.76, F1: 0.45

**Model Selection**: Random Forest selected based on superior AUC-ROC, F1 score, and recall performance.

### Experiment Tracking

All experiments logged with MLflow:
- Hyperparameters
- Performance metrics
- Feature importance
- Model artifacts
- Training history

---

## Key Findings

### Top Predictive Features

1. **hpms_lanes** (0.250): Number of lanes → multi-vehicle collision risk
2. **hpms_aadt** (0.239): High traffic volume → increased severity
3. **aadt** (0.141): Traffic exposure metric
4. **distance_to_aadt_m** (0.100): Proximity to high-traffic corridors
5. **is_major_city** (0.058): Urban vs. rural crash dynamics

### Geographic Concentration

**Top 5 Counties** (70% of all crashes):
1. Harris (Houston): 118,365 crashes (25.4%)
2. Dallas: 94,006 crashes (20.2%)
3. Bexar (San Antonio): 58,820 crashes (12.6%)
4. Tarrant (Fort Worth): 34,625 crashes (7.4%)
5. Travis (Austin): 14,746 crashes (3.2%)

### Impact Potential

For every 1,000 crashes:
- Model flags 421 as high-risk
- Catches 194 of 240 actual severe crashes (81% recall)
- Enables faster emergency response to life-threatening incidents

---

## Technical Stack

**Languages & Frameworks**:
- Python 3.10+
- Streamlit (web application)
- scikit-learn (machine learning)
- XGBoost (gradient boosting)
- MLflow (experiment tracking)

**Data Processing**:
- pandas (data manipulation)
- GeoPandas (spatial operations)
- NumPy (numerical computing)

**Visualization**:
- Matplotlib & Seaborn (static plots)
- Plotly (interactive charts)
- Folium (interactive maps)

**Development**:
- Git (version control)
- pytest (testing)
- Black (code formatting)

---

## Data Quality & Validation

### Challenge: Data Leakage Detection

**Initial Model Issue**: AUC = 0.9999 (unrealistically high)

**Root Cause**:
- 1,434 duplicate crash records
- Post-crash features (duration) causing target leakage
- Low-variance features inflating importance

**Resolution**:
- Deduplication of crash records
- Removal of post-crash features
- Strict temporal validation split
- Feature variance analysis

**Result**: AUC corrected to realistic 0.93

### Class Imbalance Handling

**Approach**:
- `class_weight='balanced'` in Random Forest
- Threshold optimization (0.5 → 0.45)
- Focus on recall over precision (emergency response priority)

**Impact**: Recall improved from 6% to 81%

---

## Limitations

1. **Geographic Scope**: Texas only (features are transferable to other states)
2. **Work Zone Temporal Mismatch**: Real-time feed vs. historical crash data
3. **Urban Bias**: 70% of crashes in 5 urban counties
4. **HPMS Completeness**: 12.8-58.6% spatial match rates
5. **Severity Definition Change**: Texas reporting system updated in 2021

---

## Future Work

1. **Multi-State Expansion**: Validate model on California and New York data
2. **Historical Work Zone Integration**: If TxDOT archives become available
3. **Traffic Slowdown Prediction**: Integrate real-time traffic sensor data
4. **Operational Deployment**: API integration with 911 dispatch systems
5. **Explainability**: SHAP values for instance-level predictions
6. **Continuous Learning**: Automated monthly retraining pipeline

---

## Documentation

**Project Reports**:
- `FINAL_REPORT_OUTLINE.md` - Comprehensive technical report
- `REPORT_OUTLINE_ASSIGNMENT.md` - Executive summary (3-page format)

**Analysis**:
- `analysis/reports/figures/` - Publication-ready figures
- `analysis/exploratory/figures/` - Exploratory data analysis

**Data Dictionaries**:
- `data/gold/ml_datasets/crash_level/DATA_DICTIONARY.md` - Feature definitions

**Model Documentation**:
- `models/artifacts/*/README.md` - Individual model usage guides
- `models/artifacts/*/metadata.json` - Performance metrics

---

## Team

**Team IntelliZone** - University of Michigan MADS Capstone Fall 2025

- Julien Hovan (jhovan@umich.edu)
- Zahra Ahmed (zahraf@umich.edu)
- Deepthi Kurup (drkurup@umich.edu)

---

## References

1. National Highway Traffic Safety Administration (NHTSA). Fatality Analysis Reporting System (FARS), 2013-2023.
2. Kaggle US Accidents Dataset. https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
3. Texas Department of Transportation (TxDOT). AADT Data and HPMS 2023.
4. Work Zone Data Exchange (WZDx) Specification. https://github.com/usdot-jpo-ode/wzdx
5. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

---

## License

This project is developed for academic purposes as part of the University of Michigan Master of Applied Data Science program.

---

## Acknowledgments

**Data Sources**:
- Kaggle US Accidents Dataset
- Texas Department of Transportation (TxDOT)
- National Oceanic and Atmospheric Administration (NOAA)
- Federal Highway Administration (FHWA)

**Tools & Frameworks**:
- Streamlit, scikit-learn, XGBoost, MLflow
- pandas, GeoPandas, Folium, Plotly

---

**University of Michigan - School of Information - MADS Capstone - Fall 2025**
