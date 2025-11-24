# Work Zone Safety & Traffic Impact Prediction

**Team Road Watch (Team 25)** - MADS Capstone Fall 2025

Real-time crash severity prediction system for emergency response prioritization using machine learning.

## Course Information

**SIADS 699: Capstone** - University of Michigan School of Information

**Instructors**: Elle O'Brien, Winston Featherly-Bean, Michelle LeBlanc, Laura Stagnaro, Kyle Balog, Paul Resnick, Shiv Saxena

**Course Manager**: Kirtana Choragudi | **Course Assistant**: Chris McAllister

## Overview

This project develops a machine learning system to predict crash severity in real-time using readily available information (location, time, weather, road characteristics) to support emergency response decision-making. The Random Forest model achieves 93% AUC-ROC and 81% recall on severe crashes, enabling data-driven resource allocation for emergency dispatch.

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

**Dataset**:
- 466,190 Texas crashes (2016-2022)
- 5 integrated data sources (crashes, AADT, HPMS, weather, work zones)
- 67 engineered features

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run app/Home.py
# Opens at http://localhost:8501
```

## Application Features

The Streamlit application provides 7 interactive pages:

1. **Crashes Explorer**: Exploratory data analysis on 466K crashes
2. **Work Zones**: TxDOT work zone feed visualization
3. **Crash ML Dataset**: Dataset viewer and statistics
4. **Segment ML Dataset**: Road segment-level data exploration
5. **Area Selector**: Geographic region selection tool
6. **Segment Risk Map**: Predicted crash risk heat map by road segment
7. **Crash Severity Predictor**: Real-time severity prediction (primary feature)

**Real-Time Predictor**:
- Input: Crash location (interactive map), time, weather, road characteristics
- Output: Severity probability, risk classification, response recommendation
- Latency: <200ms

## Team Contributions

### Julien Hovan - Crash Severity Prediction System

**Objective**: Real-time crash severity prediction for emergency response prioritization.

**Components**:
- Data pipeline (`data_engineering/`), ML pipeline (`ml_engineering/`), Streamlit app (`app/`)
- 466,190 Texas crashes (2016-2022), 67 engineered features
- Models: Logistic Regression (baseline), Random Forest (production, AUC=0.93), XGBoost
- MLflow experiment tracking (40+ runs)

**Deliverables**:
- Interactive Streamlit dashboard (7 pages)
- Trained models with metadata (`models/artifacts/`)
- Real-time prediction API (<200ms latency)

### Deepthi Kurup - Work Zone Location Risk Analysis

**Objective**: Identify work zones in historically high-crash road segments for safety resource prioritization.

**Components**:
- Processing scripts (`deepthi-src/`), notebooks (`notebooks/deepthi/`)
- 1,653 active work zones (TxDOT WZDx, April 2024)
- 19,178 historical crashes within 1km of work zones (2016-2023)
- Models: Stacking Ensemble (Random Forest + XGBoost + Logistic meta-learner, AUC=0.986)
- Data leakage detection and removal, GroupKFold validation

**Deliverables**:
- 12 sequential Jupyter notebooks
- Trained ensemble models (`deepthi-data/models/`)
- Interactive Texas maps (geographic hotspots: DFW, San Antonio, El Paso)
- SHAP feature importance analysis

**Note**: Temporal mismatch between work zone data (2024) and crash data (2016-2023) means this analysis identifies location-based risk rather than work zone-specific crash causation.

## Project Structure

```
Capstone/
├── app/                               # Streamlit application (Julien)
├── data/                              # Medallion architecture: bronze/silver/gold (Julien)
├── data_engineering/                  # Data processing pipeline (Julien)
├── ml_engineering/                    # ML training pipeline (Julien)
├── models/artifacts/                  # Trained models with metadata (Julien)
├── analysis/                          # EDA and report figures (Julien)
├── deepthi-src/                       # Work zone analysis code (Deepthi)
├── deepthi-data/                      # Work zone models and outputs (Deepthi)
├── notebooks/deepthi/                 # 12 analysis notebooks (Deepthi)
└── mlruns/                            # MLflow experiment tracking (Both)
```

See `PROJECT_STRUCTURE.md` for detailed structure.

## Data Sources

1. **Crash Data**: Kaggle US Accidents Dataset (466,190 Texas crashes, 2016-2022)
2. **Traffic Volume**: TxDOT AADT Stations (44,160 monitoring stations statewide)
3. **Road Infrastructure**: HPMS 2023 (speed limits, lanes, functional classification)
4. **Weather Data**: NOAA and crowdsourced APIs (temperature, visibility, precipitation, wind)
5. **Work Zones**: TxDOT WZDx Feed (1,653 active work zones, April 2024)

## Machine Learning Pipeline

**Data Processing (Medallion Architecture)**:
- Bronze: Raw data as downloaded
- Silver: Cleaned, validated, standardized
- Gold: ML-ready datasets with engineered features

**Temporal Split**:
- Training: 2016-2020 (370,971 crashes)
- Test: 2022 (95,219 crashes)

**Models Trained**:
1. Logistic Regression (Baseline): AUC=0.84, F1=0.11
2. Random Forest (Production): AUC=0.93, F1=0.67, isotonic calibration
3. XGBoost: AUC=0.76, F1=0.45

**Top Predictive Features**:
1. hpms_lanes (0.250): Multi-vehicle collision risk
2. hpms_aadt (0.239): High traffic volume
3. aadt (0.141): Traffic exposure
4. distance_to_aadt_m (0.100): Proximity to high-traffic corridors
5. is_major_city (0.058): Urban vs. rural dynamics

**Geographic Concentration**: 70% of crashes in 5 counties (Harris, Dallas, Bexar, Tarrant, Travis)

## Technical Stack

Python 3.10+, Streamlit, scikit-learn, XGBoost, MLflow, pandas, GeoPandas, Plotly, Folium

## Limitations

**Crash Severity Prediction (Julien)**:
- Geographic scope: Texas only
- Urban bias: 70% of crashes in 5 counties
- HPMS completeness: 12.8-58.6% spatial match rates

**Work Zone Location Risk Analysis (Deepthi)**:
- Temporal mismatch: Work zone data (2024) vs. crash data (2016-2023)
- Analysis identifies location-based risk rather than work zone-specific causation
- Dataset reduced to 219 unique work zones after deduplication

## Team

**Team Road Watch (Team 25)** - University of Michigan MADS Capstone Fall 2025

| Member | Role | Responsibilities |
|--------|------|------------------|
| Zahra Ahmed (zahraf@umich.edu) | Technical Writer | Project management, final report, poster/blog |
| Julien Hovan (jhovan@umich.edu) | Data Engineer Lead | Raw-to-ML-ready data pipelines, reproducibility |
| Deepthi Kurup (drkurup@umich.edu) | ML Engineer Lead | Work zone analysis, ML approaches, interpretation, figures |
