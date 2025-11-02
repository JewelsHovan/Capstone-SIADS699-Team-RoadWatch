# Texas Crash Analysis Dashboard

**SIADS 699 Capstone Project**

Comprehensive Streamlit dashboard for analyzing Texas traffic crashes, work zones, and machine learning datasets for crash risk prediction.

## 📊 Dashboard Pages

### 1. 🏠 Home (app.py)
- **Overview** of all datasets and project goals
- **Summary metrics** across all data sources
- **Dataset composition** visualization
- **Navigation guide** to other pages

### 2. 🚗 Crashes Explorer
- **Raw crash data** analysis (Kaggle US Accidents + Austin)
- **Temporal analysis**: trends over time, hourly patterns
- **Geographic analysis**: heatmaps and point maps
- **Severity analysis**: distribution and patterns
- **Feature distributions**: histograms and box plots
- **Interactive filters**: date range, severity levels
- **Sample size**: Up to 100,000 crashes

### 3. 🚧 Work Zones Analysis
- **Active work zones** across Texas (2,180 zones)
- **Geographic visualization**: interactive map
- **Analytics**: county and road distributions
- **Duration analysis**: work zone timeframes
- **Data table**: searchable and downloadable

### 4. 🤖 Crash-Level ML Dataset
- **1.1M crashes** with 78 engineered features
- **Target**: high_severity (binary classification)
- **Train/Val/Test splits**: 2016-2021 / 2022 / 2023
- **Feature categories**: temporal, location, road, traffic, weather
- **Correlation analysis**: heatmaps and top predictors
- **Data quality**: missing data assessment
- **Use case**: Individual crash severity prediction

### 5. 🗺️ Segment-Level ML Dataset
- **303K segment-quarters** from 75,650 road segments
- **39 aggregated features**
- **Multiple targets**: crash_count, severity_rate, traffic_impact, risk_score, risk_category
- **Risk categories**: LOW/MEDIUM/HIGH/VERY_HIGH
- **Geographic view**: segments colored by risk level
- **Temporal patterns**: quarterly trends and seasonality
- **Use case**: Work zone risk prediction

## 🚀 Running the Dashboard

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.17.0
- folium >= 0.14.0
- streamlit-folium >= 0.15.0
- numpy >= 1.24.0

### Launch

From the `app/` directory:

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Data Requirements

The dashboard expects data in the following structure:

```
data/
├── raw/texas/
│   ├── crashes/
│   │   ├── kaggle_us_accidents_texas.csv
│   │   └── austin_crashes_latest.csv
│   ├── workzones/
│   │   └── texas_wzdx_feed.csv
│   ├── weather/
│   │   └── texas_weather_latest.csv
│   └── traffic/
│       └── txdot_aadt_annual.gpkg
└── processed/
    ├── crash_level/
    │   ├── train_latest.csv
    │   ├── val_latest.csv
    │   └── test_latest.csv
    └── segment_level/
        ├── segment_train_latest.csv
        ├── segment_val_latest.csv
        └── segment_test_latest.csv
```

## 📁 Project Structure

```
app/
├── app.py                              # Home page
├── config.py                           # Configuration and styling
├── pages/
│   ├── 1_🚗_Crashes_Explorer.py        # Raw crash data analysis
│   ├── 2_🚧_Work_Zones.py              # Work zones analysis
│   ├── 3_🤖_Crash_ML_Dataset.py        # Crash-level ML dataset
│   └── 4_🗺️_Segment_ML_Dataset.py     # Segment-level ML dataset
└── utils/
    ├── __init__.py
    ├── data_loader.py                  # Data loading with caching
    ├── visualizations.py               # Plotly chart functions
    └── map_utils.py                    # Folium map functions
```

## 🎨 Features

### Performance Optimizations
- **Caching**: Streamlit `@st.cache_data` for efficient data loading
- **Sampling**: Configurable sample sizes for large datasets
- **Lazy loading**: Data loaded only when needed

### Interactive Elements
- **Filters**: Date ranges, severity levels, geographic areas
- **Sample size controls**: Adjust performance vs completeness
- **Multiple visualizations**: Charts, maps, tables
- **Download options**: Export filtered data as CSV

### Visualizations
- **Plotly charts**: Interactive plots with zoom, pan, hover
- **Folium maps**: Heatmaps, point maps, cluster maps
- **Box plots**: Feature distributions by category
- **Correlation heatmaps**: Feature relationships
- **Time series**: Trends with moving averages

## 📊 Dataset Details

### Raw Data
- **Kaggle US Accidents**: 582,837 Texas crashes (213 MB)
- **Austin Crashes**: 223,713 crashes (81 MB)
- **Work Zones**: 2,180 active zones
- **Weather**: NOAA daily data (2016-2023)
- **Traffic**: TxDOT AADT (41,467 stations)

### ML Datasets
- **Crash-Level**: 1,135,762 crashes × 78 features
  - Target: high_severity (binary)
  - Splits: Train (2016-2021), Val (2022), Test (2023)

- **Segment-Level**: 303,281 segment-quarters × 39 features
  - 75,650 unique road segments
  - Targets: crash_count, severity_rate, traffic_impact, risk_score, risk_category
  - Quarterly aggregation

## 🔗 Resources

- **GitHub**: [Capstone-SIADS699](https://github.com/JewelsHovan/Capstone-SIADS699)
- **Google Drive**: [Project Data](https://drive.google.com/drive/folders/1xVGXbxUFHSdSawo2C9wnmABj15wPEX3A)
- **Data Sources**:
  - Kaggle US Accidents Dataset
  - City of Austin Open Data Portal
  - TxDOT Work Zone Data Exchange (WZDx)
  - TxDOT AADT Traffic Counts
  - NOAA Climate Data

## 📝 Notes

- **Sample sizes** can be adjusted in sidebar for performance
- **Maps** are limited to 1,000-10,000 points for responsiveness
- **Caching** persists for 1 hour (3600 seconds)
- **File paths** use symlinks pointing to latest versions

## 🎯 Future Enhancements

- Risk prediction tool with polygon drawing
- Model performance metrics and comparisons
- Real-time work zone feed integration
- Predictive analytics dashboard
- Export to various formats (Excel, JSON)
- Advanced filtering and querying
- Custom risk scoring calculator

---

**University of Michigan School of Information** | **SIADS 699 Capstone** | **2025**
