"""
Configuration for Texas Crash Analysis Dashboard
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Medallion Architecture Paths
DATA_DIR = BASE_DIR / "data"

# Silver Layer (Cleaned)
SILVER_DIR = DATA_DIR / "silver"
TEXAS_SILVER = SILVER_DIR / "texas"
TEXAS_SILVER_ROADWAY = TEXAS_SILVER / "roadway"
HPMS_TEXAS_2023 = TEXAS_SILVER_ROADWAY / "hpms_texas_2023.gpkg"

# Gold Layer (ML-Ready)
GOLD_DIR = DATA_DIR / "gold"
ML_DATASETS_DIR = GOLD_DIR / "ml_datasets"
CRASH_LEVEL_ML_DIR = ML_DATASETS_DIR / "crash_level"
SEGMENT_LEVEL_ML_DIR = ML_DATASETS_DIR / "segment_level"

# Models
MODELS_DIR = BASE_DIR / "models"
CRASH_SEVERITY_MODEL = MODELS_DIR / "crash_severity_model.pkl"
SEGMENT_RISK_MODEL = MODELS_DIR / "segment_risk_model.pkl"

# Page configuration
PAGE_CONFIG = {
    "page_title": "Texas Crash Analysis Dashboard",
    "page_icon": "🚗",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS styling
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #2c3e50;
    }

    .info-box h3, .info-box h4, .info-box b, .info-box strong {
        color: #2c3e50;
    }

    .info-box ul li {
        color: #2c3e50;
    }

    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #2c3e50;
    }

    .warning-box h3, .warning-box h4, .warning-box b, .warning-box strong {
        color: #2c3e50;
    }

    .warning-box ul li {
        color: #2c3e50;
    }

    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #2c3e50;
    }

    .success-box h3, .success-box h4, .success-box b, .success-box strong {
        color: #2c3e50;
    }

    .success-box ul li {
        color: #2c3e50;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        color: #2c3e50;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
        color: #2c3e50;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    .download-btn {
        background-color: #27ae60;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
"""

# Texas center coordinates
TEXAS_CENTER = (31.0, -100.0)

# Default sampling sizes for large datasets
DEFAULT_SAMPLE_SIZES = {
    'crash_data': 50000,
    'austin_crashes': 50000,
    'weather_data': 10000,
    'crash_ml_train': 100000,
    'segment_ml_train': 50000
}

# Color schemes
SEVERITY_COLORS = {
    1: '#2ecc71',  # Green
    2: '#f39c12',  # Orange
    3: '#e74c3c',  # Red
    4: '#c0392b'   # Dark Red
}

RISK_COLORS = {
    'LOW': '#2ecc71',
    'MEDIUM': '#f39c12',
    'HIGH': '#e67e22',
    'VERY_HIGH': '#e74c3c'
}

# Feature categories for ML datasets
CRASH_ML_FEATURES = {
    'identifiers': ['ID', 'Source'],
    'temporal': ['year', 'month', 'quarter', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour'],
    'location': ['Start_Lat', 'Start_Lng', 'City', 'County', 'State', 'Zipcode'],
    'road': ['highway_type', 'num_lanes', 'speed_limit', 'is_bridge', 'is_tunnel'],
    'traffic': ['aadt', 'distance_to_aadt_m'],
    'weather': ['Temperature', 'Humidity', 'Pressure', 'Visibility', 'Wind_Speed', 'Precipitation'],
    'conditions': ['Weather_Condition', 'adverse_weather'],
    'infrastructure': ['Junction', 'Traffic_Signal', 'Stop', 'Crossing'],
    'target': ['Severity', 'high_severity']
}

SEGMENT_ML_FEATURES = {
    'identifiers': ['segment_id'],
    'temporal': ['year', 'quarter', 'year_quarter'],
    'location': ['City', 'Start_Lat', 'Start_Lng'],
    'road': ['highway_type', 'num_lanes', 'speed_limit'],
    'aggregates': ['crash_count', 'Severity', 'high_severity'],
    'traffic': ['aadt'],
    'weather': ['Temperature', 'Humidity', 'Pressure', 'Visibility'],
    'targets': ['crash_count', 'severity_rate', 'traffic_impact', 'crash_density', 'risk_score_simple', 'risk_category']
}
