"""
Project Path Configuration

Centralized path definitions for data, models, and outputs
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
ML_READY = DATA_ROOT / "ml_ready"

# Texas data subdirectories
TEXAS_RAW = RAW_DATA / "texas"
TEXAS_CRASHES = TEXAS_RAW / "crashes"
TEXAS_WORKZONES = TEXAS_RAW / "workzones"
TEXAS_TRAFFIC = TEXAS_RAW / "traffic"
TEXAS_WEATHER = TEXAS_RAW / "weather"

# Processed datasets
CRASH_LEVEL_DATA = PROCESSED_DATA / "crash_level"
SEGMENT_LEVEL_DATA = PROCESSED_DATA / "segment_level"

# ML-ready datasets
CRASH_LEVEL_ML = ML_READY / "crash_level"
SEGMENT_LEVEL_ML = ML_READY / "segment_level"

# Outputs
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
MAPS = OUTPUTS_ROOT / "maps"
VISUALIZATIONS = OUTPUTS_ROOT / "visualizations"
MODELS = OUTPUTS_ROOT / "models"

# Default files
DEFAULT_CRASH_FILE = TEXAS_CRASHES / "kaggle_us_accidents_texas.csv"
DEFAULT_WORKZONE_FILE = TEXAS_WORKZONES / "texas_wzdx_feed.json"
DEFAULT_AADT_FILE = TEXAS_TRAFFIC / "txdot_aadt_annual.gpkg"


def ensure_directories():
    """Create all necessary directories if they don't exist"""
    dirs = [
        RAW_DATA, PROCESSED_DATA, ML_READY,
        TEXAS_CRASHES, TEXAS_WORKZONES, TEXAS_TRAFFIC, TEXAS_WEATHER,
        CRASH_LEVEL_DATA, SEGMENT_LEVEL_DATA,
        CRASH_LEVEL_ML, SEGMENT_LEVEL_ML,
        MAPS, VISUALIZATIONS, MODELS
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
