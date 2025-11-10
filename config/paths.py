"""
Project Path Configuration

Centralized path definitions for data, models, and outputs
Using Medallion Architecture: Bronze (raw) → Silver (cleaned) → Gold (ML-ready)
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# ==============================================================================
# MEDALLION ARCHITECTURE (Bronze / Silver / Gold)
# ==============================================================================

DATA_ROOT = PROJECT_ROOT / "data"

# Bronze Layer: Raw, immutable data (as downloaded)
BRONZE = DATA_ROOT / "bronze"
BRONZE_TEXAS = BRONZE / "texas"
BRONZE_CALIFORNIA = BRONZE / "california"
BRONZE_NEW_YORK = BRONZE / "new_york"

# Silver Layer: Cleaned, validated, standardized
SILVER = DATA_ROOT / "silver"
SILVER_TEXAS = SILVER / "texas"
SILVER_CALIFORNIA = SILVER / "california"
SILVER_NEW_YORK = SILVER / "new_york"

# Gold Layer: Business-level aggregates, ML-ready datasets
GOLD = DATA_ROOT / "gold"
GOLD_ML_DATASETS = GOLD / "ml_datasets"
GOLD_ANALYTICS = GOLD / "analytics"
TEXAS_GOLD_ANALYTICS = GOLD_ANALYTICS / "texas"

# ==============================================================================
# TEXAS DATA PATHS (Bronze Layer)
# ==============================================================================

TEXAS_BRONZE_CRASHES = BRONZE_TEXAS / "crashes"
TEXAS_BRONZE_WORKZONES = BRONZE_TEXAS / "workzones"
TEXAS_BRONZE_TRAFFIC = BRONZE_TEXAS / "traffic"
TEXAS_BRONZE_WEATHER = BRONZE_TEXAS / "weather"
TEXAS_BRONZE_HPMS = BRONZE_TEXAS / "HPMS2023.gdb"

# ==============================================================================
# TEXAS DATA PATHS (Silver Layer)
# ==============================================================================

TEXAS_SILVER_ROADWAY = SILVER_TEXAS / "roadway"
TEXAS_SILVER_WORKZONES = SILVER_TEXAS / "workzones"
TEXAS_SILVER_TRAFFIC = SILVER_TEXAS / "traffic"
TEXAS_SILVER_WEATHER = SILVER_TEXAS / "weather"

# ==============================================================================
# ML-READY DATASETS (Gold Layer)
# ==============================================================================

CRASH_LEVEL_ML = GOLD_ML_DATASETS / "crash_level"
SEGMENT_LEVEL_ML = GOLD_ML_DATASETS / "segment_level"
CRASH_LEVEL_ARCHIVE = GOLD_ML_DATASETS / "crash_level_archive"

# ==============================================================================
# BACKWARD COMPATIBILITY (Old path names → New architecture)
# ==============================================================================

# Map old variable names to new paths for backward compatibility
RAW_DATA = BRONZE  # old: data/raw → new: data/bronze
PROCESSED_DATA = SILVER  # old: data/processed → new: data/silver
ML_READY = GOLD_ML_DATASETS  # old: data/ml_ready → new: data/gold/ml_datasets

# Texas raw data (backward compatible)
TEXAS_RAW = BRONZE_TEXAS  # old: data/raw/texas → new: data/bronze/texas
TEXAS_CRASHES = TEXAS_BRONZE_CRASHES
TEXAS_WORKZONES = TEXAS_BRONZE_WORKZONES
TEXAS_TRAFFIC = TEXAS_BRONZE_TRAFFIC
TEXAS_WEATHER = TEXAS_BRONZE_WEATHER

# Processed datasets (backward compatible)
CRASH_LEVEL_DATA = CRASH_LEVEL_ML  # old: data/processed/crash_level → new: data/gold/ml_datasets/crash_level
SEGMENT_LEVEL_DATA = SEGMENT_LEVEL_ML  # old: data/processed/segment_level → new: data/gold/ml_datasets/segment_level

# ==============================================================================
# DEFAULT FILES
# ==============================================================================

DEFAULT_CRASH_FILE = TEXAS_BRONZE_CRASHES / "kaggle_us_accidents_texas.csv"
DEFAULT_WORKZONE_FILE = TEXAS_BRONZE_WORKZONES / "texas_wzdx_feed.json"
DEFAULT_AADT_FILE = TEXAS_BRONZE_TRAFFIC / "txdot_aadt_annual.gpkg"
DEFAULT_HPMS_FILE = TEXAS_SILVER_ROADWAY / "hpms_texas_2023.gpkg"

# ==============================================================================
# OUTPUTS
# ==============================================================================

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
MAPS = OUTPUTS_ROOT / "maps"
VISUALIZATIONS = OUTPUTS_ROOT / "visualizations"
MODELS = OUTPUTS_ROOT / "models"
REPORTS = OUTPUTS_ROOT / "reports"

# ==============================================================================
# DIRECTORY INITIALIZATION
# ==============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist"""

    # Bronze layer directories
    bronze_dirs = [
        BRONZE,
        BRONZE_TEXAS, BRONZE_CALIFORNIA, BRONZE_NEW_YORK,
        TEXAS_BRONZE_CRASHES, TEXAS_BRONZE_WORKZONES,
        TEXAS_BRONZE_TRAFFIC, TEXAS_BRONZE_WEATHER
    ]

    # Silver layer directories
    silver_dirs = [
        SILVER,
        SILVER_TEXAS, SILVER_CALIFORNIA, SILVER_NEW_YORK,
        TEXAS_SILVER_ROADWAY, TEXAS_SILVER_WORKZONES,
        TEXAS_SILVER_TRAFFIC, TEXAS_SILVER_WEATHER
    ]

    # Gold layer directories
    gold_dirs = [
        GOLD,
        GOLD_ML_DATASETS, GOLD_ANALYTICS,
        TEXAS_GOLD_ANALYTICS,
        CRASH_LEVEL_ML, SEGMENT_LEVEL_ML, CRASH_LEVEL_ARCHIVE
    ]

    # Output directories
    output_dirs = [
        OUTPUTS_ROOT, MAPS, VISUALIZATIONS, MODELS, REPORTS
    ]

    # Create all directories
    all_dirs = bronze_dirs + silver_dirs + gold_dirs + output_dirs
    for directory in all_dirs:
        directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ARCHITECTURE DOCUMENTATION
# ==============================================================================

ARCHITECTURE_DOCS = """
MEDALLION DATA ARCHITECTURE
===========================

Bronze Layer (data/bronze/):
  - Raw, immutable data as downloaded
  - Original file formats preserved
  - Never modified after download
  - Example: HPMS2023.gdb, kaggle_us_accidents_texas.csv

Silver Layer (data/silver/):
  - Cleaned, validated, and standardized
  - Single source of truth for clean data
  - Deduplication, type fixing, date filtering
  - Example: hpms_texas_2023.gpkg (extracted from GDB)

Gold Layer (data/gold/):
  - Business-level aggregates
  - ML-ready datasets with train/val/test splits
  - Feature engineering applied
  - Example: crash_level/, segment_level/

Path Migration:
  - data/raw → data/bronze
  - data/processed → data/silver
  - data/ml_ready → data/gold/ml_datasets
"""

def print_architecture():
    """Print architecture documentation"""
    print(ARCHITECTURE_DOCS)
