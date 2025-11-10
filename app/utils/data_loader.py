"""
Data loading utilities for Texas Crash Analysis Dashboard
Handles loading and caching of all datasets with smart sampling
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import os

# Base paths - using Medallion architecture (Bronze/Silver/Gold)
BASE_DIR = Path(__file__).parent.parent.parent

# Bronze Layer: Raw data
BRONZE_TEXAS = BASE_DIR / "data" / "bronze" / "texas"
TEXAS_BRONZE_CRASHES = BRONZE_TEXAS / "crashes"
TEXAS_BRONZE_WORKZONES = BRONZE_TEXAS / "workzones"
TEXAS_BRONZE_WEATHER = BRONZE_TEXAS / "weather"

# Gold Layer: ML-ready datasets
GOLD_ML_DATASETS = BASE_DIR / "data" / "gold" / "ml_datasets"
CRASH_LEVEL_ML = GOLD_ML_DATASETS / "crash_level"
SEGMENT_LEVEL_ML = GOLD_ML_DATASETS / "segment_level"

# Backward compatibility aliases
RAW_DATA_DIR = BRONZE_TEXAS
PROCESSED_DATA_DIR = GOLD_ML_DATASETS


@st.cache_data(ttl=3600)
def load_crash_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load Kaggle US Accidents Texas data

    Args:
        sample_size: Number of rows to sample (None for all data)

    Returns:
        DataFrame with crash data
    """
    file_path = RAW_DATA_DIR / "crashes" / "kaggle_us_accidents_texas.csv"

    if sample_size:
        # Read with sampling for performance
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)

    # Parse dates
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    if 'End_Time' in df.columns:
        df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    return df


@st.cache_data(ttl=3600)
def load_austin_crashes(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load Austin crashes data

    Args:
        sample_size: Number of rows to sample (None for all data)

    Returns:
        DataFrame with Austin crash data
    """
    file_path = RAW_DATA_DIR / "crashes" / "austin_crashes_latest.csv"

    if not file_path.exists():
        # Try to find any austin crashes file
        crash_dir = RAW_DATA_DIR / "crashes"
        austin_files = list(crash_dir.glob("austin_crashes_*.csv"))
        if austin_files:
            file_path = sorted(austin_files)[-1]  # Get latest

    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)

    # Parse dates if present
    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')

    return df


@st.cache_data(ttl=3600)
def load_work_zones() -> pd.DataFrame:
    """
    Load work zones data with coordinates extracted from JSON

    Returns:
        DataFrame with work zone data including lat/lon
    """
    import json

    # Load from JSON to get coordinates
    json_path = RAW_DATA_DIR / "workzones" / "texas_wzdx_feed.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract coordinates from geometry_multipoint
    def extract_coords(geom):
        """Extract first coordinate pair from geometry"""
        if pd.isna(geom) or not isinstance(geom, dict):
            return None, None

        coords = geom.get('coordinates', [])
        if coords and len(coords) > 0:
            # Coordinates are [longitude, latitude]
            lon, lat = coords[0]
            return lat, lon
        return None, None

    # Apply coordinate extraction
    df[['beginning_latitude', 'beginning_longitude']] = df['geometry_multipoint'].apply(
        lambda x: pd.Series(extract_coords(x))
    )

    # Parse dates
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    # Add county_name if not present (extract from subidentifier or road_event_id)
    if 'county_name' not in df.columns and 'subidentifier' in df.columns:
        df['county_name'] = df['subidentifier']

    return df


@st.cache_data(ttl=3600)
def load_weather_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load weather data

    Args:
        sample_size: Number of rows to sample (None for all data)

    Returns:
        DataFrame with weather data
    """
    file_path = RAW_DATA_DIR / "weather" / "texas_weather_latest.csv"

    if not file_path.exists():
        # Try to find any weather file
        weather_dir = RAW_DATA_DIR / "weather"
        weather_files = list(weather_dir.glob("texas_weather_*.csv"))
        if weather_files:
            file_path = sorted(weather_files)[-1]  # Get latest

    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)

    # Parse dates
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    return df


@st.cache_data(ttl=3600)
def load_crash_ml_dataset(split: str = 'train', sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load crash-level ML dataset

    Args:
        split: 'train', 'val', or 'test'
        sample_size: Number of rows to sample (None for all data)

    Returns:
        DataFrame with crash-level ML data
    """
    file_path = CRASH_LEVEL_ML / f"{split}_latest.csv"

    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    # Parse dates
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

    return df


@st.cache_data(ttl=3600)
def load_segment_ml_dataset(split: str = 'train', sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load segment-level ML dataset

    Args:
        split: 'train', 'val', or 'test'
        sample_size: Number of rows to sample (None for all data)

    Returns:
        DataFrame with segment-level ML data
    """
    # Segment files don't have 'segment_' prefix in gold layer
    file_path = SEGMENT_LEVEL_ML / f"{split}_latest.csv"

    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)

    return df


def get_data_summary() -> Dict[str, Any]:
    """
    Get summary statistics across all datasets

    Returns:
        Dictionary with summary stats
    """
    summary = {}

    try:
        # Crash data counts (just count lines for performance)
        kaggle_path = TEXAS_BRONZE_CRASHES / "kaggle_us_accidents_texas.csv"
        austin_path = TEXAS_BRONZE_CRASHES / "austin_crashes_latest.csv"

        # Use wc -l for fast line counting
        import subprocess

        if kaggle_path.exists():
            result = subprocess.run(['wc', '-l', str(kaggle_path)], capture_output=True, text=True)
            summary['kaggle_crashes'] = int(result.stdout.split()[0]) - 1  # Subtract header

        if austin_path.exists():
            result = subprocess.run(['wc', '-l', str(austin_path)], capture_output=True, text=True)
            summary['austin_crashes'] = int(result.stdout.split()[0]) - 1

        summary['total_crashes'] = summary.get('kaggle_crashes', 0) + summary.get('austin_crashes', 0)

        # Work zones
        wz_df = load_work_zones()
        summary['work_zones'] = len(wz_df)
        summary['wz_counties'] = wz_df['county_name'].nunique() if 'county_name' in wz_df.columns else 0

        # ML datasets
        crash_ml = load_crash_ml_dataset('train', sample_size=1000)
        summary['ml_crash_features'] = len(crash_ml.columns)

        segment_ml = load_segment_ml_dataset('train', sample_size=1000)
        summary['ml_segment_features'] = len(segment_ml.columns)
        summary['unique_segments'] = segment_ml['segment_id'].nunique() if 'segment_id' in segment_ml.columns else 0

    except Exception as e:
        st.warning(f"Could not compute all summary stats: {e}")

    return summary


@st.cache_data(ttl=3600)
def get_file_sizes() -> Dict[str, str]:
    """
    Get file sizes for all datasets

    Returns:
        Dictionary with file sizes
    """
    import os

    def format_size(bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0

    sizes = {}

    # Raw data from bronze layer
    kaggle_path = TEXAS_BRONZE_CRASHES / "kaggle_us_accidents_texas.csv"
    if kaggle_path.exists():
        sizes['kaggle_crashes'] = format_size(os.path.getsize(kaggle_path))

    austin_path = TEXAS_BRONZE_CRASHES / "austin_crashes_latest.csv"
    if austin_path.exists():
        sizes['austin_crashes'] = format_size(os.path.getsize(austin_path))

    # ML datasets from gold layer
    crash_ml_path = CRASH_LEVEL_ML / "train_latest.csv"
    if crash_ml_path.exists():
        sizes['crash_ml_train'] = format_size(os.path.getsize(crash_ml_path))

    # Segment files don't have 'segment_' prefix
    segment_ml_path = SEGMENT_LEVEL_ML / "train_latest.csv"
    if segment_ml_path.exists():
        sizes['segment_ml_train'] = format_size(os.path.getsize(segment_ml_path))

    return sizes
