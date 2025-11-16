"""
HPMS data loading utilities with caching and filtering.
Eliminates duplicate loading logic across prediction pages.
"""

import streamlit as st
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple
from shapely.geometry import box, Point

from config import HPMS_TEXAS_2023


@st.cache_resource
def load_hpms_full() -> gpd.GeoDataFrame:
    """
    Load full HPMS dataset with caching.

    This function is cached to avoid reloading the 1.1 GB file
    on every interaction. First load takes 5-30 seconds, subsequent
    loads are instant.

    Returns:
        GeoDataFrame with HPMS road segments

    Raises:
        FileNotFoundError: If HPMS file doesn't exist
    """
    if not HPMS_TEXAS_2023.exists():
        raise FileNotFoundError(
            f"HPMS file not found: {HPMS_TEXAS_2023}\n"
            f"Please ensure the file exists before running the app."
        )

    with st.spinner("Loading HPMS roadway data (1.1 GB)..."):
        hpms = gpd.read_file(HPMS_TEXAS_2023)

    # Ensure CRS is set to WGS84 (lat/lon)
    if hpms.crs is None:
        hpms = hpms.set_crs('EPSG:4326')
    elif hpms.crs != 'EPSG:4326':
        hpms = hpms.to_crs('EPSG:4326')

    st.success(f"✅ Loaded {len(hpms):,} road segments from HPMS")

    return hpms


def filter_hpms_by_bbox(
    hpms: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    buffer: float = 0.01
) -> gpd.GeoDataFrame:
    """
    Filter HPMS segments by bounding box around a point.

    Args:
        hpms: Full HPMS GeoDataFrame
        lat: Center latitude
        lon: Center longitude
        buffer: Buffer distance in degrees (~0.01 = ~1km)

    Returns:
        Filtered GeoDataFrame with segments in bounding box
    """
    bbox = box(
        lon - buffer,
        lat - buffer,
        lon + buffer,
        lat + buffer
    )

    # Spatial filter
    filtered = hpms[hpms.intersects(bbox)].copy()

    return filtered


def load_hpms_for_location(
    lat: float,
    lon: float,
    buffer: float = 0.01
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load HPMS data filtered to area around a location.

    Convenience function that loads full HPMS (cached) and filters
    to bounding box around the specified location.

    Args:
        lat: Center latitude
        lon: Center longitude
        buffer: Buffer distance in degrees (~0.01 = ~1km)

    Returns:
        Tuple of (full_hpms, filtered_hpms)
    """
    # Load full dataset (cached)
    hpms_full = load_hpms_full()

    # Filter to bounding box
    hpms_filtered = filter_hpms_by_bbox(hpms_full, lat, lon, buffer)

    if len(hpms_filtered) == 0:
        st.warning(
            f"⚠️ No road segments found within {buffer * 111:.1f} km "
            f"of location ({lat:.4f}, {lon:.4f})"
        )

    return hpms_full, hpms_filtered


def get_segment_at_location(
    hpms: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    max_distance: float = 0.001
) -> Optional[dict]:
    """
    Find the closest HPMS segment to a point.

    Args:
        hpms: HPMS GeoDataFrame
        lat: Point latitude
        lon: Point longitude
        max_distance: Maximum distance to search (degrees)

    Returns:
        Dictionary with segment attributes, or None if no segment found
    """
    point = Point(lon, lat)

    # Calculate distances
    hpms = hpms.copy()
    hpms['distance'] = hpms.geometry.distance(point)

    # Find closest
    closest_idx = hpms['distance'].idxmin()
    closest_dist = hpms.loc[closest_idx, 'distance']

    if closest_dist > max_distance:
        return None

    segment = hpms.loc[closest_idx].to_dict()
    segment['distance_km'] = closest_dist * 111  # Rough conversion to km

    return segment
