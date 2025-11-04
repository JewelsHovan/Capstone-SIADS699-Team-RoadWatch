"""
Work Zone Integration Module

Integrates work zone data with crash data using spatial and temporal joins.

Functions:
    - integrate_workzones_spatial: Full spatial-temporal join (requires date overlap)
    - add_workzone_proximity_features: Spatial-only features (proximity, density)

Author: Data Engineering Team
Date: 2025-11-02
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_workzones(wz_file, verbose=True):
    """
    Load work zones from WZDx JSON feed

    Args:
        wz_file: Path to work zone JSON file
        verbose: Print progress

    Returns:
        GeoDataFrame with work zones
    """
    import json

    if verbose:
        print(f"Loading work zones from {wz_file}...")

    with open(wz_file, 'r') as f:
        wz_data = json.load(f)

    wz_df = pd.DataFrame(wz_data)

    # Extract coordinates from geometry_multipoint
    def extract_coords(geom):
        if pd.isna(geom) or not isinstance(geom, dict):
            return None, None
        coords = geom.get('coordinates', [])
        if coords and len(coords) > 0:
            lon, lat = coords[0]  # GeoJSON format: [lon, lat]
            return lat, lon
        return None, None

    wz_df[['lat', 'lon']] = wz_df['geometry_multipoint'].apply(
        lambda x: pd.Series(extract_coords(x))
    )

    # Remove work zones without coordinates
    wz_df = wz_df.dropna(subset=['lat', 'lon'])

    # Create GeoDataFrame
    wz_gdf = gpd.GeoDataFrame(
        wz_df,
        geometry=gpd.points_from_xy(wz_df['lon'], wz_df['lat']),
        crs='EPSG:4326'
    )

    # Parse dates
    if 'start_date' in wz_gdf.columns:
        wz_gdf['start_date'] = pd.to_datetime(wz_gdf['start_date'], errors='coerce')
    if 'end_date' in wz_gdf.columns:
        wz_gdf['end_date'] = pd.to_datetime(wz_gdf['end_date'], errors='coerce')

    if verbose:
        print(f"  ✓ Loaded {len(wz_gdf):,} work zones")
        if 'start_date' in wz_gdf.columns:
            print(f"  Date range: {wz_gdf['start_date'].min()} to {wz_gdf['end_date'].max()}")

    return wz_gdf


def integrate_workzones_spatial(crashes_gdf, workzones_gdf, buffer_meters=500, verbose=True):
    """
    Integrate work zones with crashes using spatial AND temporal matching

    WARNING: This requires temporal overlap between crash dates and work zone dates!

    Args:
        crashes_gdf: GeoDataFrame with crashes (must have Start_Time and geometry)
        workzones_gdf: GeoDataFrame with work zones (must have start_date, end_date, geometry)
        buffer_meters: Matching distance in meters (default 500m)
        verbose: Print progress

    Returns:
        DataFrame with work zone features added
    """

    if verbose:
        print(f"\nIntegrating work zones (spatial-temporal join)...")
        print(f"  Buffer distance: {buffer_meters}m")

    # Convert to UTM for metric buffers (Texas: EPSG:3083)
    crashes_utm = crashes_gdf.to_crs('EPSG:3083')
    wz_utm = workzones_gdf.to_crs('EPSG:3083')

    # Create buffers around work zones
    wz_utm['buffer'] = wz_utm.geometry.buffer(buffer_meters)
    wz_buffer = wz_utm.set_geometry('buffer')

    # Parse crash dates
    crashes_utm['crash_date'] = pd.to_datetime(crashes_utm['Start_Time']).dt.date

    # Check temporal overlap
    crash_date_range = (crashes_utm['crash_date'].min(), crashes_utm['crash_date'].max())
    wz_date_range = (workzones_gdf['start_date'].min().date(), workzones_gdf['end_date'].max().date())

    if verbose:
        print(f"  Crash dates: {crash_date_range[0]} to {crash_date_range[1]}")
        print(f"  Work zone dates: {wz_date_range[0]} to {wz_date_range[1]}")

    # Check for overlap
    has_overlap = (crash_date_range[1] >= wz_date_range[0]) and (crash_date_range[0] <= wz_date_range[1])

    if not has_overlap:
        print(f"\n  ⚠️  WARNING: No temporal overlap detected!")
        print(f"     Crashes: {crash_date_range[0]} to {crash_date_range[1]}")
        print(f"     Work zones: {wz_date_range[0]} to {wz_date_range[1]}")
        print(f"     No crashes will match work zones temporally.")
        print(f"     Consider using add_workzone_proximity_features() instead.")

        # Add empty columns
        crashes_utm['in_work_zone'] = 0
        crashes_utm['wz_distance_m'] = np.nan
        return pd.DataFrame(crashes_utm.drop(columns=['geometry', 'buffer', 'crash_date'], errors='ignore'))

    # Spatial-temporal join
    if verbose:
        print(f"  Performing spatial-temporal join...")

    # Initialize work zone features
    crashes_utm['in_work_zone'] = 0
    crashes_utm['wz_distance_m'] = np.nan
    crashes_utm['wz_type'] = None
    crashes_utm['wz_road_name'] = None

    # For each crash, find overlapping work zones
    matched_count = 0

    for idx in crashes_utm.index:
        crash = crashes_utm.loc[idx]
        crash_point = crash.geometry
        crash_date = crash.crash_date

        # Find work zones that overlap spatially and temporally
        wz_matches = wz_buffer[
            (wz_buffer.buffer.contains(crash_point)) &
            (wz_buffer['start_date'].dt.date <= crash_date) &
            (wz_buffer['end_date'].dt.date >= crash_date)
        ]

        if len(wz_matches) > 0:
            # Crash occurred in/near active work zone
            wz = wz_matches.iloc[0]  # Take first match

            crashes_utm.loc[idx, 'in_work_zone'] = 1
            crashes_utm.loc[idx, 'wz_distance_m'] = crash_point.distance(wz.geometry)
            crashes_utm.loc[idx, 'wz_type'] = wz.get('type_of_work', 'unknown')
            crashes_utm.loc[idx, 'wz_road_name'] = wz.get('road_name', None)

            matched_count += 1

    if verbose:
        print(f"  ✓ Matched {matched_count:,} / {len(crashes_utm):,} crashes ({matched_count/len(crashes_utm)*100:.2f}%)")

    # Convert back to DataFrame
    result_df = pd.DataFrame(crashes_utm.drop(columns=['geometry', 'buffer', 'crash_date'], errors='ignore'))

    return result_df


def add_workzone_proximity_features(crashes_gdf, workzones_gdf, verbose=True):
    """
    Add spatial proximity features to work zones (no temporal matching)

    Use this when work zone dates don't overlap with crash dates.
    Creates features like:
    - Distance to nearest work zone
    - Number of work zones within 1km
    - Work zone density in area

    Args:
        crashes_gdf: GeoDataFrame with crashes
        workzones_gdf: GeoDataFrame with work zones
        verbose: Print progress

    Returns:
        DataFrame with proximity features added
    """

    if verbose:
        print(f"\nAdding work zone proximity features...")

    # Convert to UTM
    crashes_utm = crashes_gdf.to_crs('EPSG:3083')
    wz_utm = workzones_gdf.to_crs('EPSG:3083')

    # Nearest work zone distance
    if verbose:
        print(f"  Computing nearest work zone distances...")

    crashes_with_dist = crashes_utm.sjoin_nearest(
        wz_utm[['geometry']],
        how='left',
        distance_col='wz_nearest_distance_m'
    )

    # Work zone density (count within 1km) - simplified approach
    if verbose:
        print(f"  Computing work zone density (1km radius)...")

    # Initialize count column
    crashes_with_dist['wz_count_1km'] = 0

    # For each crash, count work zones within 1km
    for idx in crashes_utm.index:
        crash_point = crashes_utm.loc[idx, 'geometry']
        # Count work zones within 1km
        distances = wz_utm.geometry.distance(crash_point)
        count_within_1km = (distances <= 1000).sum()
        crashes_with_dist.loc[idx, 'wz_count_1km'] = count_within_1km

    # Work zone density categories
    crashes_with_dist['wz_density_cat'] = pd.cut(
        crashes_with_dist['wz_count_1km'],
        bins=[-1, 0, 2, 5, np.inf],
        labels=['none', 'low', 'medium', 'high']
    )

    if verbose:
        print(f"  ✓ Features added:")
        print(f"    - wz_nearest_distance_m: Distance to nearest work zone")
        print(f"    - wz_count_1km: Work zones within 1km")
        print(f"    - wz_density_cat: Density category (none/low/medium/high)")
        print(f"\n  Summary:")
        print(f"    Mean distance to nearest WZ: {crashes_with_dist['wz_nearest_distance_m'].mean():.0f}m")
        print(f"    Crashes with WZ within 1km: {(crashes_with_dist['wz_count_1km'] > 0).sum():,} ({(crashes_with_dist['wz_count_1km'] > 0).mean()*100:.1f}%)")

    # Convert back to DataFrame
    result_df = pd.DataFrame(crashes_with_dist.drop(columns=['geometry'], errors='ignore'))

    # Drop index_right to avoid conflicts with subsequent spatial joins
    result_df = result_df.drop(columns=['index_right'], errors='ignore')

    return result_df


if __name__ == "__main__":
    # Test the integration
    print("Work Zone Integration Module - Test")
    print("=" * 80)

    # This would be run as a test
    print("\nTo use this module:")
    print("  from data_engineering.integrate import integrate_workzones_spatial")
    print("  crashes_with_wz = integrate_workzones_spatial(crashes_gdf, wz_gdf)")
