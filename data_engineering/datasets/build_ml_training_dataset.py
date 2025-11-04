#!/usr/bin/env python3
"""
Build ML Training Dataset for Work Zone Crash Risk Prediction

This script creates a training dataset by:
1. Loading historical crash data (2016-2023)
2. Extracting road features using OSMnx
3. Attaching AADT traffic data
4. Engineering temporal and weather features
5. Creating train/val/test splits

Usage:
    python build_ml_training_dataset.py --city Houston --sample 10000
    python build_ml_training_dataset.py --all-cities --output data/processed/ml_dataset.csv
"""

import argparse
import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for data_engineering imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_crashes(crash_file, cities=None, sample_size=None):
    """Load and filter crash data"""
    print(f'\n{"="*80}')
    print('STEP 1: LOADING CRASH DATA')
    print(f'{"="*80}')

    print(f'\nReading {crash_file}...')
    crashes_df = pd.read_csv(crash_file)

    print(f'  Total crashes in file: {len(crashes_df):,}')

    # Filter by cities if specified
    if cities:
        crashes_df = crashes_df[crashes_df['City'].isin(cities)]
        print(f'  After city filter: {len(crashes_df):,}')

    # Sample if specified
    if sample_size and sample_size < len(crashes_df):
        crashes_df = crashes_df.sample(sample_size, random_state=42)
        print(f'  After sampling: {len(crashes_df):,}')

    # Parse datetime
    crashes_df['Start_Time'] = pd.to_datetime(crashes_df['Start_Time'], format='mixed')

    # Add year for splitting
    crashes_df['year'] = crashes_df['Start_Time'].dt.year

    print(f'\nDate range: {crashes_df["Start_Time"].min()} to {crashes_df["Start_Time"].max()}')
    print(f'Years: {sorted(crashes_df["year"].unique())}')

    # Create target variable BEFORE removing Severity
    print(f'\n📊 Creating target variable (high_severity)...')
    if 'Severity' in crashes_df.columns:
        crashes_df['high_severity'] = (crashes_df['Severity'] >= 3).astype(int)
        high_sev_count = crashes_df['high_severity'].sum()
        high_sev_pct = high_sev_count / len(crashes_df) * 100
        print(f'   ✓ Created target: {high_sev_count:,} high severity ({high_sev_pct:.1f}%)')
    else:
        print('   ⚠️  Severity column not found - cannot create target')

    # Remove data leakage features
    print(f'\n🚨 Removing data leakage features...')
    LEAKAGE_FEATURES = ['Severity', 'End_Time', 'End_Lat', 'End_Lng']

    features_to_remove = [f for f in LEAKAGE_FEATURES if f in crashes_df.columns]

    if features_to_remove:
        print(f'   Removing: {", ".join(features_to_remove)}')
        crashes_df = crashes_df.drop(columns=features_to_remove)
        print(f'   ✓ Removed {len(features_to_remove)} leakage features')
    else:
        print(f'   ✓ No leakage features found (already clean)')

    # Convert to GeoDataFrame
    crashes_gdf = gpd.GeoDataFrame(
        crashes_df,
        geometry=[Point(lon, lat) for lon, lat in zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])],
        crs='EPSG:4326'
    )

    print(f'\n✓ Loaded {len(crashes_gdf):,} crashes')

    return crashes_gdf


def extract_road_features_batch(crashes_gdf, batch_size=1000, buffer_dist=50):
    """
    Extract road features for crashes in batches using OSMnx

    Args:
        crashes_gdf: GeoDataFrame of crashes
        batch_size: Number of crashes to process per batch
        buffer_dist: Buffer distance in meters around crash for road matching
    """
    print(f'\n{"="*80}')
    print('STEP 2: EXTRACTING ROAD FEATURES FROM OSMnx')
    print(f'{"="*80}')

    print(f'\nProcessing {len(crashes_gdf):,} crashes in batches of {batch_size}...')
    print(f'Buffer distance: {buffer_dist}m')

    # Group crashes by city for efficiency
    cities = crashes_gdf['City'].value_counts()
    print(f'\nCrashes by city:')
    for city, count in cities.head(10).items():
        print(f'  {city}: {count:,}')

    all_results = []

    # Process top 10 cities (covers >95% of crashes)
    for city in cities.index[:10]:
        print(f'\n--- Processing {city} ---')
        city_crashes = crashes_gdf[crashes_gdf['City'] == city].copy()

        print(f'  Crashes: {len(city_crashes):,}')

        # Get city center and download road network
        center_lat = city_crashes.geometry.y.mean()
        center_lon = city_crashes.geometry.x.mean()

        # Determine appropriate radius based on crash extent
        lat_range = city_crashes.geometry.y.max() - city_crashes.geometry.y.min()
        lon_range = city_crashes.geometry.x.max() - city_crashes.geometry.x.min()
        max_range = max(lat_range, lon_range) * 111000  # Convert degrees to meters (approx)
        dist = min(max_range / 2 + 1000, 15000)  # Add 1km buffer, cap at 15km radius

        print(f'  Downloading road network (center: {center_lat:.4f}, {center_lon:.4f}, radius: {dist:.0f}m)...')

        try:
            G = ox.graph_from_point(
                (center_lat, center_lon),
                dist=dist,
                network_type='drive',
                simplify=True
            )

            edges_gdf = ox.graph_to_gdfs(G, nodes=False)
            print(f'  ✓ Road network: {len(edges_gdf)} segments')

            # Create edge ID
            edges_gdf['edge_id'] = range(len(edges_gdf))

            # Convert to UTM for distance calculations
            edges_utm = edges_gdf.to_crs('EPSG:3083')
            city_crashes_utm = city_crashes.to_crs('EPSG:3083')

            # Spatial join - use sjoin_nearest to avoid duplicates
            print(f'  Matching crashes to road segments...')
            crashes_with_roads = city_crashes_utm.sjoin_nearest(
                edges_utm[['geometry', 'edge_id', 'highway', 'name', 'lanes',
                          'maxspeed', 'oneway', 'bridge', 'tunnel']],
                how='left',
                max_distance=buffer_dist,
                distance_col='distance_to_road_m'
            )

            # Calculate road features
            def parse_val(val):
                return val[0] if isinstance(val, list) else val

            crashes_with_roads['highway_type'] = crashes_with_roads['highway'].apply(parse_val)

            def parse_lanes(val):
                try:
                    return float(val[0] if isinstance(val, list) else val)
                except:
                    return np.nan

            crashes_with_roads['num_lanes'] = crashes_with_roads['lanes'].apply(parse_lanes)

            # Parse speed limit
            def parse_speed(val):
                try:
                    if pd.isna(val):
                        return np.nan
                    val_str = str(val[0] if isinstance(val, list) else val)
                    # Extract numeric value
                    import re
                    match = re.search(r'(\d+)', val_str)
                    return float(match.group(1)) if match else np.nan
                except:
                    return np.nan

            # Rename to avoid conflicts with HPMS data
            crashes_with_roads['osmnx_speed_limit'] = crashes_with_roads['maxspeed'].apply(parse_speed)

            # Binary features
            crashes_with_roads['is_oneway'] = crashes_with_roads['oneway'].fillna(False).astype(int)
            crashes_with_roads['is_bridge'] = crashes_with_roads['bridge'].notna().astype(int)
            crashes_with_roads['is_tunnel'] = crashes_with_roads['tunnel'].notna().astype(int)
            crashes_with_roads['road_name'] = crashes_with_roads['name']

            # Drop geometry for efficiency
            result_df = pd.DataFrame(crashes_with_roads.drop(columns=['geometry'], errors='ignore'))

            # Drop index_right to avoid conflicts with subsequent spatial joins
            result_df = result_df.drop(columns=['index_right'], errors='ignore')

            all_results.append(result_df)

            matched = crashes_with_roads['edge_id'].notna().sum()
            print(f'  ✓ Matched {matched:,} / {len(city_crashes):,} crashes ({matched/len(city_crashes)*100:.1f}%)')

        except Exception as e:
            print(f'  ✗ Error processing {city}: {e}')
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f'\n✓ Total crashes with road features: {len(combined_df):,}')
        return combined_df
    else:
        print('\n✗ No results generated')
        return None


def integrate_hpms_road_characteristics(crashes_df, hpms_file):
    """Integrate HPMS road characteristics (speed limits, lanes, IRI)"""
    print(f'\n{"="*80}')
    print('STEP 3: INTEGRATING HPMS ROAD CHARACTERISTICS')
    print(f'{"="*80}')

    try:
        from config.paths import TEXAS_RAW

        # Check if HPMS file exists
        if hpms_file is None:
            hpms_file = TEXAS_RAW / 'roadway_characteristics' / 'hpms_texas_2023.gpkg'

        if not Path(hpms_file).exists():
            print(f'\n⚠️  HPMS file not found: {hpms_file}')
            print('   Skipping HPMS integration (speed limits, lanes, IRI will be missing)')
            print('   Download HPMS data and run: python data_engineering/download/download_hpms_texas.py')
            return crashes_df

        print(f'\nLoading HPMS data from {hpms_file}...')
        hpms_gdf = gpd.read_file(hpms_file)
        print(f'  ✓ Loaded {len(hpms_gdf):,} road segments')

        # Convert to GeoDataFrame if needed
        if not isinstance(crashes_df, gpd.GeoDataFrame):
            crashes_df['geometry'] = [Point(lon, lat) for lon, lat in
                                       zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])]
            crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry='geometry', crs='EPSG:4326')
        else:
            crashes_gdf = crashes_df

        # Convert to UTM for accurate distance calculations
        print('  Performing spatial join (nearest road segment within 100m)...')
        crashes_utm = crashes_gdf.to_crs('EPSG:3083')
        hpms_utm = hpms_gdf.to_crs('EPSG:3083')

        # Spatial join to nearest road segment
        crashes_with_hpms = crashes_utm.sjoin_nearest(
            hpms_utm[['geometry', 'speed_limit', 'through_lanes', 'iri', 'f_system',
                     'nhs', 'aadt', 'median_type', 'shoulder_type', 'lane_width']],
            how='left',
            max_distance=100,  # Only match if within 100m
            distance_col='hpms_distance_m'
        )

        # Rename columns for clarity
        crashes_with_hpms = crashes_with_hpms.rename(columns={
            'speed_limit': 'hpms_speed_limit',
            'through_lanes': 'hpms_lanes',
            'iri': 'hpms_iri',
            'f_system': 'hpms_functional_class',
            'nhs': 'hpms_nhs',
            'aadt': 'hpms_aadt',
            'median_type': 'hpms_median_type',
            'shoulder_type': 'hpms_shoulder_type',
            'lane_width': 'hpms_lane_width'
        })

        # Convert back to DataFrame
        result_df = pd.DataFrame(crashes_with_hpms.drop(columns=['geometry'], errors='ignore'))

        # Drop index_right to avoid conflicts with subsequent spatial joins
        result_df = result_df.drop(columns=['index_right'], errors='ignore')

        # Print matching stats
        matched = result_df['hpms_speed_limit'].notna().sum()
        print(f'  ✓ Matched {matched:,} / {len(result_df):,} crashes ({matched/len(result_df)*100:.1f}%)')
        print(f'  Mean distance to road: {result_df["hpms_distance_m"].mean():.1f}m')

        # Feature completeness
        print('\n  HPMS Feature Completeness:')
        print(f'    Speed limits: {result_df["hpms_speed_limit"].notna().sum():,} ({result_df["hpms_speed_limit"].notna().mean()*100:.1f}%)')
        print(f'    Lane counts:  {result_df["hpms_lanes"].notna().sum():,} ({result_df["hpms_lanes"].notna().mean()*100:.1f}%)')
        print(f'    IRI (pavement): {result_df["hpms_iri"].notna().sum():,} ({result_df["hpms_iri"].notna().mean()*100:.1f}%)')

        return result_df

    except Exception as e:
        print(f'\n⚠️  HPMS integration failed: {e}')
        print(f'   Continuing without HPMS features...')
        return crashes_df


def integrate_work_zones(crashes_df, workzone_file):
    """Integrate work zone proximity features"""
    print(f'\n{"="*80}')
    print('STEP 4: INTEGRATING WORK ZONE FEATURES')
    print(f'{"="*80}')

    try:
        from data_engineering.integrate.integrate_workzones import (
            load_workzones,
            add_workzone_proximity_features
        )

        print(f'\nLoading work zones from {workzone_file}...')
        wz_gdf = load_workzones(workzone_file, verbose=False)

        # Convert to GeoDataFrame if needed
        if not isinstance(crashes_df, gpd.GeoDataFrame):
            crashes_df['geometry'] = [Point(lon, lat) for lon, lat in
                                       zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])]
            crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry='geometry', crs='EPSG:4326')
        else:
            crashes_gdf = crashes_df

        # Add proximity features
        crashes_with_wz = add_workzone_proximity_features(crashes_gdf, wz_gdf, verbose=True)

        # Clean up extra columns from spatial join
        cols_to_drop = [col for col in crashes_with_wz.columns if col in ['index_right', 'index_left']]
        if cols_to_drop:
            crashes_with_wz = crashes_with_wz.drop(columns=cols_to_drop)

        return crashes_with_wz

    except Exception as e:
        print(f'\n⚠️  Work zone integration failed: {e}')
        print(f'   Continuing without work zone features...')
        return crashes_df


def add_lighting_conditions(crashes_df):
    """Add lighting condition features using sunrise/sunset API"""
    print(f'\n{"="*80}')
    print('STEP 5: ADDING LIGHTING CONDITION FEATURES')
    print(f'{"="*80}')

    try:
        from data_engineering.features.add_lighting_conditions import add_lighting_features

        print('\nAdding sunrise/sunset based lighting features...')
        crashes_with_lighting = add_lighting_features(
            crashes_df,
            time_col='Start_Time',
            lat_col='Start_Lat',
            lon_col='Start_Lng',
            verbose=True,
            rate_limit_delay=0.05  # Faster rate limit (20 requests/sec)
        )

        return crashes_with_lighting

    except Exception as e:
        print(f'\n⚠️  Lighting features failed: {e}')
        print(f'   Continuing without lighting features...')
        return crashes_df


def attach_aadt_traffic(crashes_df, aadt_file):
    """Attach AADT traffic data using nearest station"""
    print(f'\n{"="*80}')
    print('STEP 6: ATTACHING AADT TRAFFIC DATA')
    print(f'{"="*80}')

    print(f'\nLoading AADT data from {aadt_file}...')
    aadt_gdf = gpd.read_file(aadt_file)

    print(f'  AADT stations: {len(aadt_gdf):,}')

    # Convert to GeoDataFrame if not already
    if not isinstance(crashes_df, gpd.GeoDataFrame):
        crashes_df['geometry'] = [Point(lon, lat) for lon, lat in
                                   zip(crashes_df['Start_Lng'], crashes_df['Start_Lat'])]
        crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry='geometry', crs='EPSG:4326')
    else:
        crashes_gdf = crashes_df

    # Convert to UTM
    crashes_utm = crashes_gdf.to_crs('EPSG:3083')
    aadt_utm = aadt_gdf[['geometry', 'AADT_RPT_QTY']].to_crs('EPSG:3083')

    print(f'  Performing spatial join (nearest station)...')
    crashes_with_aadt = crashes_utm.sjoin_nearest(
        aadt_utm,
        how='left',
        distance_col='distance_to_aadt_m'
    )

    crashes_with_aadt = crashes_with_aadt.rename(columns={'AADT_RPT_QTY': 'aadt'})

    # Convert back to regular DataFrame
    result_df = pd.DataFrame(crashes_with_aadt.drop(columns=['geometry'], errors='ignore'))

    # Drop index_right to avoid conflicts with subsequent spatial joins
    result_df = result_df.drop(columns=['index_right'], errors='ignore')

    matched = result_df['aadt'].notna().sum()
    print(f'  ✓ Matched {matched:,} / {len(result_df):,} crashes ({matched/len(result_df)*100:.1f}%)')
    print(f'  Mean distance to station: {result_df["distance_to_aadt_m"].mean():.0f}m')

    return result_df


def engineer_features(df):
    """Engineer temporal, weather, and categorical features"""
    print(f'\n{"="*80}')
    print('STEP 7: FEATURE ENGINEERING')
    print(f'{"="*80}')

    print('\nCreating features...')

    # Temporal features
    df['hour'] = pd.to_datetime(df['Start_Time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['Start_Time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['Start_Time']).dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = (
        ((df['hour'] >= 6) & (df['hour'] <= 9)) |
        ((df['hour'] >= 16) & (df['hour'] <= 19))
    ).astype(int)

    # Time of day
    def time_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    df['time_of_day'] = df['hour'].apply(time_of_day)

    # Weather features
    def categorize_weather(condition):
        if pd.isna(condition):
            return 'unknown'
        condition = str(condition).lower()
        if 'clear' in condition or 'fair' in condition:
            return 'clear'
        elif 'cloud' in condition or 'overcast' in condition:
            return 'cloudy'
        elif 'rain' in condition or 'drizzle' in condition or 'shower' in condition:
            return 'rain'
        elif 'fog' in condition or 'mist' in condition:
            return 'fog'
        elif 'snow' in condition or 'sleet' in condition:
            return 'snow'
        elif 'thunder' in condition or 'storm' in condition:
            return 'storm'
        else:
            return 'other'

    df['weather_category'] = df['Weather_Condition'].apply(categorize_weather)
    df['adverse_weather'] = df['weather_category'].isin(['rain', 'fog', 'snow', 'storm']).astype(int)
    df['low_visibility'] = (df['Visibility(mi)'] < 5).astype(int)

    # Temperature categories
    def temp_category(temp):
        if pd.isna(temp):
            return 'unknown'
        if temp < 32:
            return 'freezing'
        elif temp < 50:
            return 'cold'
        elif temp < 70:
            return 'mild'
        elif temp < 85:
            return 'warm'
        else:
            return 'hot'

    df['temp_category'] = df['Temperature(F)'].apply(temp_category)

    # Location features
    urban_cities = ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth',
                   'El Paso', 'Arlington', 'Corpus Christi']
    df['is_urban'] = df['City'].isin(urban_cities).astype(int)

    # Target variable (now created in load_crashes to avoid using leakage feature Severity)
    if 'high_severity' not in df.columns:
        print('  ⚠️  Target variable high_severity not found - may need to check load_crashes()')
        # Fallback: create if Severity still exists (shouldn't happen)
        if 'Severity' in df.columns:
            df['high_severity'] = (df['Severity'] >= 3).astype(int)

    print('  ✓ Temporal: hour, day_of_week, month, is_weekend, is_rush_hour, time_of_day')
    print('  ✓ Weather: weather_category, adverse_weather, low_visibility, temp_category')
    print('  ✓ Location: is_urban')
    if 'high_severity' in df.columns:
        print('  ✓ Target: high_severity (created earlier to avoid leakage)')

    return df


def create_train_val_test_split(df):
    """Split data by year: 2016-2021 train, 2022 val, 2023 test"""
    print(f'\n{"="*80}')
    print('STEP 8: CREATING TRAIN/VAL/TEST SPLITS')
    print(f'{"="*80}')

    train = df[df['year'] <= 2021].copy()
    val = df[df['year'] == 2022].copy()
    test = df[df['year'] == 2023].copy()

    print(f'\nTrain (2016-2021): {len(train):,} samples ({len(train)/len(df)*100:.1f}%)')
    print(f'Val   (2022):      {len(val):,} samples ({len(val)/len(df)*100:.1f}%)')
    print(f'Test  (2023):      {len(test):,} samples ({len(test)/len(df)*100:.1f}%)')

    # Check class balance
    print('\nClass distribution (high_severity):')
    print(f'  Train: {train["high_severity"].sum():,} / {len(train):,} ({train["high_severity"].mean()*100:.1f}%)')
    print(f'  Val:   {val["high_severity"].sum():,} / {len(val):,} ({val["high_severity"].mean()*100:.1f}%)')
    print(f'  Test:  {test["high_severity"].sum():,} / {len(test):,} ({test["high_severity"].mean()*100:.1f}%)')

    return train, val, test


def save_datasets(train, val, test, output_dir):
    """Save train/val/test datasets"""
    print(f'\n{"="*80}')
    print('STEP 9: SAVING DATASETS')
    print(f'{"="*80}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_path = output_dir / f'train_{timestamp}.csv'
    val_path = output_dir / f'val_{timestamp}.csv'
    test_path = output_dir / f'test_{timestamp}.csv'

    print(f'\nSaving datasets to {output_dir}...')

    train.to_csv(train_path, index=False)
    print(f'  ✓ Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)')

    val.to_csv(val_path, index=False)
    print(f'  ✓ Val:   {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)')

    test.to_csv(test_path, index=False)
    print(f'  ✓ Test:  {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)')

    # Create symlinks to latest
    for name, path in [('train_latest.csv', train_path),
                       ('val_latest.csv', val_path),
                       ('test_latest.csv', test_path)]:
        latest_link = output_dir / name
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.name)

    print(f'\n  ✓ Created symlinks: train_latest.csv, val_latest.csv, test_latest.csv')

    return train_path, val_path, test_path


def print_dataset_summary(train, val, test):
    """Print summary statistics"""
    print(f'\n{"="*80}')
    print('DATASET SUMMARY')
    print(f'{"="*80}')

    print(f'\nTotal samples: {len(train) + len(val) + len(test):,}')
    print(f'  Train: {len(train):,}')
    print(f'  Val:   {len(val):,}')
    print(f'  Test:  {len(test):,}')

    # Feature counts
    feature_cols = [col for col in train.columns if col not in ['ID', 'Start_Time', 'End_Time']]
    print(f'\nTotal features: {len(feature_cols)}')

    # Completeness
    print('\nFeature completeness (train):')
    important_features = ['highway_type', 'num_lanes', 'speed_limit', 'aadt',
                         'hpms_speed_limit', 'hpms_lanes', 'hpms_iri',
                         'lighting_condition', 'is_civil_twilight',
                         'wz_nearest_distance_m', 'wz_count_1km',
                         'Temperature(F)', 'Visibility(mi)', 'weather_category']
    for feat in important_features:
        if feat in train.columns:
            pct = train[feat].notna().sum() / len(train) * 100
            print(f'  {feat:25s}: {pct:5.1f}%')

    print('\n' + '='*80)
    print('✅ DATASET CREATION COMPLETE!')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Build ML training dataset for work zone crash risk prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--crash-file', type=str,
                       default='data/raw/texas/crashes/kaggle_us_accidents_texas.csv',
                       help='Path to crash data CSV')
    parser.add_argument('--aadt-file', type=str,
                       default='data/raw/texas/traffic/txdot_aadt_annual.gpkg',
                       help='Path to AADT traffic data')
    parser.add_argument('--workzone-file', type=str,
                       default='data/raw/texas/workzones/texas_wzdx_feed.json',
                       help='Path to work zone JSON file')
    parser.add_argument('--hpms-file', type=str,
                       default=None,
                       help='Path to HPMS file (default: data/raw/texas/roadway_characteristics/hpms_texas_2023.gpkg)')
    parser.add_argument('--cities', type=str, nargs='+',
                       help='Cities to include (default: top 5 cities)')
    parser.add_argument('--sample', type=int,
                       help='Sample size for testing (default: use all)')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/crash_level',
                       help='Output directory for datasets')
    parser.add_argument('--skip-osmnx', action='store_true',
                       help='Skip OSMnx road feature extraction (faster for testing)')
    parser.add_argument('--skip-hpms', action='store_true',
                       help='Skip HPMS road characteristics (faster for testing)')
    parser.add_argument('--skip-lighting', action='store_true',
                       help='Skip lighting condition features (faster for testing)')
    parser.add_argument('--skip-workzones', action='store_true',
                       help='Skip work zone integration (faster for testing)')

    args = parser.parse_args()

    print(f'\n{"="*80}')
    print('ML TRAINING DATASET BUILDER')
    print(f'{"="*80}')
    print(f'\nTimestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Crash file: {args.crash_file}')
    print(f'AADT file: {args.aadt_file}')
    print(f'Output dir: {args.output_dir}')

    # Load crashes
    crashes_gdf = load_crashes(args.crash_file, cities=args.cities, sample_size=args.sample)

    # Extract road features
    if not args.skip_osmnx:
        crashes_with_roads = extract_road_features_batch(crashes_gdf)
        if crashes_with_roads is None:
            print('\n✗ Failed to extract road features, continuing without them...')
            crashes_with_roads = pd.DataFrame(crashes_gdf)
    else:
        print('\n⚠ Skipping OSMnx road feature extraction')
        crashes_with_roads = pd.DataFrame(crashes_gdf.drop(columns=['geometry'], errors='ignore'))

    # Integrate HPMS road characteristics (speed limits, lanes, IRI)
    if not args.skip_hpms:
        crashes_with_hpms = integrate_hpms_road_characteristics(crashes_with_roads, args.hpms_file)
    else:
        print('\n⚠ Skipping HPMS integration')
        crashes_with_hpms = crashes_with_roads

    # Integrate work zones
    if not args.skip_workzones:
        crashes_with_wz = integrate_work_zones(crashes_with_hpms, args.workzone_file)
    else:
        print('\n⚠ Skipping work zone integration')
        crashes_with_wz = crashes_with_hpms

    # Add lighting conditions (sunrise/sunset based)
    if not args.skip_lighting:
        crashes_with_lighting = add_lighting_conditions(crashes_with_wz)
    else:
        print('\n⚠ Skipping lighting condition features')
        crashes_with_lighting = crashes_with_wz

    # Attach AADT
    crashes_with_aadt = attach_aadt_traffic(crashes_with_lighting, args.aadt_file)

    # Engineer features
    ml_dataset = engineer_features(crashes_with_aadt)

    # Create splits
    train, val, test = create_train_val_test_split(ml_dataset)

    # Save datasets
    train_path, val_path, test_path = save_datasets(train, val, test, args.output_dir)

    # Print summary
    print_dataset_summary(train, val, test)

    print(f'\n✅ Datasets saved:')
    print(f'   {train_path}')
    print(f'   {val_path}')
    print(f'   {test_path}')
    print()


if __name__ == '__main__':
    main()
