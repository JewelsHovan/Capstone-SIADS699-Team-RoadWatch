"""
Integrate Texas AADT Traffic Data with Work Zones

This script performs spatial joins to match work zones with nearby traffic stations
and adds AADT (Annual Average Daily Traffic) as a feature for ML modeling.

Usage:
    python scripts/integrate_texas_aadt.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_texas_data():
    """Load Texas work zones and AADT data"""

    print("\n" + "="*60)
    print("LOADING TEXAS DATA")
    print("="*60)

    # Load work zones
    print("\n1. Loading work zones...")
    wz = pd.read_csv('data/processed/texas_work_zones_analysis.csv')

    # Filter to records with coordinates
    wz_with_coords = wz[wz['latitude'].notna() & wz['longitude'].notna()].copy()

    print(f"   Total records: {len(wz):,}")
    print(f"   Records with coordinates: {len(wz_with_coords):,} ({len(wz_with_coords)/len(wz)*100:.1f}%)")
    print(f"   Unique work zones: {wz_with_coords['base_event_id'].nunique():,}")

    # Convert to GeoDataFrame
    wz_gdf = gpd.GeoDataFrame(
        wz_with_coords,
        geometry=gpd.points_from_xy(wz_with_coords.longitude, wz_with_coords.latitude),
        crs='EPSG:4326'
    )

    # Load AADT data
    print("\n2. Loading AADT traffic data...")
    tx_aadt = gpd.read_file('data/raw/traffic/txdot_aadt_annual.gpkg')

    print(f"   Traffic stations: {len(tx_aadt):,}")
    print(f"   AADT range: {tx_aadt['AADT_RPT_QTY'].min():,.0f} - {tx_aadt['AADT_RPT_QTY'].max():,.0f}")
    print(f"   Mean AADT: {tx_aadt['AADT_RPT_QTY'].mean():,.0f}")

    return wz_gdf, tx_aadt

def spatial_join_with_aadt(wz_gdf, tx_aadt, max_distance_meters=500):
    """
    Perform spatial join to match work zones with nearest traffic stations

    Parameters:
    -----------
    wz_gdf : GeoDataFrame
        Work zones with point geometry
    tx_aadt : GeoDataFrame
        Traffic stations with AADT data
    max_distance_meters : int
        Maximum distance in meters to match (default: 500m)

    Returns:
    --------
    GeoDataFrame with AADT data joined
    """

    print("\n" + "="*60)
    print("SPATIAL JOIN - MATCHING WORK ZONES TO TRAFFIC STATIONS")
    print("="*60)

    print(f"\nParameters:")
    print(f"  Max distance: {max_distance_meters}m")

    # Reproject to meters for accurate distance calculation
    # EPSG:3857 is Web Mercator (meters)
    print(f"\n1. Reprojecting to EPSG:3857 (meters)...")
    wz_meter = wz_gdf.to_crs('EPSG:3857')
    aadt_meter = tx_aadt.to_crs('EPSG:3857')

    # Spatial join nearest
    print(f"2. Finding nearest traffic station for each work zone...")
    wz_with_aadt = gpd.sjoin_nearest(
        wz_meter,
        aadt_meter[['AADT_RPT_QTY', 'CNTY_NM', 'DIST_NM', 'TRFC_STATN_ID', 'CATEGORY', 'AADT_RPT_YEAR', 'geometry']],
        how='left',
        max_distance=max_distance_meters,
        distance_col='distance_to_station_m'
    )

    # Convert back to WGS84
    wz_with_aadt = wz_with_aadt.to_crs('EPSG:4326')

    # Calculate match statistics
    matched = wz_with_aadt['AADT_RPT_QTY'].notna().sum()
    match_rate = matched / len(wz_with_aadt) * 100

    print(f"\n{'='*60}")
    print("MATCH RESULTS")
    print(f"{'='*60}")
    print(f"Total work zones: {len(wz_with_aadt):,}")
    print(f"Matched to traffic station: {matched:,} ({match_rate:.1f}%)")
    print(f"Not matched: {len(wz_with_aadt) - matched:,} ({100-match_rate:.1f}%)")

    if matched > 0:
        print(f"\nDistance statistics (for matched records):")
        distances = wz_with_aadt[wz_with_aadt['distance_to_station_m'].notna()]['distance_to_station_m']
        print(f"  Mean distance: {distances.mean():.0f}m")
        print(f"  Median distance: {distances.median():.0f}m")
        print(f"  Max distance: {distances.max():.0f}m")

        print(f"\nAADT statistics (for matched records):")
        aadt = wz_with_aadt[wz_with_aadt['AADT_RPT_QTY'].notna()]['AADT_RPT_QTY']
        print(f"  Mean AADT: {aadt.mean():,.0f} vehicles/day")
        print(f"  Median AADT: {aadt.median():,.0f} vehicles/day")
        print(f"  Range: {aadt.min():,.0f} - {aadt.max():,.0f}")

    return wz_with_aadt

def handle_missing_aadt(wz_with_aadt):
    """
    Handle missing AADT values using fallback strategies

    Strategy:
    1. Use matched AADT when available
    2. Fall back to county average for unmatched
    3. Fall back to statewide median if county avg unavailable
    """

    print("\n" + "="*60)
    print("HANDLING MISSING AADT VALUES")
    print("="*60)

    # Calculate county averages from matched records
    county_avg = wz_with_aadt[wz_with_aadt['AADT_RPT_QTY'].notna()].groupby('CNTY_NM')['AADT_RPT_QTY'].mean()

    print(f"\nCounty averages calculated from matched records:")
    print(f"  Counties with data: {len(county_avg)}")
    if len(county_avg) > 0:
        print(f"  County AADT range: {county_avg.min():,.0f} - {county_avg.max():,.0f}")

    # Calculate statewide median
    state_median = wz_with_aadt[wz_with_aadt['AADT_RPT_QTY'].notna()]['AADT_RPT_QTY'].median()
    print(f"  Statewide median: {state_median:,.0f}")

    # Create filled AADT column
    def fill_aadt(row):
        # If we have matched AADT, use it
        if pd.notna(row['AADT_RPT_QTY']):
            return row['AADT_RPT_QTY'], 'matched'

        # Try county average
        if pd.notna(row.get('CNTY_NM')) and row.get('CNTY_NM') in county_avg:
            return county_avg[row['CNTY_NM']], 'county_avg'

        # Fall back to state median
        return state_median, 'state_median'

    wz_with_aadt[['aadt_filled', 'aadt_source']] = wz_with_aadt.apply(
        lambda row: pd.Series(fill_aadt(row)),
        axis=1
    )

    print(f"\n{'='*60}")
    print("AADT FILL SUMMARY")
    print(f"{'='*60}")
    print(wz_with_aadt['aadt_source'].value_counts())

    return wz_with_aadt

def calculate_crash_rate_features(wz_with_aadt):
    """
    Calculate crash rate and exposure-based features

    Note: This is a placeholder - you'll need actual crash counts
    """

    print("\n" + "="*60)
    print("CALCULATING EXPOSURE-BASED FEATURES")
    print("="*60)

    # Placeholder for crash count (you'll add this when you get crash data)
    wz_with_aadt['crash_count'] = 0  # REPLACE with actual crash data

    # Vehicle-Miles of Travel (VMT) estimation
    # VMT = AADT × duration_days × segment_length_miles
    # For now, assume 0.5 mile average segment length (adjust based on your data)

    wz_with_aadt['estimated_segment_miles'] = 0.5  # REPLACE with actual segment length

    wz_with_aadt['vehicle_miles_traveled'] = (
        wz_with_aadt['aadt_filled'] *
        wz_with_aadt['duration_days'] *
        wz_with_aadt['estimated_segment_miles']
    )

    # Crash rate per million vehicle-miles (standard DOT metric)
    # Only calculate where we have crash data
    wz_with_aadt['crash_rate_per_mvmt'] = np.where(
        wz_with_aadt['vehicle_miles_traveled'] > 0,
        (wz_with_aadt['crash_count'] / wz_with_aadt['vehicle_miles_traveled']) * 1_000_000,
        np.nan
    )

    print("\nVehicle-Miles Traveled (VMT):")
    print(f"  Mean VMT: {wz_with_aadt['vehicle_miles_traveled'].mean():,.0f}")
    print(f"  Median VMT: {wz_with_aadt['vehicle_miles_traveled'].median():,.0f}")

    print("\nNote: Crash rate calculation is placeholder - add actual crash data!")

    return wz_with_aadt

def create_traffic_risk_features(wz_with_aadt):
    """
    Create additional traffic-based features for ML model
    """

    print("\n" + "="*60)
    print("ENGINEERING TRAFFIC FEATURES")
    print("="*60)

    # Traffic volume categories
    wz_with_aadt['traffic_volume_category'] = pd.cut(
        wz_with_aadt['aadt_filled'],
        bins=[0, 1000, 5000, 15000, 30000, 100000],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    print("\nTraffic Volume Categories:")
    print(wz_with_aadt['traffic_volume_category'].value_counts().sort_index())

    # Exposure-weighted duration (higher traffic × longer duration = higher exposure)
    wz_with_aadt['exposure_score'] = (
        wz_with_aadt['aadt_filled'] / 10000 *  # Normalize AADT
        np.log1p(wz_with_aadt['duration_days'])  # Log-scale duration
    )

    # Lane closure risk (some lanes closed on high traffic road = high risk)
    wz_with_aadt['lane_closure_risk'] = np.where(
        wz_with_aadt['vehicle_impact'] == 'some-lanes-closed',
        wz_with_aadt['aadt_filled'] / wz_with_aadt['total_num_lanes'],
        0
    )

    print("\nCreated features:")
    print("  - traffic_volume_category (very_low/low/medium/high/very_high)")
    print("  - exposure_score (AADT × log(duration))")
    print("  - lane_closure_risk (AADT per open lane)")

    return wz_with_aadt

def save_enriched_data(wz_with_aadt):
    """Save the enriched dataset"""

    print("\n" + "="*60)
    print("SAVING ENRICHED DATA")
    print("="*60)

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV (drop geometry for easier use)
    output_csv = os.path.join(output_dir, 'texas_work_zones_with_aadt.csv')

    # Select key columns
    columns_to_save = [
        'road_event_id', 'base_event_id', 'road_name', 'direction',
        'start_date_parsed', 'end_date_parsed', 'duration_days',
        'latitude', 'longitude',
        'total_num_lanes', 'vehicle_impact',
        'AADT_RPT_QTY', 'aadt_filled', 'aadt_source',
        'distance_to_station_m',
        'CNTY_NM', 'DIST_NM', 'TRFC_STATN_ID', 'CATEGORY', 'AADT_RPT_YEAR',
        'vehicle_miles_traveled', 'crash_rate_per_mvmt',
        'traffic_volume_category', 'exposure_score', 'lane_closure_risk'
    ]

    # Only keep columns that exist
    columns_to_save = [col for col in columns_to_save if col in wz_with_aadt.columns]

    df_to_save = wz_with_aadt[columns_to_save].copy()

    # Convert datetime columns to string for CSV
    for col in df_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
            df_to_save[col] = df_to_save[col].astype(str)

    df_to_save.to_csv(output_csv, index=False)

    file_size_mb = os.path.getsize(output_csv) / 1024 / 1024
    print(f"\n✓ Saved to: {output_csv}")
    print(f"  Records: {len(df_to_save):,}")
    print(f"  Columns: {len(df_to_save.columns)}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Also save as GeoPackage (with geometry)
    output_gpkg = os.path.join(output_dir, 'texas_work_zones_with_aadt.gpkg')
    wz_with_aadt.to_file(output_gpkg, driver='GPKG')
    print(f"\n✓ Saved to: {output_gpkg}")

    return output_csv, output_gpkg

def print_next_steps():
    """Print usage instructions"""

    print("\n" + "="*60)
    print("NEXT STEPS - HOW TO USE THIS DATA")
    print("="*60)

    print("\n1. Load the enriched data:")
    print("   import pandas as pd")
    print("   tx_wz = pd.read_csv('data/processed/texas_work_zones_with_aadt.csv')")

    print("\n2. Use AADT features in ML model:")
    print("   feature_cols = [")
    print("       'aadt_filled',              # Traffic volume")
    print("       'duration_days',            # Work zone duration")
    print("       'total_num_lanes',          # Road size")
    print("       'exposure_score',           # Combined AADT × duration")
    print("       'lane_closure_risk',        # AADT per lane")
    print("       # Add more features...")
    print("   ]")
    print("   X = tx_wz[feature_cols]")
    print("   y = tx_wz['crash_count']  # Target variable (add crash data!)")

    print("\n3. Add crash data:")
    print("   - Get Texas crash data from TIMS or TxDOT")
    print("   - Spatial join crashes to work zones")
    print("   - Count crashes per work zone")
    print("   - Then recalculate crash rates!")

    print("\n4. Train baseline model:")
    print("   from sklearn.ensemble import GradientBoostingRegressor")
    print("   model = GradientBoostingRegressor()")
    print("   model.fit(X_train, y_train)")
    print("   predictions = model.predict(X_test)")

    print("\n5. Compare with/without AADT:")
    print("   - Train model WITHOUT AADT features")
    print("   - Train model WITH AADT features")
    print("   - Compare AUC-ROC / R² scores")
    print("   - AADT should significantly improve model!")

    print("\n" + "="*60)

def main():
    """Main execution"""

    print("\n" + "="*60)
    print("TEXAS WORK ZONES + AADT INTEGRATION")
    print("="*60)

    # Load data
    wz_gdf, tx_aadt = load_texas_data()

    # Spatial join
    wz_with_aadt = spatial_join_with_aadt(wz_gdf, tx_aadt, max_distance_meters=500)

    # Handle missing values
    wz_with_aadt = handle_missing_aadt(wz_with_aadt)

    # Calculate crash rate features (placeholder - add crash data!)
    wz_with_aadt = calculate_crash_rate_features(wz_with_aadt)

    # Create traffic risk features
    wz_with_aadt = create_traffic_risk_features(wz_with_aadt)

    # Save results
    csv_path, gpkg_path = save_enriched_data(wz_with_aadt)

    # Print usage instructions
    print_next_steps()

    print("\n" + "="*60)
    print("INTEGRATION COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
