#!/usr/bin/env python3
"""
Match Texas crash data to work zones spatially

This script:
1. Loads crash data (Austin) with lat/lon
2. Loads Texas work zone data
3. Performs spatial matching with configurable buffer distance
4. Creates integrated dataset with crashes matched to work zones

Usage:
    python match_crashes_to_workzones.py
    python match_crashes_to_workzones.py --buffer 1000  # 1km buffer
    python match_crashes_to_workzones.py --construction-only  # Only crashes marked as construction
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import argparse
import sys

# File paths
CRASH_FILE = Path("data/raw/crashes/austin_crashes_latest.csv")
WORKZONE_FILE = Path("data/processed/texas_work_zones_with_aadt.csv")
OUTPUT_DIR = Path("data/processed")

def load_crashes(filepath, construction_only=False):
    """Load crash data and convert to GeoDataFrame"""
    print(f"📂 Loading crash data from {filepath}...")

    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} crashes")

    # Filter for construction zones if requested
    if construction_only:
        if 'road_constr_zone_fl' in df.columns:
            df = df[df['road_constr_zone_fl'] == True]
            print(f"   Filtered to {len(df):,} construction zone crashes")
        else:
            print("   ⚠️  No construction zone flag found in data")

    # Convert to numeric and drop missing coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    before_drop = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    dropped = before_drop - len(df)

    if dropped > 0:
        print(f"   ⚠️  Dropped {dropped:,} crashes with missing coordinates")

    print(f"   Final crash count: {len(df):,}")

    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    print(f"   ✅ Created GeoDataFrame with {len(gdf):,} points")

    return gdf

def load_workzones(filepath):
    """Load work zone data and convert to GeoDataFrame"""
    print(f"\n📂 Loading work zone data from {filepath}...")

    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} work zones")

    # Convert to numeric and drop missing coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    before_drop = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    dropped = before_drop - len(df)

    if dropped > 0:
        print(f"   ⚠️  Dropped {dropped:,} work zones with missing coordinates")

    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    print(f"   ✅ Created GeoDataFrame with {len(gdf):,} points")

    return gdf

def spatial_join_crashes_to_workzones(crashes_gdf, workzones_gdf, buffer_meters=500):
    """
    Match crashes to work zones using spatial buffer

    Args:
        crashes_gdf: GeoDataFrame of crashes
        workzones_gdf: GeoDataFrame of work zones
        buffer_meters: Buffer distance in meters

    Returns:
        GeoDataFrame with matched crashes
    """
    print(f"\n🔍 Spatial matching with {buffer_meters}m buffer...")

    # Convert to projected CRS (meters) for buffering
    print("   Converting to UTM projection...")
    crashes_utm = crashes_gdf.to_crs('EPSG:3857')  # Web Mercator (meters)
    workzones_utm = workzones_gdf.to_crs('EPSG:3857')

    # Buffer work zones
    print(f"   Creating {buffer_meters}m buffer around work zones...")
    workzones_buffered = workzones_utm.copy()
    workzones_buffered['geometry'] = workzones_buffered.geometry.buffer(buffer_meters)

    # Spatial join
    print("   Performing spatial join...")
    matched = gpd.sjoin(
        crashes_utm,
        workzones_buffered,
        how='inner',
        predicate='intersects'
    )

    print(f"   ✅ Matched {len(matched):,} crashes to work zones")

    # Calculate distance to work zone center
    if len(matched) > 0:
        print("   Calculating distances...")
        distances = []
        for idx, row in matched.iterrows():
            crash_geom = crashes_utm.loc[idx, 'geometry']
            wz_idx = row['index_right']
            wz_geom = workzones_utm.loc[wz_idx, 'geometry']
            dist = crash_geom.distance(wz_geom)
            distances.append(dist)
        matched['distance_to_wz_m'] = distances

    # Convert back to WGS84
    matched = matched.to_crs('EPSG:4326')

    return matched

def create_summary_stats(matched_gdf):
    """Generate summary statistics"""
    print("\n" + "="*70)
    print("📊 MATCHING SUMMARY")
    print("="*70)

    # Overall stats
    unique_crashes = matched_gdf.index.nunique()
    unique_workzones = matched_gdf['road_event_id'].nunique() if 'road_event_id' in matched_gdf.columns else matched_gdf['index_right'].nunique()

    print(f"\nTotal crash-workzone matches: {len(matched_gdf):,}")
    print(f"Unique crashes matched: {unique_crashes:,}")
    print(f"Unique work zones with crashes: {unique_workzones:,}")

    if unique_crashes > 0:
        avg_matches_per_crash = len(matched_gdf) / unique_crashes
        print(f"Average matches per crash: {avg_matches_per_crash:.2f}")

    # Distance statistics
    if 'distance_to_wz_m' in matched_gdf.columns:
        print(f"\nDistance to work zone (meters):")
        print(f"  Mean: {matched_gdf['distance_to_wz_m'].mean():.1f}m")
        print(f"  Median: {matched_gdf['distance_to_wz_m'].median():.1f}m")
        print(f"  Max: {matched_gdf['distance_to_wz_m'].max():.1f}m")

    # Severity distribution
    if 'crash_sev_id' in matched_gdf.columns:
        print(f"\nSeverity distribution of matched crashes:")
        severity_map = {
            0: 'Unknown',
            1: 'Incapacitating Injury',
            2: 'Non-Incapacitating Injury',
            3: 'Possible Injury',
            4: 'Killed',
            5: 'Not Injured'
        }
        for sev_id, count in matched_gdf['crash_sev_id'].value_counts().sort_index().items():
            sev_name = severity_map.get(int(sev_id), f'Code {sev_id}')
            print(f"  {sev_name}: {count:,}")

    # Deaths and injuries
    if 'death_cnt' in matched_gdf.columns:
        total_deaths = pd.to_numeric(matched_gdf['death_cnt'], errors='coerce').sum()
        print(f"\nTotal deaths in matched crashes: {int(total_deaths):,}")

    if 'tot_injry_cnt' in matched_gdf.columns:
        total_injuries = pd.to_numeric(matched_gdf['tot_injry_cnt'], errors='coerce').sum()
        print(f"Total injuries in matched crashes: {int(total_injuries):,}")

    # Work zone characteristics
    if 'vehicle_impact' in matched_gdf.columns:
        print(f"\nWork zone vehicle impact distribution:")
        for impact, count in matched_gdf['vehicle_impact'].value_counts().items():
            print(f"  {impact}: {count:,}")

    print("\n" + "="*70)

def aggregate_crashes_per_workzone(matched_gdf):
    """Aggregate crash statistics per work zone"""
    print("\n📊 Aggregating crashes per work zone...")

    # Determine work zone ID column
    wz_id_col = 'road_event_id' if 'road_event_id' in matched_gdf.columns else 'index_right'

    # Group by work zone and aggregate
    agg_dict = {
        'cris_crash_id': 'count',  # Number of crashes
    }

    # Add severity counts if available
    if 'crash_sev_id' in matched_gdf.columns:
        for sev_id, sev_name in [(1, 'incap_injury'), (2, 'non_incap_injury'),
                                  (3, 'possible_injury'), (4, 'fatal'), (5, 'no_injury')]:
            matched_gdf[f'{sev_name}_count'] = (matched_gdf['crash_sev_id'] == sev_id).astype(int)
            agg_dict[f'{sev_name}_count'] = 'sum'

    # Add death/injury counts if available
    if 'death_cnt' in matched_gdf.columns:
        matched_gdf['death_cnt_num'] = pd.to_numeric(matched_gdf['death_cnt'], errors='coerce').fillna(0)
        agg_dict['death_cnt_num'] = 'sum'

    if 'tot_injry_cnt' in matched_gdf.columns:
        matched_gdf['tot_injry_cnt_num'] = pd.to_numeric(matched_gdf['tot_injry_cnt'], errors='coerce').fillna(0)
        agg_dict['tot_injry_cnt_num'] = 'sum'

    # Distance stats
    if 'distance_to_wz_m' in matched_gdf.columns:
        agg_dict['distance_to_wz_m'] = ['min', 'mean', 'max']

    # Aggregate
    crash_stats = matched_gdf.groupby(wz_id_col).agg(agg_dict)

    # Flatten multi-level columns
    crash_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                           for col in crash_stats.columns]

    # Rename count column
    crash_stats = crash_stats.rename(columns={'cris_crash_id': 'crash_count'})

    print(f"   ✅ Aggregated {len(crash_stats):,} work zones with crashes")

    return crash_stats

def save_results(matched_gdf, crash_stats, buffer_meters):
    """Save matched data and aggregated statistics"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full matched data
    output_file = OUTPUT_DIR / f"texas_crashes_matched_to_workzones_{buffer_meters}m.csv"
    matched_gdf.to_csv(output_file, index=False)
    print(f"\n💾 Saved matched crashes to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Save aggregated stats
    stats_file = OUTPUT_DIR / f"texas_workzones_crash_stats_{buffer_meters}m.csv"
    crash_stats.to_csv(stats_file)
    print(f"\n💾 Saved crash statistics to: {stats_file}")
    print(f"   File size: {stats_file.stat().st_size / 1024:.1f} KB")

    # Save as GeoPackage for GIS
    gpkg_file = OUTPUT_DIR / f"texas_crashes_matched_to_workzones_{buffer_meters}m.gpkg"
    matched_gdf.to_file(gpkg_file, driver='GPKG')
    print(f"\n💾 Saved GeoPackage to: {gpkg_file}")
    print(f"   File size: {gpkg_file.stat().st_size / 1024 / 1024:.1f} MB")

    return output_file, stats_file, gpkg_file

def main():
    parser = argparse.ArgumentParser(
        description="Match Texas crash data to work zones spatially",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--buffer', type=int, default=500,
                       help='Buffer distance in meters (default: 500)')
    parser.add_argument('--construction-only', action='store_true',
                       help='Only match crashes marked as construction zones')
    parser.add_argument('--crash-file', type=str,
                       help='Override default crash file path')
    parser.add_argument('--workzone-file', type=str,
                       help='Override default work zone file path')

    args = parser.parse_args()

    # Determine file paths
    crash_file = Path(args.crash_file) if args.crash_file else CRASH_FILE
    workzone_file = Path(args.workzone_file) if args.workzone_file else WORKZONE_FILE

    # Check files exist
    if not crash_file.exists():
        print(f"❌ Crash file not found: {crash_file}")
        print("\nRun this first:")
        print("  python scripts/download_austin_crashes.py --all")
        sys.exit(1)

    if not workzone_file.exists():
        print(f"❌ Work zone file not found: {workzone_file}")
        sys.exit(1)

    print("🚀 Texas Crash-Workzone Matcher")
    print("="*70 + "\n")

    # Load data
    crashes_gdf = load_crashes(crash_file, args.construction_only)
    workzones_gdf = load_workzones(workzone_file)

    # Spatial matching
    matched_gdf = spatial_join_crashes_to_workzones(
        crashes_gdf,
        workzones_gdf,
        buffer_meters=args.buffer
    )

    # Summary statistics
    create_summary_stats(matched_gdf)

    # Aggregate per work zone
    crash_stats = aggregate_crashes_per_workzone(matched_gdf)

    # Save results
    save_results(matched_gdf, crash_stats, args.buffer)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
