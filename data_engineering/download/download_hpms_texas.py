"""
Download HPMS (Highway Performance Monitoring System) data for Texas

This script provides instructions and automation for downloading comprehensive road characteristics:
- Speed limits (SPEED_LIMIT)
- Lane counts (THROUGH_LANES)
- Pavement condition (IRI - International Roughness Index)
- Shoulder width/type
- Median type
- Functional classification
- AADT (Annual Average Daily Traffic)

HPMS Data Sources:
1. FHWA Official: https://www.fhwa.dot.gov/policyinformation/hpms.cfm
2. BTS GeoData: https://geodata.bts.gov/search?q=hpms
3. Data.gov: https://catalog.data.gov/dataset/highway-performance-monitoring-system-hpms

RECOMMENDED APPROACH:
Due to API limitations, we recommend manual download from BTS GeoData:
1. Visit: https://geodata.bts.gov/
2. Search: "HPMS Texas" or "Highway Performance Monitoring System"
3. Download shapefile or GeoPackage
4. Place in: data/raw/texas/roadway_characteristics/

This script will then process the downloaded file.

Usage:
    # After manual download:
    python data_engineering/download/download_hpms_texas.py --file path/to/downloaded/hpms.shp

    # Or let script guide you through manual download:
    python data_engineering/download/download_hpms_texas.py
"""

import geopandas as gpd
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import TEXAS_RAW, ensure_directories


def process_hpms_file(input_file, output_dir=None, year=2023):
    """
    Process downloaded HPMS file

    Args:
        input_file: Path to downloaded HPMS shapefile/geopackage
        output_dir: Directory to save processed data
        year: Year of data (for filename)

    Returns:
        geopandas.GeoDataFrame with HPMS road characteristics
    """

    if output_dir is None:
        output_dir = TEXAS_RAW / "roadway_characteristics"

    print("\n" + "="*80)
    print("HPMS TEXAS DATA PROCESSOR")
    print("="*80)

    print(f"\nReading HPMS data from: {input_file}")

    try:
        # Check if it's a geodatabase with multiple layers
        if str(input_file).endswith('.gdb'):
            # Always load the known Texas layer
            layer_name = 'HPMS_FULL_TX_2023'
            print(f"Loading Texas layer: {layer_name}")
            gdf = gpd.read_file(input_file, layer=layer_name)
        else:
            gdf = gpd.read_file(input_file)

        print(f"✓ Loaded {len(gdf):,} road segments")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

    # Filter to Texas if needed (in case file contains multiple states)
    if 'STATE_CODE' in gdf.columns:
        texas_gdf = gdf[gdf['STATE_CODE'] == 48].copy()
        if len(texas_gdf) < len(gdf):
            print(f"✓ Filtered to Texas: {len(texas_gdf):,} segments (was {len(gdf):,})")
            gdf = texas_gdf
    elif 'State_Code' in gdf.columns:
        texas_gdf = gdf[gdf['State_Code'] == 48].copy()
        if len(texas_gdf) < len(gdf):
            print(f"✓ Filtered to Texas: {len(texas_gdf):,} segments (was {len(gdf):,})")
            gdf = texas_gdf

    # Normalize column names to lowercase for consistency
    gdf.columns = gdf.columns.str.lower()

    # Save processed data
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'hpms_texas_{year}.gpkg'

    try:
        gdf.to_file(output_file, driver='GPKG')
        print(f"✓ Saved to: {output_file}")
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  File size: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return None

    # Print summary
    print_hpms_summary(gdf)

    return gdf


def print_hpms_summary(gdf):
    """Print summary statistics for HPMS data"""

    print(f"\n{'='*80}")
    print(f"HPMS TEXAS DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total road segments: {len(gdf):,}")

    # Key fields for crash prediction (normalized to lowercase)
    key_fields = {
        'speed_limit': 'Speed Limit',
        'through_lanes': 'Lane Count',
        'iri': 'IRI (Pavement Condition)',
        'aadt': 'AADT (Traffic Volume)',
        'f_system': 'Functional Class',
        'urban_code': 'Urban/Rural',
        'nhs': 'National Highway System',
        'route_id': 'Route ID'
    }

    print(f"\nKey Fields Available:")
    found_fields = []
    for field, description in key_fields.items():
        if field in gdf.columns:
            found_fields.append(field)
            non_null = gdf[field].notna().sum()
            pct = non_null / len(gdf) * 100
            print(f"  ✓ {description:30s}: {non_null:6,} / {len(gdf):6,} ({pct:5.1f}%)")

            # Statistics for numeric fields
            if field in ['speed_limit', 'through_lanes', 'iri', 'aadt']:
                values = gdf[field].dropna()
                if len(values) > 0 and pd.api.types.is_numeric_dtype(values):
                    print(f"      Range: {values.min():.1f} - {values.max():.1f}")
                    print(f"      Mean: {values.mean():.1f}, Median: {values.median():.1f}")
        else:
            print(f"  ✗ {description:30s}: Not found")

    # Print all available columns
    print(f"\nAll Available Columns ({len(gdf.columns)}):")
    for i, col in enumerate(sorted(gdf.columns), 1):
        if col != 'geometry':
            print(f"  {i:2d}. {col}")

    # Geometry
    print(f"\nGeometry:")
    print(f"  Type: {gdf.geometry.type.unique()}")
    print(f"  Valid: {gdf.geometry.is_valid.sum():,} / {len(gdf):,}")
    bounds = gdf.total_bounds
    print(f"  Extent: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")

    return found_fields


def print_download_instructions():
    """Print manual download instructions"""

    print("\n" + "="*80)
    print("HPMS DATA DOWNLOAD INSTRUCTIONS")
    print("="*80)

    print("\nThe HPMS API endpoints have changed. Please download manually:")

    print("\n" + "─"*80)
    print("OPTION 1: BTS GeoData Portal (Recommended)")
    print("─"*80)
    print("\n1. Visit: https://geodata.bts.gov/")
    print("2. Search: 'HPMS' or 'Highway Performance Monitoring System'")
    print("3. Filter to 2023 (latest available)")
    print("4. Click on Texas or National dataset")
    print("5. Download as: Shapefile or GeoPackage")
    print("6. Save to: data/raw/texas/roadway_characteristics/")

    print("\n" + "─"*80)
    print("OPTION 2: FHWA Official Source")
    print("─"*80)
    print("\n1. Visit: https://www.fhwa.dot.gov/policyinformation/hpms.cfm")
    print("2. Click 'HPMS Public Release'")
    print("3. Download Texas shapefile")
    print("4. Extract and save to: data/raw/texas/roadway_characteristics/")

    print("\n" + "─"*80)
    print("OPTION 3: Data.gov")
    print("─"*80)
    print("\n1. Visit: https://catalog.data.gov/dataset/highway-performance-monitoring-system-hpms")
    print("2. Download Texas subset or full national dataset")
    print("3. Save to: data/raw/texas/roadway_characteristics/")

    print("\n" + "="*80)
    print("After download, process the file:")
    print("="*80)
    print("\npython data_engineering/download/download_hpms_texas.py --file path/to/hpms.shp")


def main():
    """Main execution"""

    parser = argparse.ArgumentParser(description='Download and process HPMS Texas data')
    parser.add_argument('--file', type=str, help='Path to downloaded HPMS file (shapefile/geopackage)')
    parser.add_argument('--year', type=int, default=2023, help='Year of HPMS data (default: 2023)')

    args = parser.parse_args()

    if args.file:
        # Process provided file
        input_file = Path(args.file)

        if not input_file.exists():
            print(f"✗ Error: File not found: {input_file}")
            sys.exit(1)

        gdf = process_hpms_file(input_file, year=args.year)

        if gdf is not None:
            print("\n" + "="*80)
            print("PROCESSING COMPLETE!")
            print("="*80)
            print("\nNext steps:")
            print("1. Load the data:")
            print("   import geopandas as gpd")
            print("   from config.paths import TEXAS_RAW")
            print(f"   hpms = gpd.read_file(TEXAS_RAW / 'roadway_characteristics' / 'hpms_texas_{args.year}.gpkg')")
            print("\n2. Spatial join with crashes:")
            print("   crashes_with_road = crashes_gdf.sjoin_nearest(hpms, max_distance=50)")
            print("\n3. Fill missing crash attributes from HPMS:")
            print("   crashes['Speed_Limit'].fillna(crashes['speed_limit'], inplace=True)")
            print("   crashes['Number_of_Lanes'].fillna(crashes['through_lanes'], inplace=True)")
    else:
        # No file provided, print instructions
        print_download_instructions()


if __name__ == "__main__":
    main()
