"""
Download TxDOT AADT Annuals (Public View) - Mainline Traffic Data
This downloads traffic counts for ALL roadways (not just ramps!)

Source: https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/TxDOT_AADT_Annuals_(Public_View)/FeatureServer/0

Dataset Info:
- 41,467+ traffic count stations
- Categories: ANNUAL ACR (annual counts), TOLL (toll roads)
- Years: Historical data + current (2024)
- Includes AADT for mainline highways, not just ramps!

Usage:
    python scripts/download_txdot_aadt_annual.py
"""

import requests
import geopandas as gpd
import json
import os
from pathlib import Path

from config.paths import TEXAS_BRONZE_TRAFFIC

def download_txdot_aadt_annual(output_dir=None):
    """
    Download TxDOT AADT Annual traffic data

    Returns:
        geopandas.GeoDataFrame with traffic count data
    """

    if output_dir is None:
        output_dir = TEXAS_BRONZE_TRAFFIC
    output_dir = Path(output_dir)

    print("\n" + "="*60)
    print("TXDOT AADT ANNUAL TRAFFIC DATA DOWNLOADER")
    print("="*60)

    # TxDOT AADT Annuals Feature Server URL
    base_url = "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/TxDOT_AADT_Annuals_(Public_View)/FeatureServer/0/query"

    print(f"\nSource: TxDOT AADT Annuals (Public View)")
    print("Fetching all annual count records...")

    # Query parameters to get all records
    params = {
        'where': '1=1',  # Get all records
        'outFields': '*',  # Get all fields
        'f': 'geojson',  # Return as GeoJSON
        'resultRecordCount': 2000,  # Max per request
        'resultOffset': 0
    }

    all_features = []
    offset = 0
    batch_num = 1

    while True:
        params['resultOffset'] = offset

        try:
            print(f"\nFetching batch {batch_num} (offset: {offset})...", end='')

            response = requests.get(base_url, params=params, timeout=120)

            if response.status_code != 200:
                print(f"\n✗ Error: HTTP {response.status_code}")
                break

            # Parse GeoJSON
            data = response.json()

            if 'features' not in data:
                print(f"\n✗ No features in response")
                break

            features = data['features']

            if len(features) == 0:
                print(f"\n✓ Complete (no more features)")
                break

            all_features.extend(features)
            print(f" ✓ Got {len(features)} features (Total: {len(all_features):,})")

            # If we got fewer features than requested, we're done
            if len(features) < params['resultRecordCount']:
                print(f"✓ Complete (last batch)")
                break

            offset += params['resultRecordCount']
            batch_num += 1

        except Exception as e:
            print(f"\n✗ Error: {e}")
            if len(all_features) > 0:
                print(f"Continuing with {len(all_features):,} features downloaded so far...")
            break

    if len(all_features) == 0:
        print("\n✗ No features downloaded!")
        return None

    print(f"\n{'='*60}")
    print(f"Processing {len(all_features):,} features...")

    # Convert to GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')

    except Exception as e:
        print(f"✗ Error creating GeoDataFrame: {e}")
        return None

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'txdot_aadt_annual.gpkg')

    try:
        gdf.to_file(output_file, driver='GPKG')
        print(f"✓ Saved to: {output_file}")
        file_size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"  File size: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return None

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(gdf):,}")

    # Category breakdown
    if 'CATEGORY' in gdf.columns:
        print(f"\nCategory breakdown:")
        print(gdf['CATEGORY'].value_counts())

    # Year statistics
    if 'AADT_RPT_YEAR' in gdf.columns:
        print(f"\nYear range:")
        print(f"  Min: {gdf['AADT_RPT_YEAR'].min()}")
        print(f"  Max: {gdf['AADT_RPT_YEAR'].max()}")
        print(f"\nRecords by year:")
        print(gdf['AADT_RPT_YEAR'].value_counts().sort_index().tail(5))

    # AADT statistics
    if 'AADT_RPT_QTY' in gdf.columns:
        aadt = gdf[gdf['AADT_RPT_QTY'] > 0]['AADT_RPT_QTY']
        if len(aadt) > 0:
            print(f"\nAADT Statistics (current year):")
            print(f"  Records with AADT: {len(aadt):,} ({len(aadt)/len(gdf)*100:.1f}%)")
            print(f"  Range: {aadt.min():,.0f} - {aadt.max():,.0f}")
            print(f"  Mean: {aadt.mean():,.0f}")
            print(f"  Median: {aadt.median():,.0f}")

    # County coverage
    if 'CNTY_NM' in gdf.columns:
        counties = gdf['CNTY_NM'].nunique()
        print(f"\nCounty coverage: {counties} counties")
        print(f"\nTop 10 counties by station count:")
        print(gdf['CNTY_NM'].value_counts().head(10))

    # Check geometry
    print(f"\nGeometry:")
    print(f"  Type: {gdf.geometry.type.unique()}")
    print(f"  Valid: {gdf.geometry.is_valid.sum():,} / {len(gdf):,}")

    # Geographic extent
    bounds = gdf.total_bounds
    print(f"  Extent: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")

    # Key fields
    print(f"\nKey Fields Available:")
    key_fields = ['DIST_NM', 'CNTY_NM', 'TRFC_STATN_ID', 'CATEGORY',
                  'AADT_RPT_YEAR', 'AADT_RPT_QTY', 'COUNT_CYCLE']
    for field in key_fields:
        if field in gdf.columns:
            print(f"  ✓ {field}")

    return gdf

def main():
    """Main execution"""

    # Download the data
    gdf = download_txdot_aadt_annual()

    if gdf is not None:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Load the data:")
        print("   import geopandas as gpd")
        print(f"   tx_aadt = gpd.read_file('{TEXAS_BRONZE_TRAFFIC / 'txdot_aadt_annual.gpkg'}')")
        print("\n2. Filter to most recent year:")
        print("   latest = tx_aadt[tx_aadt['AADT_RPT_YEAR'] == tx_aadt['AADT_RPT_YEAR'].max()]")
        print("\n3. Spatial join with work zones:")
        print("   work_zones_with_traffic = gpd.sjoin_nearest(work_zones_gdf, latest)")
        print("\n4. Use as ML feature:")
        print("   features_df['aadt'] = work_zones_with_traffic['AADT_RPT_QTY']")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("DOWNLOAD FAILED")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify URL is still active:")
        print("   https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/")
        print("   TxDOT_AADT_Annuals_(Public_View)/FeatureServer/0")
        print("3. Try manual download via browser:")
        print("   Add '?outFields=*&where=1=1&f=geojson' to the URL")
        print("="*60)

if __name__ == "__main__":
    main()
