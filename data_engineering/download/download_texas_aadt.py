"""
Download Texas DOT AADT (Traffic Volume) Data
Uses TxDOT ArcGIS REST API to fetch traffic count data

Source: https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/TxDOT_Ramp_AADT_Traffic_Counts/FeatureServer/0
"""

import requests
import geopandas as gpd
import json
import os
from pathlib import Path

from config.paths import TEXAS_BRONZE_TRAFFIC

def download_texas_aadt(output_dir=None):
    """
    Download Texas AADT traffic data from TxDOT ArcGIS REST API

    Returns:
        geopandas.GeoDataFrame with traffic count data
    """

    print("\n" + "="*60)
    print("TEXAS AADT TRAFFIC DATA DOWNLOADER")
    print("="*60)

    if output_dir is None:
        output_dir = TEXAS_BRONZE_TRAFFIC
    output_dir = Path(output_dir)

    # TxDOT AADT Feature Server URL
    base_url = "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/TxDOT_Ramp_AADT_Traffic_Counts/FeatureServer/0/query"

    print(f"\nSource: {base_url}")
    print("Fetching all records...")

    # Query parameters to get all records
    params = {
        'where': '1=1',  # Get all records
        'outFields': '*',  # Get all fields
        'f': 'geojson',  # Return as GeoJSON
        'resultRecordCount': 10000,  # Max per request (adjust if needed)
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
        # GeoJSON structure: features have properties and geometry
        gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')

    except Exception as e:
        print(f"✗ Error creating GeoDataFrame: {e}")
        return None

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'texas_aadt_traffic_counts.gpkg')

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
    print(f"\nColumns available:")
    for col in gdf.columns:
        if col != 'geometry':
            print(f"  - {col}")

    # Look for AADT-related fields
    print(f"\nAADT-related fields:")
    aadt_cols = [col for col in gdf.columns if 'AADT' in col.upper() or 'TRAFFIC' in col.upper() or 'COUNT' in col.upper() or 'VOLUME' in col.upper()]

    if aadt_cols:
        for col in aadt_cols:
            non_null = gdf[col].notna().sum()
            print(f"  ✓ {col}: {non_null:,} non-null values ({non_null/len(gdf)*100:.1f}%)")

            # Show sample statistics if numeric
            if gdf[col].dtype in ['int64', 'float64']:
                try:
                    print(f"      Range: {gdf[col].min():,.0f} - {gdf[col].max():,.0f}")
                    print(f"      Mean: {gdf[col].mean():,.0f}")
                except:
                    pass
    else:
        print("  ⚠ No obvious AADT fields found")
        print("  Available columns:", list(gdf.columns))

    # Check geometry
    print(f"\nGeometry:")
    print(f"  Type: {gdf.geometry.type.unique()}")
    print(f"  Valid: {gdf.geometry.is_valid.sum():,} / {len(gdf):,}")

    # Geographic extent
    bounds = gdf.total_bounds
    print(f"  Extent: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")

    return gdf

def main():
    """Main execution"""

    # Download the data
    gdf = download_texas_aadt()

    if gdf is not None:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Load the data:")
        print("   import geopandas as gpd")
        print(f"   tx_traffic = gpd.read_file('{TEXAS_BRONZE_TRAFFIC / 'texas_aadt_traffic_counts.gpkg'}')")
        print("\n2. Inspect the data:")
        print("   print(tx_traffic.columns)")
        print("   print(tx_traffic.head())")
        print("\n3. Spatial join with work zones:")
        print("   work_zones_with_traffic = gpd.sjoin_nearest(work_zones_gdf, tx_traffic)")
        print("\n4. Use as ML feature:")
        print("   features_df['aadt'] = work_zones_with_traffic['AADT']  # or whatever the column name is")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("DOWNLOAD FAILED")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify URL is still active:")
        print("   https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/TxDOT_Ramp_AADT_Traffic_Counts/FeatureServer/0")
        print("3. Try manual download via browser:")
        print("   Add '?outFields=*&where=1=1&f=geojson' to the URL")
        print("="*60)

if __name__ == "__main__":
    main()
