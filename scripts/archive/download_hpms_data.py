"""
Download HPMS Traffic Data from FHWA
Fetches AADT (traffic volume) data for California, Texas, New York

Usage:
    python scripts/download_hpms_data.py
"""

import geopandas as gpd
import requests
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def download_hpms_via_rest(state_name, state_fips, output_dir='data/raw/hpms'):
    """
    Download HPMS data via REST API (no authentication needed)

    Parameters:
    -----------
    state_name : str
        State name (e.g., 'California')
    state_fips : str
        State FIPS code (e.g., '06' for CA)
    output_dir : str
        Output directory

    Returns:
    --------
    geopandas.GeoDataFrame or None
    """

    print(f"\n{'='*60}")
    print(f"Downloading HPMS data for {state_name}")
    print(f"{'='*60}")

    # FHWA HPMS 2018 Public Release Feature Server
    # Note: This is the most recent publicly available HPMS data
    base_url = "https://geo.dot.gov/server/rest/services/Hosted/HPMS_Full_Extent_2018/FeatureServer/0"

    print(f"Source: {base_url}")
    print(f"Filtering: STATE_CODE = '{state_fips}'")

    # Query parameters
    params = {
        'where': f"STATE_CODE = '{state_fips}'",
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'geojson',
        'resultRecordCount': 1000  # Max per request
    }

    all_features = []
    offset = 0
    max_iterations = 200  # Safety limit

    print(f"\nDownloading in batches of 1000...")

    for iteration in range(max_iterations):
        params['resultOffset'] = offset

        try:
            response = requests.get(f"{base_url}/query", params=params, timeout=120)

            if response.status_code != 200:
                print(f"✗ Error: HTTP {response.status_code}")
                if len(all_features) > 0:
                    print(f"Continuing with {len(all_features)} features downloaded so far...")
                    break
                return None

            data = response.json()

            # Check for errors in response
            if 'error' in data:
                print(f"✗ API Error: {data['error']}")
                if len(all_features) > 0:
                    break
                return None

            if 'features' not in data:
                print("✗ No 'features' in response")
                break

            features = data['features']

            if len(features) == 0:
                print(f"\n✓ Download complete")
                break

            all_features.extend(features)
            offset += 1000

            print(f"  Batch {iteration + 1}: Downloaded {len(all_features):,} segments total", end='\r')

            # If we got fewer than 1000, we're done
            if len(features) < 1000:
                print(f"\n✓ Download complete (last batch)")
                break

        except requests.exceptions.Timeout:
            print(f"\n⚠ Request timeout at offset {offset}")
            if len(all_features) > 0:
                print(f"Continuing with {len(all_features)} features...")
                break
            return None

        except Exception as e:
            print(f"\n✗ Error: {e}")
            if len(all_features) > 0:
                print(f"Continuing with {len(all_features)} features...")
                break
            return None

    if len(all_features) == 0:
        print("✗ No features downloaded")
        return None

    print(f"\n\nProcessing {len(all_features):,} features...")

    # Convert to GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    except Exception as e:
        print(f"✗ Error creating GeoDataFrame: {e}")
        return None

    # Check for AADT field
    if 'AADT' not in gdf.columns and 'aadt' not in gdf.columns:
        print("⚠ Warning: AADT field not found in data")
        print(f"Available columns: {list(gdf.columns)}")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{state_name.lower().replace(" ", "_")}_hpms_2018.gpkg')

    try:
        gdf.to_file(output_file, driver='GPKG')
        print(f"✓ Saved to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return None

    # Print summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY - {state_name}")
    print(f"{'='*60}")
    print(f"Total segments: {len(gdf):,}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    # AADT statistics (handle different possible column names)
    aadt_col = 'AADT' if 'AADT' in gdf.columns else 'aadt' if 'aadt' in gdf.columns else None

    if aadt_col:
        aadt_data = gdf[aadt_col].dropna()
        if len(aadt_data) > 0:
            print(f"\nAADT Statistics:")
            print(f"  Segments with AADT: {len(aadt_data):,} ({len(aadt_data)/len(gdf)*100:.1f}%)")
            print(f"  AADT range: {aadt_data.min():,.0f} - {aadt_data.max():,.0f}")
            print(f"  Mean AADT: {aadt_data.mean():,.0f}")
            print(f"  Median AADT: {aadt_data.median():,.0f}")

    # Key fields summary
    print(f"\nKey Fields Available:")
    important_fields = ['AADT', 'ROUTE_ID', 'ROUTE_NUMBER', 'F_SYSTEM', 'THROUGH_LANES',
                       'SPEED_LIMIT', 'URBAN_CODE', 'STATE_CODE']
    for field in important_fields:
        if field in gdf.columns:
            print(f"  ✓ {field}")
        elif field.lower() in gdf.columns:
            print(f"  ✓ {field.lower()}")
        else:
            print(f"  ✗ {field} (not found)")

    return gdf

def main():
    """Download HPMS data for California, Texas, and New York"""

    print("\n" + "="*60)
    print("HPMS DATA DOWNLOADER")
    print("Federal Highway Administration HPMS 2018 Data")
    print("="*60)

    # State FIPS codes
    states = {
        'California': '06',
        'Texas': '48',
        'New York': '36'
    }

    results = {}

    for state_name, fips_code in states.items():
        try:
            gdf = download_hpms_via_rest(state_name, fips_code)
            results[state_name] = gdf is not None
        except Exception as e:
            print(f"\n✗ {state_name} failed: {e}")
            results[state_name] = False

        print()  # Blank line between states

    # Final summary
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for state_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{state_name}: {status}")

    successful = sum(results.values())
    print(f"\nDownloaded: {successful}/{len(states)} states")

    if successful > 0:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Load HPMS data:")
        print("   import geopandas as gpd")
        print("   ca_hpms = gpd.read_file('data/raw/hpms/california_hpms_2018.gpkg')")

        print("\n2. Spatial join with work zones:")
        print("   work_zones_with_aadt = gpd.sjoin_nearest(work_zones_gdf, ca_hpms)")

        print("\n3. Use AADT as ML feature:")
        print("   features_df['aadt'] = work_zones_with_aadt['AADT']")

        print("\n4. Calculate crash rate:")
        print("   crash_rate = crashes / (AADT * days * length / 1M)")

        print("\nSee docs/HPMS_DATA_ACCESS_GUIDE.md for detailed instructions")
        print("="*60)
    else:
        print("\n⚠ All downloads failed!")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify FHWA server is online: https://geo.dot.gov/server/rest/services")
        print("3. Try manual download via QGIS (see HPMS_DATA_ACCESS_GUIDE.md)")
        print("4. Contact FHWA: thomas.roff@dot.gov")

    print()

if __name__ == "__main__":
    main()
