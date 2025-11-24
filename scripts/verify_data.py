#!/usr/bin/env python3
"""
Data Verification Script

Checks that all required data files are present and valid
before running the data pipeline.

Usage:
    python scripts/verify_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
from config.paths import (
    DEFAULT_CRASH_FILE,
    DEFAULT_HPMS_FILE,
    BRONZE_TEXAS,
    SILVER_TEXAS,
    ensure_directories
)


def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f'✓ {description}: {size_mb:.1f} MB')
        return True
    else:
        print(f'✗ {description}: NOT FOUND')
        print(f'  Expected: {file_path}')
        return False


def verify_crash_data(crash_file):
    """Verify crash data is valid"""
    try:
        df = pd.read_csv(crash_file, nrows=100)
        required_cols = ['ID', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Severity']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f'  ✗ Missing columns: {missing}')
            return False

        # Count total rows (slower but accurate)
        total_rows = sum(1 for _ in open(crash_file)) - 1  # -1 for header
        print(f'  ✓ Contains {total_rows:,} crashes')
        print(f'  ✓ Required columns present')
        return True
    except Exception as e:
        print(f'  ✗ Error reading file: {e}')
        return False


def verify_hpms_data(hpms_file):
    """Verify HPMS data is valid"""
    try:
        # Read just the first few rows to check structure
        gdf = gpd.read_file(hpms_file, rows=100)

        # Check for key columns
        required_cols = ['geometry']
        recommended_cols = ['speed_limit', 'through_lanes', 'aadt']

        missing_required = [col for col in required_cols if col not in gdf.columns]
        if missing_required:
            print(f'  ✗ Missing required columns: {missing_required}')
            return False

        missing_recommended = [col for col in recommended_cols if col not in gdf.columns]
        if missing_recommended:
            print(f'  ⚠️  Missing recommended columns: {missing_recommended}')

        # Count total features (full read - may take a moment)
        print(f'  Counting total road segments...')
        full_gdf = gpd.read_file(hpms_file)
        print(f'  ✓ Contains {len(full_gdf):,} road segments')
        print(f'  ✓ Geometry type: {full_gdf.geometry.geom_type.mode()[0]}')
        print(f'  ✓ CRS: {full_gdf.crs}')
        return True
    except Exception as e:
        print(f'  ✗ Error reading file: {e}')
        return False


def main():
    """Main verification function"""
    print('=' * 80)
    print('DATA VERIFICATION')
    print('=' * 80)

    # Check directory structure
    print('\n1. Checking directory structure...')
    ensure_directories()
    print('  ✓ Directory structure initialized')

    # Check required files
    print('\n2. Checking required data files...')
    print()

    all_ok = True

    # Check crash data
    print('Crash Data (Bronze Layer):')
    if check_file_exists(DEFAULT_CRASH_FILE, 'Kaggle US Accidents'):
        if not verify_crash_data(DEFAULT_CRASH_FILE):
            all_ok = False
    else:
        all_ok = False
        print('  ℹ️  See DATA_ACQUISITION.md for download instructions')

    print()

    # Check HPMS data
    print('HPMS Data (Silver Layer):')
    if check_file_exists(DEFAULT_HPMS_FILE, 'HPMS Texas 2023'):
        if not verify_hpms_data(DEFAULT_HPMS_FILE):
            all_ok = False
    else:
        all_ok = False
        print('  ℹ️  This file should be provided to instructors')
        print('  ℹ️  See DATA_ACQUISITION.md for regeneration instructions')

    print()
    print('=' * 80)

    if all_ok:
        print('✓ ALL CHECKS PASSED')
        print()
        print('Next steps:')
        print('  1. Run pipeline: python scripts/run_pipeline.py')
        print('  2. Train models: python -m ml_engineering.train_with_mlflow --dataset crash --model all')
        print('  3. Launch app: streamlit run app/Home.py')
        return 0
    else:
        print('✗ VERIFICATION FAILED')
        print()
        print('Please fix the issues above before running the pipeline.')
        print('See DATA_ACQUISITION.md for data download instructions.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
