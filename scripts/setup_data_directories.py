#!/usr/bin/env python3
"""
Data Directory Setup Script

Creates the complete Medallion Architecture directory structure
for the Texas Crash Prediction project.

This should be run BEFORE acquiring data or running the pipeline.

Usage:
    python scripts/setup_data_directories.py

Author: Julien Hovan
Date: 2025-11-24
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.paths import (
    ensure_directories,
    DATA_ROOT,
    BRONZE, BRONZE_TEXAS,
    SILVER, SILVER_TEXAS,
    GOLD, GOLD_ML_DATASETS, GOLD_ANALYTICS,
    CRASH_LEVEL_ML, SEGMENT_LEVEL_ML,
    TEXAS_BRONZE_CRASHES, TEXAS_BRONZE_WORKZONES,
    TEXAS_BRONZE_TRAFFIC, TEXAS_BRONZE_WEATHER,
    TEXAS_SILVER_ROADWAY, TEXAS_SILVER_WORKZONES,
    TEXAS_SILVER_TRAFFIC, TEXAS_SILVER_WEATHER,
    OUTPUTS_ROOT, MAPS, VISUALIZATIONS, MODELS, REPORTS
)


def print_tree(directory, prefix='', is_last=True):
    """Print directory tree structure"""
    connector = '└── ' if is_last else '├── '
    print(f'{prefix}{connector}{directory.name}/')

    # Get subdirectories
    try:
        subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    except PermissionError:
        return

    # Print each subdirectory
    for i, subdir in enumerate(subdirs):
        is_last_subdir = (i == len(subdirs) - 1)
        extension = '    ' if is_last else '│   '
        print_tree(subdir, prefix + extension, is_last_subdir)


def create_readme_files():
    """Create README files in key directories to explain their purpose"""

    bronze_readme = BRONZE_TEXAS / 'README.md'
    if not bronze_readme.exists():
        bronze_readme.write_text("""# Bronze Layer - Raw Texas Data

This directory contains **raw, immutable data** as downloaded from external sources.

## Structure

- `crashes/` - Kaggle US Accidents dataset (Texas subset)
- `traffic/` - TxDOT AADT traffic count data
- `workzones/` - TxDOT WZDx work zone feed
- `weather/` - NOAA weather data
- `HPMS2023.gdb` - National HPMS geodatabase (if regenerating from scratch)

## Important

- Files in this layer should **never be modified** after download
- Original file formats are preserved
- Serves as the source of truth for all downstream processing

## Data Acquisition

See `DATA_ACQUISITION.md` in the project root for download instructions.
""")

    silver_readme = SILVER_TEXAS / 'README.md'
    if not silver_readme.exists():
        silver_readme.write_text("""# Silver Layer - Cleaned Texas Data

This directory contains **cleaned, validated, and standardized** data.

## Structure

- `roadway/` - HPMS Texas 2023 (filtered from national dataset)
- `traffic/` - Cleaned AADT traffic data
- `workzones/` - Processed work zone data
- `weather/` - Standardized weather data

## Processing Steps

Bronze → Silver transformations include:
- Column normalization (uppercase → lowercase)
- Type coercion (strings → numeric)
- CRS standardization (→ EPSG:4326)
- Deduplication
- Date filtering
- NULL value handling

## Key File

`roadway/hpms_texas_2023.gpkg` (1.1GB)
- Filtered from national HPMS 2023 dataset
- Contains 971K+ Texas road segments
- Includes AADT traffic data
- **This file is provided to instructors**
""")

    gold_readme = GOLD_ML_DATASETS / 'README.md'
    if not gold_readme.exists():
        gold_readme.write_text("""# Gold Layer - ML-Ready Datasets

This directory contains **ML-ready datasets** with train/val/test splits.

## Structure

- `crash_level/` - Crash-level predictions (one row per crash)
- `segment_level/` - Segment-level predictions (one row per road segment)
- `crash_level_archive/` - Historical dataset versions

## Dataset Characteristics

### Crash-Level
- 466K crashes (2016-2022)
- 67 engineered features
- Binary target: high_severity (0/1)
- Temporal splits: Train(2016-2020), Val(2021), Test(2022)
- **Production Model: Random Forest (AUC=0.93)**

### Segment-Level
- 971K road segments
- 11 features (road characteristics + crash aggregates)
- Target: crash_count (0-8,436)
- Stratified splits: 70/15/15

## Features

All features are carefully engineered to prevent data leakage:
- Temporal features (hour, day_of_week, is_weekend)
- Weather features (temperature, visibility, adverse_weather)
- Road characteristics (HPMS: speed, lanes, functional class, AADT)
- Geographic features (coarse regions, not exact lat/lng)

See `ml_engineering/preprocessing/feature_lists.py` for complete schema.
""")

    print(f'\n✓ Created README files in Bronze, Silver, and Gold layers')


def main():
    """Main setup function"""
    print('=' * 80)
    print('TEXAS CRASH PREDICTION - DATA DIRECTORY SETUP')
    print('=' * 80)

    print('\nCreating Medallion Architecture directory structure...\n')

    # Create all directories
    ensure_directories()

    # Print the structure
    print('Directory structure created:')
    print()
    print_tree(DATA_ROOT)

    # Create README files
    create_readme_files()

    print('\n' + '=' * 80)
    print('✓ SETUP COMPLETE')
    print('=' * 80)

    print('\nNext steps:')
    print('  1. Acquire data: See DATA_ACQUISITION.md')
    print('  2. Verify data: python scripts/verify_data.py')
    print('  3. Run pipeline: python scripts/run_pipeline.py')

    print('\nDirectory locations:')
    print(f'  Bronze (raw):      {BRONZE_TEXAS}')
    print(f'  Silver (cleaned):  {SILVER_TEXAS}')
    print(f'  Gold (ML-ready):   {GOLD_ML_DATASETS}')
    print(f'  Outputs:           {OUTPUTS_ROOT}')

    print()


if __name__ == '__main__':
    main()
