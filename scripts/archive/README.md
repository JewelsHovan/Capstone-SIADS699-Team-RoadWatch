# Archived Scripts

This directory contains scripts that were used for one-time exploratory work or deprecated functionality. They are kept for historical reference but are not part of the active codebase.

## Scripts

### New York State Exploration (Not Currently Used)
- **analyze_ny_crashes.py** - NY crash data exploration
- **integrate_ny_county_data.py** - NY county-level integration
- **download_ny_data.py** - Download NY state data

### Deprecated/Buggy Features
- **match_crashes_to_workzones.py** - Work zone matching (buggy spatial logic)
- **download_hpms_data.py** - Generic HPMS downloader (replaced by Texas-specific version)

## Why Archived?

These scripts were moved to archive because:
1. **One-time use**: Exploratory analysis completed
2. **Out of scope**: NY data not currently in project scope
3. **Deprecated**: Better implementations exist in `data_engineering/`
4. **Buggy features**: Work zone proximity had spatial join issues

## If You Need Them

If you need to resurrect any of these scripts:
1. Review and update dependencies
2. Update file paths to use `config/paths.py`
3. Test thoroughly before using

**Last Updated**: 2025-11-04
