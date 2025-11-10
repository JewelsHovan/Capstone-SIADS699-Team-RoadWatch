#!/usr/bin/env python3
"""
Download Austin crash data from City of Austin Open Data Portal (Socrata API)

This script downloads crash data from the TxDOT CRIS database for Austin/Travis County.
It includes filtering for construction zones and handles API pagination automatically.

Data source: https://data.austintexas.gov/Transportation-and-Mobility/Austin-Crash-Report-Data-Crash-Level-Records/y2wy-tgr5
API endpoint: https://data.austintexas.gov/resource/y2wy-tgr5.json

Usage:
    python download_austin_crashes.py --all              # Download all crashes
    python download_austin_crashes.py --construction     # Construction zones only
    python download_austin_crashes.py --years 2023 2024  # Specific years
    python download_austin_crashes.py --sample 1000      # Sample for testing
"""

import requests
import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime
import time

from config.paths import TEXAS_BRONZE_CRASHES

# Configuration
BASE_URL = "https://data.austintexas.gov/resource/y2wy-tgr5.json"
LIMIT_PER_REQUEST = 50000  # Socrata API limit
OUTPUT_DIR = TEXAS_BRONZE_CRASHES

def count_records(construction_only=False, start_date=None, end_date=None):
    """Get total count of records matching filters"""
    params = {"$select": "count(*)"}

    where_clauses = []
    if construction_only:
        where_clauses.append("road_constr_zone_fl=true")
    if start_date:
        where_clauses.append(f"crash_timestamp >= '{start_date}T00:00:00'")
    if end_date:
        where_clauses.append(f"crash_timestamp < '{end_date}T23:59:59'")

    if where_clauses:
        params["$where"] = " AND ".join(where_clauses)

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return int(data[0]['count'])
    except Exception as e:
        print(f"Error counting records: {e}")
        return None

def download_crashes(construction_only=False, start_date=None, end_date=None,
                    sample_size=None, verbose=True):
    """
    Download crash data from Austin Open Data Portal

    Args:
        construction_only: If True, only download crashes in construction zones
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sample_size: If set, limit to this many records (for testing)
        verbose: Print progress

    Returns:
        DataFrame with crash data
    """

    # Count total records
    if verbose:
        print("🔍 Counting records...")
        total_count = count_records(construction_only, start_date, end_date)
        if total_count:
            print(f"📊 Found {total_count:,} crashes matching filters")
            if sample_size and sample_size < total_count:
                print(f"📦 Will download sample of {sample_size:,} records")
        else:
            print("⚠️  Could not count records, proceeding with download...")

    # Build query
    params = {
        "$limit": LIMIT_PER_REQUEST,
        "$offset": 0,
        "$order": "crash_timestamp DESC"
    }

    where_clauses = []
    if construction_only:
        where_clauses.append("road_constr_zone_fl=true")
    if start_date:
        where_clauses.append(f"crash_timestamp >= '{start_date}T00:00:00'")
    if end_date:
        where_clauses.append(f"crash_timestamp < '{end_date}T23:59:59'")

    if where_clauses:
        params["$where"] = " AND ".join(where_clauses)

    # Download data with pagination
    all_data = []
    offset = 0

    if verbose:
        print(f"\n📥 Downloading crash data...")
        print(f"   API: {BASE_URL}")
        if construction_only:
            print(f"   Filter: Construction zones only")
        if start_date or end_date:
            print(f"   Date range: {start_date or 'all'} to {end_date or 'all'}")
        print()

    while True:
        # Apply sample size limit
        if sample_size and len(all_data) >= sample_size:
            print(f"✅ Reached sample size limit ({sample_size:,} records)")
            break

        params["$offset"] = offset

        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if not data:
                if verbose:
                    print(f"✅ Download complete!")
                break

            all_data.extend(data)
            offset += len(data)

            if verbose:
                print(f"   Downloaded: {len(all_data):,} records...", end='\r')

            # Respect API rate limits
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error downloading data at offset {offset}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            break

    if verbose:
        print(f"\n\n📦 Total records downloaded: {len(all_data):,}")

    # Convert to DataFrame
    if not all_data:
        print("⚠️  No data downloaded!")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    if verbose:
        print(f"📋 DataFrame shape: {df.shape}")
        print(f"📅 Date range: {df['crash_timestamp'].min()} to {df['crash_timestamp'].max()}")

    return df

def save_data(df, construction_only=False, start_year=None, end_year=None):
    """Save DataFrame to CSV with descriptive filename"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["austin_crashes"]

    if construction_only:
        parts.append("construction")

    if start_year and end_year:
        if start_year == end_year:
            parts.append(str(start_year))
        else:
            parts.append(f"{start_year}_{end_year}")
    elif start_year:
        parts.append(f"{start_year}+")

    parts.append(timestamp)
    filename = "_".join(parts) + ".csv"

    filepath = OUTPUT_DIR / filename

    # Save CSV
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved to: {filepath}")
    print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    # Create symlink to latest
    latest_link = OUTPUT_DIR / "austin_crashes_latest.csv"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filepath.name)
    print(f"   Symlink: {latest_link}")

    return filepath

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)

    print(f"\nTotal crashes: {len(df):,}")

    # Construction zones
    if 'road_constr_zone_fl' in df.columns:
        constr_count = df['road_constr_zone_fl'].sum()
        constr_pct = constr_count / len(df) * 100
        print(f"Construction zones: {constr_count:,} ({constr_pct:.1f}%)")

    # Severity
    if 'crash_sev_id' in df.columns:
        print(f"\nSeverity distribution:")
        severity_map = {
            0: 'Unknown',
            1: 'Incapacitating Injury',
            2: 'Non-Incapacitating Injury',
            3: 'Possible Injury',
            4: 'Killed',
            5: 'Not Injured'
        }
        for sev_id, count in df['crash_sev_id'].value_counts().sort_index().items():
            sev_name = severity_map.get(int(sev_id), f'Code {sev_id}')
            print(f"  {sev_name}: {count:,}")

    # Deaths and injuries
    if 'death_cnt' in df.columns:
        total_deaths = pd.to_numeric(df['death_cnt'], errors='coerce').sum()
        print(f"\nTotal deaths: {int(total_deaths):,}")

    if 'tot_injry_cnt' in df.columns:
        total_injuries = pd.to_numeric(df['tot_injry_cnt'], errors='coerce').sum()
        print(f"Total injuries: {int(total_injuries):,}")

    # Date range
    if 'crash_timestamp' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_timestamp'])
        print(f"\nDate range: {df['crash_date'].min().date()} to {df['crash_date'].max().date()}")

        # By year
        df['year'] = df['crash_date'].dt.year
        print(f"\nCrashes by year:")
        for year, count in df['year'].value_counts().sort_index().items():
            print(f"  {year}: {count:,}")

    # Geographic coverage
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_coords = df[['latitude', 'longitude']].notna().all(axis=1).sum()
        print(f"\nRecords with lat/lon: {valid_coords:,} ({valid_coords/len(df)*100:.1f}%)")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Download Austin crash data from City of Austin Open Data Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all crashes (all years)
  python download_austin_crashes.py --all

  # Construction zones only
  python download_austin_crashes.py --construction

  # Specific years
  python download_austin_crashes.py --years 2023 2024

  # Construction zones for 2023-2024
  python download_austin_crashes.py --construction --years 2023 2024

  # Sample for testing
  python download_austin_crashes.py --sample 1000
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Download all crashes (all years)')
    parser.add_argument('--construction', action='store_true',
                       help='Only download construction zone crashes')
    parser.add_argument('--years', type=int, nargs='+',
                       help='Specific years to download (e.g., --years 2023 2024)')
    parser.add_argument('--sample', type=int,
                       help='Download only this many records (for testing)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to file (useful for testing)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Validate arguments
    if not (args.all or args.construction or args.years or args.sample):
        parser.print_help()
        print("\n⚠️  Please specify at least one option (--all, --construction, --years, or --sample)")
        sys.exit(1)

    # Determine date range
    start_date = None
    end_date = None
    start_year = None
    end_year = None

    if args.years:
        start_year = min(args.years)
        end_year = max(args.years)
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

    # Download data
    print("🚀 Austin Crash Data Downloader")
    print("="*60 + "\n")

    df = download_crashes(
        construction_only=args.construction,
        start_date=start_date,
        end_date=end_date,
        sample_size=args.sample,
        verbose=not args.quiet
    )

    if df.empty:
        print("❌ No data downloaded")
        sys.exit(1)

    # Print summary
    print_summary(df)

    # Save to file
    if not args.no_save:
        save_data(df, args.construction, start_year, end_year)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
