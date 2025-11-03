"""
Texas Work Zone Data Extractor (Socrata API)
Downloads TxDOT work zone data from USDOT Open Data Portal using Socrata API

Data Source:
- API: datahub.transportation.gov
- Dataset ID: h4kh-i7b7
- Format: WZDx v2.0
- URL: https://datahub.transportation.gov/Roadways-and-Bridges/Texas-DOT-TxDOT-Work-Zone-Data-Schema-Version-2-0/h4kh-i7b7

Output Structure:
- data/raw/texas_wzdx_feed.json - Raw JSON feed
- data/raw/texas_wzdx_feed.csv - CSV format
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from sodapy import Socrata


class TexasWorkZoneExtractor:
    """Handles downloading and processing Texas work zone data via Socrata API"""

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize Texas work zone data extractor

        Args:
            data_dir: Root data directory (default: 'data')
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'

        # Socrata API configuration
        self.socrata_domain = "datahub.transportation.gov"
        self.dataset_id = "h4kh-i7b7"  # Texas DOT Work Zone Data

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_work_zones(self,
                           limit: int = 5000,
                           save_json: bool = True,
                           save_csv: bool = True,
                           app_token: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Download Texas work zone data from Socrata API

        Args:
            limit: Maximum number of records to fetch (default: 5000)
            save_json: Save raw JSON feed to file
            save_csv: Save as CSV file
            app_token: Optional Socrata app token for higher rate limits

        Returns:
            pd.DataFrame: Work zone data or None if failed
        """
        print("\n" + "="*70)
        print("TEXAS WORK ZONE DATA (Socrata API)")
        print("="*70)
        print(f"Source: {self.socrata_domain}")
        print(f"Dataset: {self.dataset_id}")
        print(f"Limit: {limit:,} records")

        try:
            # Initialize Socrata client
            # Unauthenticated client (public data)
            # For higher rate limits, pass app_token as first parameter after domain
            print("\nConnecting to Socrata API...", end='')
            client = Socrata(self.socrata_domain, app_token)
            print(" ✓")

            # Fetch data
            print(f"Fetching up to {limit:,} records...", end='')
            results = client.get(self.dataset_id, limit=limit)
            print(f" ✓ ({len(results):,} records)")

            # Close client
            client.close()

            if not results:
                print("✗ No records returned from API")
                return None

            # Convert to DataFrame
            print("Converting to DataFrame...", end='')
            df = pd.DataFrame.from_records(results)
            print(" ✓")

            # Timestamp for files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as JSON
            if save_json:
                json_file = self.raw_dir / 'texas_wzdx_feed.json'
                print(f"\nSaving JSON...", end='')
                with open(json_file, 'w') as f:
                    json.dump(results, f, indent=2)
                file_size = json_file.stat().st_size / 1024  # KB
                print(f" ✓")
                print(f"  File: {json_file}")
                print(f"  Size: {file_size:.1f} KB")

                # Also save timestamped version
                backup_json = self.raw_dir / f'texas_wzdx_feed_{timestamp}.json'
                with open(backup_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Backup: {backup_json}")

            # Save as CSV
            if save_csv:
                csv_file = self.raw_dir / 'texas_wzdx_feed.csv'
                print(f"\nSaving CSV...", end='')
                df.to_csv(csv_file, index=False)
                file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                print(f" ✓")
                print(f"  File: {csv_file}")
                print(f"  Size: {file_size:.1f} MB")

                # Also save timestamped version
                backup_csv = self.raw_dir / f'texas_wzdx_feed_{timestamp}.csv'
                df.to_csv(backup_csv, index=False)
                print(f"  Backup: {backup_csv}")

            # Display summary statistics
            self._display_summary(df)

            return df

        except Exception as e:
            print(f"\n✗ Error fetching Texas work zone data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _display_summary(self, df: pd.DataFrame):
        """Display summary statistics about the fetched data"""
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)

        print(f"\nTotal Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")

        # Key fields analysis
        key_fields = {
            'road_event_id': 'Work Zone ID',
            'road_name': 'Road Name',
            'direction': 'Direction',
            'vehicle_impact': 'Vehicle Impact',
            'start_date': 'Start Date',
            'end_date': 'End Date',
            'workers_present': 'Workers Present',
            'subidentifier': 'Region/District'
        }

        print("\nField Completeness:")
        for field, label in key_fields.items():
            if field in df.columns:
                non_null = df[field].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f"  {label:20s}: {non_null:6,} ({pct:5.1f}%)")

        # Date range
        if 'start_date' in df.columns:
            print("\nDate Range:")
            df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date_parsed'] = pd.to_datetime(df['end_date'], errors='coerce')
            print(f"  Start Dates: {df['start_date_parsed'].min()} to {df['start_date_parsed'].max()}")
            print(f"  End Dates: {df['end_date_parsed'].min()} to {df['end_date_parsed'].max()}")

            # Active work zones (if end_date is in future)
            now = pd.Timestamp.now()
            active = df[(df['start_date_parsed'] <= now) & (df['end_date_parsed'] >= now)]
            print(f"  Currently Active: {len(active):,} ({len(active)/len(df)*100:.1f}%)")

        # Vehicle impact distribution
        if 'vehicle_impact' in df.columns:
            print("\nVehicle Impact Distribution:")
            impact_counts = df['vehicle_impact'].value_counts()
            for impact, count in impact_counts.head(5).items():
                pct = (count / len(df)) * 100
                print(f"  {str(impact)[:30]:30s}: {count:5,} ({pct:5.1f}%)")

        # Top regions
        if 'subidentifier' in df.columns:
            print("\nTop 5 Regions/Districts:")
            region_counts = df['subidentifier'].value_counts()
            for i, (region, count) in enumerate(region_counts.head(5).items(), 1):
                print(f"  {i}. {str(region)[:40]:40s}: {count:5,}")

        # Top roads
        if 'road_name' in df.columns:
            print("\nTop 5 Roads by Work Zone Count:")
            road_counts = df['road_name'].value_counts()
            for i, (road, count) in enumerate(road_counts.head(5).items(), 1):
                print(f"  {i}. {str(road)[:40]:40s}: {count:5,}")

        # Feed metadata
        if 'road_event_feed_info_feed_update_date' in df.columns:
            print("\nFeed Information:")
            feed_date = df['road_event_feed_info_feed_update_date'].mode()
            if len(feed_date) > 0:
                print(f"  Last Update: {feed_date.iloc[0]}")
            if 'road_event_feed_info_version' in df.columns:
                version = df['road_event_feed_info_version'].mode()
                if len(version) > 0:
                    print(f"  Schema Version: {version.iloc[0]}")

    def get_active_work_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only currently active work zones

        Args:
            df: Full work zone DataFrame

        Returns:
            pd.DataFrame: Active work zones only
        """
        if 'start_date' not in df.columns or 'end_date' not in df.columns:
            print("⚠ Cannot filter active zones: missing date columns")
            return df

        df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['end_date_parsed'] = pd.to_datetime(df['end_date'], errors='coerce')

        now = pd.Timestamp.now()
        active = df[(df['start_date_parsed'] <= now) & (df['end_date_parsed'] >= now)]

        print(f"\nActive Work Zones: {len(active):,} of {len(df):,} ({len(active)/len(df)*100:.1f}%)")

        return active


def main():
    """Main execution - download Texas work zone data"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Texas work zone data from Socrata API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default 5000 records
  python scripts/download_texas_feed.py

  # Download 10000 records
  python scripts/download_texas_feed.py --limit 10000

  # Download with app token (higher rate limits)
  python scripts/download_texas_feed.py --app-token YOUR_TOKEN

  # Save only JSON (not CSV)
  python scripts/download_texas_feed.py --no-csv

  # Save only CSV (not JSON)
  python scripts/download_texas_feed.py --no-json
        """
    )

    parser.add_argument('--data-dir', default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--limit', type=int, default=5000,
                       help='Maximum records to fetch (default: 5000)')
    parser.add_argument('--app-token',
                       help='Socrata app token for higher rate limits')
    parser.add_argument('--no-json', action='store_true',
                       help='Do not save JSON format')
    parser.add_argument('--no-csv', action='store_true',
                       help='Do not save CSV format')
    parser.add_argument('--active-only', action='store_true',
                       help='Save only currently active work zones')

    args = parser.parse_args()

    # Initialize extractor
    extractor = TexasWorkZoneExtractor(data_dir=args.data_dir)

    print("\n" + "="*70)
    print(" " * 20 + "TEXAS WORK ZONE EXTRACTION")
    print("="*70)

    # Download data
    df = extractor.download_work_zones(
        limit=args.limit,
        save_json=not args.no_json,
        save_csv=not args.no_csv,
        app_token=args.app_token
    )

    if df is None:
        print("\n✗ Download failed")
        return

    # Filter to active only if requested
    if args.active_only:
        df_active = extractor.get_active_work_zones(df)

        # Save active-only dataset
        active_csv = extractor.processed_dir / 'texas_work_zones_active.csv'
        df_active.to_csv(active_csv, index=False)
        print(f"\n✓ Saved active work zones: {active_csv}")

    # Success summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    if not args.no_json:
        print("  - data/raw/texas_wzdx_feed.json")
    if not args.no_csv:
        print("  - data/raw/texas_wzdx_feed.csv")
    if args.active_only:
        print("  - data/processed/texas_work_zones_active.csv")

    print("\nNext steps:")
    print("1. Analyze work zones:")
    print("   python scripts/analyze_texas_workzones.py")
    print("\n2. Integrate with AADT:")
    print("   python scripts/integrate_texas_aadt.py")
    print("\n3. View in dashboard:")
    print("   streamlit run app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
