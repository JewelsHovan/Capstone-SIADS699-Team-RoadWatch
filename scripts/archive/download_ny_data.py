"""
New York Data Extractor
Downloads NY work zone and crash data from official sources

Data Sources:
- Work Zones: https://511ny.org/api/wzdx (WZDx v4.2 feed)
- Crashes: https://data.ny.gov/resource/e8ky-4vqe.json (NY Open Data - Case Information)

Output Structure:
- data/raw/ny_wzdx_feed.json - Work zone feed
- data/raw/crashes/ny_crashes.csv - Crash data (CSV)
- data/raw/crashes/ny_crashes.json - Crash data (JSON)
"""

import requests
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


class NewYorkDataExtractor:
    """Handles downloading and processing NY work zone and crash data"""

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize NY data extractor

        Args:
            data_dir: Root data directory (default: 'data')
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.crashes_dir = self.raw_dir / 'crashes'
        self.processed_dir = self.data_dir / 'processed'

        # API endpoints
        self.wzdx_url = "https://511ny.org/api/wzdx"
        self.crash_api_url = "https://data.ny.gov/resource/e8ky-4vqe.json"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_work_zones(self, save_raw: bool = True) -> Optional[Dict]:
        """
        Download NY work zone data from WZDx feed

        Args:
            save_raw: Save raw JSON feed to file

        Returns:
            dict: WZDx feed data or None if failed
        """
        print("\n" + "="*70)
        print("NEW YORK WORK ZONE DATA (WZDx)")
        print("="*70)
        print(f"Source: {self.wzdx_url}")

        try:
            print("\nFetching WZDx feed...", end='')
            response = requests.get(self.wzdx_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            print(" ✓")

            # Save raw feed
            if save_raw:
                output_file = self.raw_dir / 'ny_wzdx_feed.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Saved to: {output_file}")

                # File stats
                file_size = output_file.stat().st_size / 1024  # KB
                print(f"  File size: {file_size:.1f} KB")

            # Summary stats
            feed_info = data.get('feed_info', {})
            features = data.get('features', [])

            print(f"\nFeed Information:")
            print(f"  Publisher: {feed_info.get('publisher', 'N/A')}")
            print(f"  Version: {feed_info.get('version', 'N/A')}")
            print(f"  Last Update: {feed_info.get('update_date', 'N/A')}")
            print(f"  Total Features: {len(features)}")

            # Count work zones vs devices
            work_zones = sum(1 for f in features
                           if f.get('properties', {}).get('core_details', {}).get('event_type') == 'work-zone')
            devices = len(features) - work_zones

            print(f"  Work Zones: {work_zones}")
            print(f"  Field Devices: {devices}")

            return data

        except requests.exceptions.RequestException as e:
            print(f"\n✗ Error fetching work zone data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"\n✗ Error parsing JSON: {e}")
            return None
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_crashes(self,
                        limit: Optional[int] = None,
                        years: Optional[List[str]] = None,
                        save_formats: List[str] = ['csv', 'json']) -> Optional[pd.DataFrame]:
        """
        Download NY crash data from Open Data API

        Args:
            limit: Maximum number of records (None = all available, with pagination)
            years: Filter by specific years (e.g., ['2023', '2024'])
            save_formats: Output formats ('csv', 'json', or both)

        Returns:
            pd.DataFrame: Crash data or None if failed
        """
        print("\n" + "="*70)
        print("NEW YORK CRASH DATA (NY Open Data)")
        print("="*70)
        print(f"Source: {self.crash_api_url}")

        try:
            # Build query parameters
            params = {
                '$limit': limit if limit else 50000,  # Socrata default max
                '$offset': 0,
                '$order': 'date DESC'
            }

            # Add year filter if specified
            if years:
                year_filter = " OR ".join([f"year='{year}'" for year in years])
                params['$where'] = year_filter
                print(f"Filtering: Years {', '.join(years)}")

            all_records = []
            batch_num = 1

            print("\nFetching crash records...")

            while True:
                print(f"  Batch {batch_num} (offset: {params['$offset']:,})...", end='')

                response = requests.get(self.crash_api_url, params=params, timeout=60)
                response.raise_for_status()

                batch = response.json()

                if not batch or len(batch) == 0:
                    print(" Complete (no more records)")
                    break

                all_records.extend(batch)
                print(f" ✓ Got {len(batch):,} records (Total: {len(all_records):,})")

                # Check if we got fewer records than requested (last batch)
                if len(batch) < params['$limit']:
                    print("  Complete (last batch)")
                    break

                # If user specified a limit, stop when reached
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    print(f"  Reached limit of {limit:,} records")
                    break

                # Move to next batch
                params['$offset'] += params['$limit']
                batch_num += 1

            if not all_records:
                print("✗ No records downloaded")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_records)

            print(f"\n{'='*70}")
            print(f"DATA SUMMARY")
            print(f"{'='*70}")
            print(f"Total records: {len(df):,}")
            print(f"Columns: {len(df.columns)}")

            # Date range
            if 'date' in df.columns:
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                print(f"Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")

            # Year distribution
            if 'year' in df.columns:
                print(f"\nYear distribution:")
                for year, count in df['year'].value_counts().sort_index().items():
                    print(f"  {year}: {count:,}")

            # Accident types
            if 'accident_descriptor' in df.columns:
                print(f"\nAccident types:")
                for acc_type, count in df['accident_descriptor'].value_counts().head(5).items():
                    print(f"  {acc_type}: {count:,}")

            # County distribution
            if 'county_name' in df.columns:
                print(f"\nTop counties:")
                for county, count in df['county_name'].value_counts().head(5).items():
                    print(f"  {county}: {count:,}")

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if 'csv' in save_formats:
                csv_file = self.crashes_dir / 'ny_crashes.csv'
                df.to_csv(csv_file, index=False)
                file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                print(f"\n✓ Saved CSV: {csv_file} ({file_size:.1f} MB)")

            if 'json' in save_formats:
                json_file = self.crashes_dir / 'ny_crashes.json'
                df.to_json(json_file, orient='records', indent=2)
                file_size = json_file.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ Saved JSON: {json_file} ({file_size:.1f} MB)")

            # Also save a timestamped backup
            backup_file = self.crashes_dir / f'ny_crashes_{timestamp}.csv'
            df.to_csv(backup_file, index=False)
            print(f"✓ Backup: {backup_file}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"\n✗ Error fetching crash data: {e}")
            return None
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_all(self,
                    crash_limit: Optional[int] = None,
                    crash_years: Optional[List[str]] = None) -> Dict:
        """
        Download both work zone and crash data

        Args:
            crash_limit: Limit on crash records
            crash_years: Filter crashes by years

        Returns:
            dict: Status of downloads
        """
        print("\n" + "="*80)
        print(" " * 20 + "NEW YORK DATA EXTRACTION")
        print("="*80)

        results = {
            'work_zones': None,
            'crashes': None,
            'success': False
        }

        # Download work zones
        wzdx_data = self.download_work_zones()
        results['work_zones'] = wzdx_data is not None

        # Download crashes
        crash_data = self.download_crashes(limit=crash_limit, years=crash_years)
        results['crashes'] = crash_data is not None

        # Overall success
        results['success'] = results['work_zones'] and results['crashes']

        # Final summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Work Zones: {'✓ Success' if results['work_zones'] else '✗ Failed'}")
        print(f"Crashes:    {'✓ Success' if results['crashes'] else '✗ Failed'}")

        if results['success']:
            print("\n🎉 All data downloaded successfully!")
            print("\nNext steps:")
            print("1. Analyze work zones:")
            print("   python scripts/analyze_ny_feed.py")
            print("\n2. Explore crash data:")
            print("   import pandas as pd")
            print("   crashes = pd.read_csv('data/raw/crashes/ny_crashes.csv')")
            print("   print(crashes.head())")
            print("\n3. Integrate with work zones:")
            print("   - Spatial join crashes near work zones")
            print("   - Analyze crash risk factors")
            print("   - Build ML features")
        else:
            print("\n⚠ Some downloads failed. Check errors above.")

        print("="*80)

        return results


def main():
    """Main execution - download all NY data"""
    import argparse

    parser = argparse.ArgumentParser(description='Download New York work zone and crash data')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--crash-limit', type=int, help='Limit crash records (default: all)')
    parser.add_argument('--years', nargs='+', help='Filter crashes by years (e.g., 2023 2024)')
    parser.add_argument('--work-zones-only', action='store_true', help='Download only work zones')
    parser.add_argument('--crashes-only', action='store_true', help='Download only crashes')

    args = parser.parse_args()

    # Initialize extractor
    extractor = NewYorkDataExtractor(data_dir=args.data_dir)

    # Download based on flags
    if args.work_zones_only:
        extractor.download_work_zones()
    elif args.crashes_only:
        extractor.download_crashes(limit=args.crash_limit, years=args.years)
    else:
        extractor.download_all(crash_limit=args.crash_limit, crash_years=args.years)


if __name__ == "__main__":
    main()
