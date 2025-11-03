#!/usr/bin/env python3
"""
Download NOAA weather data for Texas counties/metros

This script uses the NOAA Climate Data Online (CDO) API to download
daily weather summaries for major Texas metropolitan areas.

API Documentation: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

Usage:
    python download_noaa_weather.py --token YOUR_TOKEN
    python download_noaa_weather.py --token YOUR_TOKEN --start 2019-01-01 --end 2024-12-31
    python download_noaa_weather.py --token YOUR_TOKEN --counties Travis,Harris,Bexar,El_Paso

Get API token at: https://www.ncdc.noaa.gov/cdo-web/token
"""

import requests
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
OUTPUT_DIR = Path("data/raw/weather")

# Major Texas metro areas with primary weather stations
TEXAS_METROS = {
    'Travis': {
        'name': 'Austin',
        'station': 'USW00013904',  # Austin-Bergstrom International
        'county_fips': '48453',
        'lat': 30.1945,
        'lon': -97.6699
    },
    'Harris': {
        'name': 'Houston',
        'station': 'USW00012960',  # George Bush Intercontinental
        'county_fips': '48201',
        'lat': 29.9844,
        'lon': -95.3414
    },
    'Dallas': {
        'name': 'Dallas',
        'station': 'USW00013960',  # Dallas/Fort Worth International
        'county_fips': '48113',
        'lat': 32.8998,
        'lon': -97.0403
    },
    'Bexar': {
        'name': 'San Antonio',
        'station': 'USW00012921',  # San Antonio International
        'county_fips': '48029',
        'lat': 29.5337,
        'lon': -98.4698
    },
    'El_Paso': {
        'name': 'El Paso',
        'station': 'USW00023044',  # El Paso International
        'county_fips': '48141',
        'lat': 31.8068,
        'lon': -106.3773
    },
    'Tarrant': {
        'name': 'Fort Worth',
        'station': 'USW00013960',  # Shares DFW airport with Dallas
        'county_fips': '48439',
        'lat': 32.8998,
        'lon': -97.0403
    }
}

# Data types to request
DATA_TYPES = [
    'PRCP',  # Precipitation (tenths of mm)
    'SNOW',  # Snowfall (mm)
    'SNWD',  # Snow depth (mm)
    'TMAX',  # Maximum temperature (tenths of degrees C)
    'TMIN',  # Minimum temperature (tenths of degrees C)
    'AWND',  # Average wind speed (tenths of m/s)
    'TAVG',  # Average temperature (tenths of degrees C)
]

def download_weather_for_station(station_id, start_date, end_date, token, verbose=True):
    """
    Download weather data for a specific station

    Args:
        station_id: NOAA station ID (e.g., 'USW00013904')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        token: NOAA API token
        verbose: Print progress

    Returns:
        DataFrame with weather data
    """

    params = {
        'dataset': 'daily-summaries',
        'stations': station_id,
        'startDate': start_date,
        'endDate': end_date,
        'dataTypes': ','.join(DATA_TYPES),
        'format': 'json',
        'units': 'standard',
        'includeStationName': 'true',
        'includeStationLocation': 'true'
    }

    headers = {
        'token': token
    }

    if verbose:
        print(f"   Requesting data for station {station_id}...")

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()

        if not data:
            if verbose:
                print(f"   ⚠️  No data returned for station {station_id}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if verbose:
            print(f"   ✅ Retrieved {len(df):,} records")

        return df

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"   ⚠️  Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return download_weather_for_station(station_id, start_date, end_date, token, verbose)
        else:
            print(f"   ❌ HTTP error: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ Error downloading data: {e}")
        return pd.DataFrame()

def download_all_metros(counties, start_date, end_date, token, verbose=True):
    """
    Download weather data for all specified counties/metros

    Args:
        counties: List of county names (keys in TEXAS_METROS)
        start_date: Start date
        end_date: End date
        token: API token
        verbose: Print progress

    Returns:
        Combined DataFrame
    """

    all_data = []

    print(f"\n📥 Downloading weather data for {len(counties)} metros")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Data types: {', '.join(DATA_TYPES)}\n")

    for county in counties:
        if county not in TEXAS_METROS:
            print(f"⚠️  Unknown county: {county}. Skipping.")
            continue

        metro = TEXAS_METROS[county]

        if verbose:
            print(f"📍 {metro['name']} ({county} County)")

        df = download_weather_for_station(
            metro['station'],
            start_date,
            end_date,
            token,
            verbose
        )

        if not df.empty:
            # Add metro metadata
            df['county_name'] = county
            df['metro_name'] = metro['name']
            df['county_fips'] = metro['county_fips']
            df['metro_lat'] = metro['lat']
            df['metro_lon'] = metro['lon']

            all_data.append(df)

        # Respect API rate limits (max 5 requests per second)
        time.sleep(1)

    if not all_data:
        print("\n❌ No data downloaded for any metro")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    print(f"\n📦 Total records downloaded: {len(combined):,}")

    return combined

def clean_weather_data(df, verbose=True):
    """Clean and transform weather data"""

    if verbose:
        print("\n🧹 Cleaning weather data...")

    # Convert date to datetime (API returns 'DATE' uppercase)
    if 'DATE' in df.columns:
        df['date'] = pd.to_datetime(df['DATE'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Convert units (NOAA uses tenths)
    if 'PRCP' in df.columns:
        df['precipitation_mm'] = pd.to_numeric(df['PRCP'], errors='coerce') / 10

    if 'TMAX' in df.columns:
        df['temp_max_c'] = pd.to_numeric(df['TMAX'], errors='coerce') / 10
        df['temp_max_f'] = df['temp_max_c'] * 9/5 + 32

    if 'TMIN' in df.columns:
        df['temp_min_c'] = pd.to_numeric(df['TMIN'], errors='coerce') / 10
        df['temp_min_f'] = df['temp_min_c'] * 9/5 + 32

    if 'TAVG' in df.columns:
        df['temp_avg_c'] = pd.to_numeric(df['TAVG'], errors='coerce') / 10
        df['temp_avg_f'] = df['temp_avg_c'] * 9/5 + 32

    if 'AWND' in df.columns:
        df['wind_speed_ms'] = pd.to_numeric(df['AWND'], errors='coerce') / 10
        df['wind_speed_mph'] = df['wind_speed_ms'] * 2.237

    if 'SNOW' in df.columns:
        df['snowfall_mm'] = pd.to_numeric(df['SNOW'], errors='coerce')

    if 'SNWD' in df.columns:
        df['snow_depth_mm'] = pd.to_numeric(df['SNWD'], errors='coerce')

    # Extract year, month, day
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    if verbose:
        print(f"   ✅ Cleaned {len(df):,} records")

    return df

def save_data(df, start_date, end_date):
    """Save weather data to CSV"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"texas_weather_{start_str}_{end_str}_{timestamp}.csv"
    filepath = OUTPUT_DIR / filename

    # Save
    df.to_csv(filepath, index=False)

    print(f"\n💾 Saved to: {filepath}")
    print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    # Create symlink to latest
    latest_link = OUTPUT_DIR / "texas_weather_latest.csv"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filepath.name)
    print(f"   Symlink: {latest_link}")

    return filepath

def print_summary(df):
    """Print summary statistics"""

    print("\n" + "="*70)
    print("📊 WEATHER DATA SUMMARY")
    print("="*70)

    print(f"\nTotal records: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # By metro
    print(f"\nRecords by metro:")
    for metro, count in df['metro_name'].value_counts().items():
        print(f"  {metro}: {count:,}")

    # By year
    if 'year' in df.columns:
        print(f"\nRecords by year:")
        for year, count in df['year'].value_counts().sort_index().items():
            print(f"  {year}: {count:,}")

    # Temperature stats
    if 'temp_avg_f' in df.columns:
        print(f"\nTemperature (°F):")
        print(f"  Mean: {df['temp_avg_f'].mean():.1f}°F")
        print(f"  Range: {df['temp_min_f'].min():.1f}°F to {df['temp_max_f'].max():.1f}°F")

    # Precipitation stats
    if 'precipitation_mm' in df.columns:
        total_precip = df['precipitation_mm'].sum()
        days_with_precip = (df['precipitation_mm'] > 0).sum()
        print(f"\nPrecipitation:")
        print(f"  Total: {total_precip:.1f} mm")
        print(f"  Days with precipitation: {days_with_precip:,} ({days_with_precip/len(df)*100:.1f}%)")

    # Missing data
    print(f"\nMissing data:")
    for col in ['precipitation_mm', 'temp_avg_f', 'wind_speed_mph']:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = missing / len(df) * 100
            print(f"  {col}: {missing:,} ({pct:.1f}%)")

    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA weather data for Texas metros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all metros for 2019-2024 (requires token)
  python download_noaa_weather.py --token YOUR_TOKEN

  # Specific date range
  python download_noaa_weather.py --token YOUR_TOKEN --start 2023-01-01 --end 2024-12-31

  # Specific counties only
  python download_noaa_weather.py --token YOUR_TOKEN --counties Travis,Harris,Bexar

Get API token at: https://www.ncdc.noaa.gov/cdo-web/token
        """
    )

    parser.add_argument('--token', type=str, required=False,
                       help='NOAA CDO API token (can also set NOAA_API_TOKEN in .env)')
    parser.add_argument('--start', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD, default: 2019-01-01)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD, default: 2024-12-31)')
    parser.add_argument('--counties', type=str,
                       help='Comma-separated county names (default: all)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Get API token (from args or environment)
    token = args.token or os.getenv('NOAA_API_TOKEN')
    if not token:
        print("❌ Error: NOAA API token required")
        print("   Either:")
        print("   1. Pass --token YOUR_TOKEN")
        print("   2. Set NOAA_API_TOKEN in .env file")
        print("\nGet token at: https://www.ncdc.noaa.gov/cdo-web/token")
        sys.exit(1)

    # Determine counties
    if args.counties:
        counties = [c.strip() for c in args.counties.split(',')]
    else:
        counties = list(TEXAS_METROS.keys())

    print("🌤️  NOAA Weather Data Downloader for Texas")
    print("="*70 + "\n")

    # Download data
    df = download_all_metros(
        counties,
        args.start,
        args.end,
        token,
        verbose=not args.quiet
    )

    if df.empty:
        print("❌ No data downloaded")
        sys.exit(1)

    # Clean data
    df = clean_weather_data(df, verbose=not args.quiet)

    # Summary
    print_summary(df)

    # Save
    if not args.no_save:
        save_data(df, args.start, args.end)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
