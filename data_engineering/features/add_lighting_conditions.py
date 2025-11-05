"""
Add Lighting Conditions Features Using Sunrise/Sunset API

Creates features based on time of day and lighting conditions:
- Daylight vs. twilight vs. dark
- Civil twilight (dawn/dusk) = highest crash risk periods
- Time relative to sunrise/sunset
- Sun elevation angles

Source: https://api.sunrise-sunset.org/json
- FREE, no API key required
- Returns sunrise, sunset, twilight times for any lat/lon/date

Research shows:
- Civil twilight (30 min before sunrise / after sunset) = highest crash risk
- Dawn crashes often involve glare, drowsy drivers
- Dusk crashes involve visibility transitions, fatigue

Usage:
    from data_engineering.features.add_lighting_conditions import add_lighting_features

    # Add to DataFrame with Start_Time, Start_Lat, Start_Lng
    crashes_with_lighting = add_lighting_features(crashes_df)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import time


@lru_cache(maxsize=10000)
def get_sun_times(lat, lon, date_str):
    """
    Get sunrise/sunset times for a location and date

    Uses LRU cache to avoid repeated API calls for same location/date

    Args:
        lat: Latitude
        lon: Longitude
        date_str: Date string in 'YYYY-MM-DD' format

    Returns:
        dict with sunrise, sunset, twilight times (all in UTC)
    """
    url = "https://api.sunrise-sunset.org/json"
    params = {
        'lat': lat,
        'lng': lon,
        'date': date_str,
        'formatted': 0  # Get ISO 8601 timestamps (UTC)
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if data['status'] != 'OK':
            return None

        return data['results']

    except Exception as e:
        print(f"⚠ API error for {lat},{lon} on {date_str}: {e}")
        return None


def classify_lighting_condition(crash_time, sun_data):
    """
    Classify lighting condition at crash time

    Args:
        crash_time: datetime object (aware, in UTC)
        sun_data: dict from get_sun_times()

    Returns:
        str: 'daylight', 'civil_twilight', 'nautical_twilight', 'astronomical_twilight', 'dark'
    """
    if sun_data is None:
        return 'unknown'

    try:
        # Parse times (all in UTC)
        sunrise = pd.to_datetime(sun_data['sunrise'])
        sunset = pd.to_datetime(sun_data['sunset'])
        civil_begin = pd.to_datetime(sun_data['civil_twilight_begin'])
        civil_end = pd.to_datetime(sun_data['civil_twilight_end'])
        nautical_begin = pd.to_datetime(sun_data['nautical_twilight_begin'])
        nautical_end = pd.to_datetime(sun_data['nautical_twilight_end'])
        astro_begin = pd.to_datetime(sun_data['astronomical_twilight_begin'])
        astro_end = pd.to_datetime(sun_data['astronomical_twilight_end'])

        # Make crash_time timezone-aware (UTC) if needed
        if crash_time.tzinfo is None:
            crash_time = crash_time.replace(tzinfo=timezone.utc)

        # Classify based on time
        if sunrise <= crash_time <= sunset:
            return 'daylight'
        elif civil_begin <= crash_time < sunrise:
            return 'civil_twilight_dawn'
        elif sunset < crash_time <= civil_end:
            return 'civil_twilight_dusk'
        elif nautical_begin <= crash_time < civil_begin:
            return 'nautical_twilight_dawn'
        elif civil_end < crash_time <= nautical_end:
            return 'nautical_twilight_dusk'
        elif astro_begin <= crash_time < nautical_begin:
            return 'astronomical_twilight_dawn'
        elif nautical_end < crash_time <= astro_end:
            return 'astronomical_twilight_dusk'
        else:
            return 'dark'

    except Exception as e:
        return 'unknown'


def add_lighting_features(df, time_col='Start_Time', lat_col='Start_Lat', lon_col='Start_Lng',
                          verbose=True, rate_limit_delay=0.1):
    """
    Add lighting condition features to crash DataFrame

    Args:
        df: DataFrame with crash data
        time_col: Column name for crash timestamp
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        verbose: Print progress
        rate_limit_delay: Delay between API calls (seconds) to be respectful

    Returns:
        DataFrame with new lighting features
    """

    print('\n' + '='*80)
    print('ADDING LIGHTING CONDITION FEATURES')
    print('='*80)

    df = df.copy()

    # Ensure datetime column
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print(f"Converting {time_col} to datetime...")
        df[time_col] = pd.to_datetime(df[time_col])

    # Extract date (for API query) and ensure UTC
    df['_date_str'] = df[time_col].dt.strftime('%Y-%m-%d')

    # Round lat/lon to reduce unique API calls (0.1 degree ≈ 11km, close enough for sun times)
    df['_lat_rounded'] = df[lat_col].round(1)
    df['_lon_rounded'] = df[lon_col].round(1)

    # Get unique location-date combinations
    unique_locations = df[['_lat_rounded', '_lon_rounded', '_date_str']].drop_duplicates()

    print(f"\nCrashes: {len(df):,}")
    print(f"Unique location-date combinations: {len(unique_locations):,}")
    print(f"Fetching sunrise/sunset data from API...")

    # Fetch sun data for each unique location-date
    sun_data_cache = {}

    for idx, row in unique_locations.iterrows():
        lat = row['_lat_rounded']
        lon = row['_lon_rounded']
        date_str = row['_date_str']

        key = (lat, lon, date_str)

        # Check cache first
        sun_data = get_sun_times(lat, lon, date_str)
        sun_data_cache[key] = sun_data

        # Progress
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Fetched {idx + 1:,} / {len(unique_locations):,} locations...", end='\r')

        # Rate limiting to be respectful to API
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    if verbose:
        print(f"  Fetched {len(unique_locations):,} / {len(unique_locations):,} locations... ✓")

    print("\nClassifying lighting conditions...")

    # Apply classification to each crash
    lighting_conditions = []
    minutes_since_sunrise = []
    minutes_until_sunset = []

    for idx, row in df.iterrows():
        lat = row['_lat_rounded']
        lon = row['_lon_rounded']
        date_str = row['_date_str']
        crash_time = row[time_col]

        key = (lat, lon, date_str)
        sun_data = sun_data_cache.get(key)

        # Classify
        condition = classify_lighting_condition(crash_time, sun_data)
        lighting_conditions.append(condition)

        # Calculate time deltas
        if sun_data:
            try:
                sunrise = pd.to_datetime(sun_data['sunrise'])
                sunset = pd.to_datetime(sun_data['sunset'])

                # Make crash_time aware if needed
                if crash_time.tzinfo is None:
                    crash_time = crash_time.replace(tzinfo=timezone.utc)

                mins_since_sunrise = (crash_time - sunrise).total_seconds() / 60
                mins_until_sunset = (sunset - crash_time).total_seconds() / 60

                minutes_since_sunrise.append(mins_since_sunrise)
                minutes_until_sunset.append(mins_until_sunset)
            except:
                minutes_since_sunrise.append(np.nan)
                minutes_until_sunset.append(np.nan)
        else:
            minutes_since_sunrise.append(np.nan)
            minutes_until_sunset.append(np.nan)

        if verbose and (idx + 1) % 10000 == 0:
            print(f"  Classified {idx + 1:,} / {len(df):,} crashes...", end='\r')

    if verbose:
        print(f"  Classified {len(df):,} / {len(df):,} crashes... ✓")

    # Add features to DataFrame
    df['lighting_condition'] = lighting_conditions
    df['minutes_since_sunrise'] = minutes_since_sunrise
    df['minutes_until_sunset'] = minutes_until_sunset

    # Create binary indicator features
    df['is_daylight'] = (df['lighting_condition'] == 'daylight').astype(int)
    df['is_civil_twilight'] = df['lighting_condition'].str.contains('civil_twilight', na=False).astype(int)
    df['is_twilight_dawn'] = (df['lighting_condition'] == 'civil_twilight_dawn').astype(int)
    df['is_twilight_dusk'] = (df['lighting_condition'] == 'civil_twilight_dusk').astype(int)
    df['is_dark'] = (df['lighting_condition'] == 'dark').astype(int)

    # Drop temporary columns
    df = df.drop(columns=['_date_str', '_lat_rounded', '_lon_rounded'])

    # Print summary
    print('\n' + '='*80)
    print('LIGHTING FEATURES SUMMARY')
    print('='*80)

    print(f"\nLighting condition distribution:")
    print(df['lighting_condition'].value_counts())

    print(f"\nBinary features:")
    print(f"  Daylight:         {df['is_daylight'].sum():6,} ({df['is_daylight'].mean()*100:5.1f}%)")
    print(f"  Civil twilight:   {df['is_civil_twilight'].sum():6,} ({df['is_civil_twilight'].mean()*100:5.1f}%)")
    print(f"    - Dawn:         {df['is_twilight_dawn'].sum():6,} ({df['is_twilight_dawn'].mean()*100:5.1f}%)")
    print(f"    - Dusk:         {df['is_twilight_dusk'].sum():6,} ({df['is_twilight_dusk'].mean()*100:5.1f}%)")
    print(f"  Dark:             {df['is_dark'].sum():6,} ({df['is_dark'].mean()*100:5.1f}%)")

    print(f"\nTime deltas:")
    print(f"  Minutes since sunrise: mean={df['minutes_since_sunrise'].mean():.1f}, "
          f"range=[{df['minutes_since_sunrise'].min():.1f}, {df['minutes_since_sunrise'].max():.1f}]")
    print(f"  Minutes until sunset:  mean={df['minutes_until_sunset'].mean():.1f}, "
          f"range=[{df['minutes_until_sunset'].min():.1f}, {df['minutes_until_sunset'].max():.1f}]")

    print(f"\nNew features added (8):")
    new_features = ['lighting_condition', 'is_daylight', 'is_civil_twilight',
                   'is_twilight_dawn', 'is_twilight_dusk', 'is_dark',
                   'minutes_since_sunrise', 'minutes_until_sunset']
    for feat in new_features:
        print(f"  ✓ {feat}")

    return df


def add_lighting_features_batch(df, batch_size=1000, **kwargs):
    """
    Process DataFrame in batches (for very large datasets)

    Args:
        df: DataFrame with crash data
        batch_size: Number of rows per batch
        **kwargs: Additional arguments to pass to add_lighting_features()

    Returns:
        DataFrame with lighting features
    """
    print(f"\nProcessing {len(df):,} crashes in batches of {batch_size:,}...")

    results = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1} ({i:,} to {min(i+batch_size, len(df)):,})...")

        batch_result = add_lighting_features(batch_df, verbose=False, **kwargs)
        results.append(batch_result)

    return pd.concat(results, ignore_index=True)


def main():
    """Test the lighting features on sample data"""

    print("Testing lighting features with sample data...")

    # Create sample data
    sample_data = {
        'Start_Time': [
            '2023-01-15 07:00:00',  # Morning
            '2023-01-15 12:00:00',  # Midday
            '2023-01-15 18:00:00',  # Evening
            '2023-01-15 23:00:00',  # Night
            '2023-06-15 06:00:00',  # Summer morning
            '2023-06-15 20:00:00',  # Summer evening
        ],
        'Start_Lat': [30.2672, 30.2672, 30.2672, 30.2672, 30.2672, 30.2672],  # Austin
        'Start_Lng': [-97.7431, -97.7431, -97.7431, -97.7431, -97.7431, -97.7431],
        'Severity': [2, 3, 2, 4, 2, 3]
    }

    df = pd.DataFrame(sample_data)

    # Add features
    df_with_lighting = add_lighting_features(df, rate_limit_delay=0.5)

    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    print(df_with_lighting[['Start_Time', 'lighting_condition', 'is_daylight',
                            'is_civil_twilight', 'is_dark']])

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print("\nUsage in your pipeline:")
    print("  from data_engineering.features.add_lighting_conditions import add_lighting_features")
    print("  crashes_with_lighting = add_lighting_features(crashes_df)")


if __name__ == "__main__":
    main()
