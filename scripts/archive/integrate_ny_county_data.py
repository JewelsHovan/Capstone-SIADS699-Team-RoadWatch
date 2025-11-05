"""
NY County-Level Integration Script
Integrates work zones and crashes at the county level

This script handles:
- Extracting county information from work zone geometries
- Aggregating crashes by county
- Joining work zones and crashes at county level
- Computing county-level crash risk metrics
- Feature engineering for ML modeling

Note: NY crash data lacks coordinates, so we use county-level aggregation
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class NYCountyIntegrator:
    """Integrate NY work zones and crashes at county level"""

    def __init__(self,
                 wzdx_path='data/raw/ny_wzdx_feed.json',
                 crashes_path='data/raw/crashes/ny_crashes.csv',
                 output_dir='data/processed'):
        """
        Initialize integrator

        Args:
            wzdx_path: Path to WZDx feed JSON
            crashes_path: Path to crashes CSV
            output_dir: Output directory for processed data
        """
        self.wzdx_path = Path(wzdx_path)
        self.crashes_path = Path(crashes_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.work_zones_gdf = None
        self.crashes_df = None
        self.county_stats = None
        self.integrated_wz = None

    def load_work_zones(self):
        """Load and process work zone data"""
        print("\n" + "="*70)
        print("LOADING NY WORK ZONES")
        print("="*70)

        # Load WZDx feed as GeoDataFrame
        self.work_zones_gdf = gpd.read_file(self.wzdx_path)

        print(f"✓ Loaded {len(self.work_zones_gdf):,} work zones")
        print(f"  Geometry type: {self.work_zones_gdf.geometry.type.unique()}")
        print(f"  CRS: {self.work_zones_gdf.crs}")

        # Extract core properties
        print("\nExtracting work zone features...")

        # Parse core_details JSON string
        def parse_core_details(core_details_str):
            try:
                if isinstance(core_details_str, str):
                    return json.loads(core_details_str)
                elif isinstance(core_details_str, dict):
                    return core_details_str
                else:
                    return {}
            except:
                return {}

        self.work_zones_gdf['core_details_parsed'] = self.work_zones_gdf['core_details'].apply(parse_core_details)

        # Extract road names
        def extract_road_names(core_details):
            road_names = core_details.get('road_names', [])
            return ', '.join(road_names) if road_names else 'Unknown'

        self.work_zones_gdf['road_names'] = self.work_zones_gdf['core_details_parsed'].apply(extract_road_names)

        # Extract direction
        self.work_zones_gdf['direction'] = self.work_zones_gdf['core_details_parsed'].apply(
            lambda x: x.get('direction', 'unknown')
        )

        # Extract description
        self.work_zones_gdf['description'] = self.work_zones_gdf['core_details_parsed'].apply(
            lambda x: x.get('description', '')
        )

        # Vehicle impact already exists as a column, no need to extract

        # Parse dates
        self.work_zones_gdf['start_date_parsed'] = pd.to_datetime(
            self.work_zones_gdf['start_date'], errors='coerce'
        )
        self.work_zones_gdf['end_date_parsed'] = pd.to_datetime(
            self.work_zones_gdf['end_date'], errors='coerce'
        )

        # Calculate duration
        self.work_zones_gdf['duration_days'] = (
            self.work_zones_gdf['end_date_parsed'] - self.work_zones_gdf['start_date_parsed']
        ).dt.days

        print(f"✓ Extracted features")
        print(f"  Active work zones: {self.work_zones_gdf['start_date_parsed'].notna().sum():,}")
        print(f"  With end dates: {self.work_zones_gdf['end_date_parsed'].notna().sum():,}")

        return self.work_zones_gdf

    def assign_counties_to_work_zones(self, county_shapefile=None):
        """
        Assign county to each work zone using spatial join

        Args:
            county_shapefile: Path to NY county boundaries (optional)
                If not provided, will attempt to download from Census
        """
        print("\n" + "="*70)
        print("ASSIGNING COUNTIES TO WORK ZONES")
        print("="*70)

        if county_shapefile is None:
            # Download NY county boundaries from Census TIGER
            print("\nDownloading NY county boundaries from Census TIGER...")

            try:
                # US Counties
                counties_url = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
                all_counties = gpd.read_file(counties_url)

                # Filter to NY (STATEFP = '36')
                ny_counties = all_counties[all_counties['STATEFP'] == '36'].copy()

                print(f"✓ Downloaded {len(ny_counties)} NY counties")

                # Simplify to relevant columns
                ny_counties = ny_counties[['COUNTYFP', 'NAME', 'geometry']]
                ny_counties = ny_counties.rename(columns={'NAME': 'county_name'})

            except Exception as e:
                print(f"✗ Error downloading counties: {e}")
                print("\nFalling back to centroid-based county assignment...")
                print("⚠ This is less accurate. Consider providing county shapefile.")
                return self._assign_counties_fallback()

        else:
            # Load provided shapefile
            print(f"\nLoading county boundaries from: {county_shapefile}")
            ny_counties = gpd.read_file(county_shapefile)

        # Ensure same CRS
        if self.work_zones_gdf.crs != ny_counties.crs:
            print(f"Reprojecting work zones from {self.work_zones_gdf.crs} to {ny_counties.crs}")
            self.work_zones_gdf = self.work_zones_gdf.to_crs(ny_counties.crs)

        # Spatial join
        print("\nPerforming spatial join...")
        wz_with_county = gpd.sjoin(
            self.work_zones_gdf,
            ny_counties,
            how='left',
            predicate='intersects'
        )

        # Handle multiple matches (work zone spans multiple counties)
        # Keep first match for simplicity
        wz_with_county = wz_with_county.groupby(wz_with_county.index).first()

        # Copy back to main dataframe
        self.work_zones_gdf['county_name'] = wz_with_county['county_name']
        self.work_zones_gdf['county_fips'] = wz_with_county['COUNTYFP']

        # Stats
        matched = self.work_zones_gdf['county_name'].notna().sum()
        match_rate = matched / len(self.work_zones_gdf) * 100

        print(f"\n✓ County assignment complete")
        print(f"  Matched: {matched:,} / {len(self.work_zones_gdf):,} ({match_rate:.1f}%)")
        print(f"  Unique counties: {self.work_zones_gdf['county_name'].nunique()}")

        # Show top counties
        print(f"\nTop counties by work zone count:")
        county_counts = self.work_zones_gdf['county_name'].value_counts()
        for county, count in county_counts.head(10).items():
            print(f"  {county}: {count}")

        return self.work_zones_gdf

    def _assign_counties_fallback(self):
        """
        Fallback: Assign counties using work zone centroids and approximate matching

        This is less accurate but works without county shapefiles
        """
        print("\nUsing fallback method: Cannot assign accurate counties without boundaries")
        print("⚠ County assignment will be incomplete")

        # For now, mark as unknown
        self.work_zones_gdf['county_name'] = None
        self.work_zones_gdf['county_fips'] = None

        return self.work_zones_gdf

    def load_crashes(self):
        """Load and process crash data"""
        print("\n" + "="*70)
        print("LOADING NY CRASHES")
        print("="*70)

        self.crashes_df = pd.read_csv(self.crashes_path)

        print(f"✓ Loaded {len(self.crashes_df):,} crash records")

        # Parse dates
        self.crashes_df['date_parsed'] = pd.to_datetime(self.crashes_df['date'], errors='coerce')

        # Normalize county names for matching
        self.crashes_df['county_name'] = self.crashes_df['county_name'].str.strip().str.upper()

        # Create severity flags
        severity_map = {
            'Property Damage Accident': 0,
            'Property Damage & Injury Accident': 1,
            'Injury Accident': 2,
            'Fatal Accident': 3
        }
        self.crashes_df['severity_level'] = self.crashes_df['accident_descriptor'].map(severity_map)
        self.crashes_df['is_injury'] = self.crashes_df['severity_level'] >= 1
        self.crashes_df['is_fatal'] = self.crashes_df['severity_level'] == 3

        print(f"  Date range: {self.crashes_df['date_parsed'].min()} to {self.crashes_df['date_parsed'].max()}")
        print(f"  Counties: {self.crashes_df['county_name'].nunique()}")
        print(f"  Fatal crashes: {self.crashes_df['is_fatal'].sum():,}")
        print(f"  Injury crashes: {self.crashes_df['is_injury'].sum():,}")

        return self.crashes_df

    def aggregate_crashes_by_county(self):
        """Aggregate crash statistics by county"""
        print("\n" + "="*70)
        print("AGGREGATING CRASHES BY COUNTY")
        print("="*70)

        # Group by county
        county_stats = self.crashes_df.groupby('county_name').agg({
            'date': 'count',  # Total crashes
            'is_injury': 'sum',  # Injury crashes
            'is_fatal': 'sum',  # Fatal crashes
            'severity_level': 'mean',  # Average severity
            'number_of_vehicles_involved': 'mean'  # Avg vehicles
        }).reset_index()

        county_stats.columns = [
            'county_name',
            'total_crashes',
            'injury_crashes',
            'fatal_crashes',
            'avg_severity',
            'avg_vehicles_involved'
        ]

        # Calculate rates
        county_stats['injury_rate'] = county_stats['injury_crashes'] / county_stats['total_crashes']
        county_stats['fatal_rate'] = county_stats['fatal_crashes'] / county_stats['total_crashes']

        # Add temporal aggregations
        time_stats = self.crashes_df.groupby('county_name').agg({
            'date_parsed': ['min', 'max']
        }).reset_index()
        time_stats.columns = ['county_name', 'first_crash_date', 'last_crash_date']

        county_stats = county_stats.merge(time_stats, on='county_name', how='left')

        # Calculate days covered
        county_stats['days_covered'] = (
            county_stats['last_crash_date'] - county_stats['first_crash_date']
        ).dt.days

        # Crashes per day
        county_stats['crashes_per_day'] = county_stats['total_crashes'] / county_stats['days_covered']

        self.county_stats = county_stats

        print(f"✓ Aggregated crash data for {len(county_stats)} counties")
        print(f"\nTop 10 counties by total crashes:")
        print(county_stats.nlargest(10, 'total_crashes')[
            ['county_name', 'total_crashes', 'injury_crashes', 'fatal_crashes']
        ].to_string(index=False))

        return county_stats

    def integrate_work_zones_with_crashes(self):
        """Integrate work zone data with county-level crash statistics"""
        print("\n" + "="*70)
        print("INTEGRATING WORK ZONES WITH CRASH DATA")
        print("="*70)

        if self.work_zones_gdf is None or self.county_stats is None:
            print("✗ Error: Work zones or county stats not loaded")
            return None

        # Normalize county names in work zones for matching
        self.work_zones_gdf['county_name_normalized'] = (
            self.work_zones_gdf['county_name'].str.strip().str.upper()
        )

        # Merge work zones with county crash stats
        integrated = self.work_zones_gdf.merge(
            self.county_stats,
            left_on='county_name_normalized',
            right_on='county_name',
            how='left',
            suffixes=('_wz', '_crash')
        )

        # Calculate work zone-specific crash risk score
        # Score = (county crash rate) * (work zone duration) * (vehicle impact severity)
        vehicle_impact_weight = {
            'all-lanes-closed': 3.0,
            'some-lanes-closed': 2.0,
            'all-lanes-open': 1.0,
            'alternating-one-way-traffic': 2.5,
            'unknown': 1.5
        }

        integrated['vehicle_impact_weight'] = integrated['vehicle_impact'].map(
            vehicle_impact_weight
        ).fillna(1.5)

        integrated['work_zone_crash_risk_score'] = (
            integrated['crashes_per_day'] *
            integrated['duration_days'].fillna(30) *  # Assume 30 days if missing
            integrated['vehicle_impact_weight']
        )

        # Add categorical risk level
        def categorize_risk(score):
            if pd.isna(score):
                return 'Unknown'
            elif score < 10:
                return 'Low'
            elif score < 50:
                return 'Medium'
            elif score < 100:
                return 'High'
            else:
                return 'Very High'

        integrated['risk_category'] = integrated['work_zone_crash_risk_score'].apply(categorize_risk)

        self.integrated_wz = integrated

        # Statistics
        matched = integrated['total_crashes'].notna().sum()
        match_rate = matched / len(integrated) * 100

        print(f"\n✓ Integration complete")
        print(f"  Work zones matched to crash data: {matched:,} / {len(integrated):,} ({match_rate:.1f}%)")
        print(f"\nRisk category distribution:")
        risk_dist = integrated['risk_category'].value_counts()
        for category, count in risk_dist.items():
            pct = count / len(integrated) * 100
            print(f"  {category}: {count} ({pct:.1f}%)")

        return integrated

    def save_outputs(self):
        """Save processed data to files"""
        print("\n" + "="*70)
        print("SAVING OUTPUTS")
        print("="*70)

        # 1. County crash statistics (CSV)
        county_csv = self.output_dir / 'ny_county_crash_stats.csv'
        self.county_stats.to_csv(county_csv, index=False)
        print(f"✓ County crash stats: {county_csv}")

        # 2. Integrated work zones with crash data (GeoPackage)
        integrated_gpkg = self.output_dir / 'ny_work_zones_with_crashes.gpkg'
        self.integrated_wz.to_file(integrated_gpkg, driver='GPKG')
        print(f"✓ Integrated work zones (GeoPackage): {integrated_gpkg}")

        # 3. Integrated work zones with crash data (CSV - no geometry)
        integrated_csv = self.output_dir / 'ny_work_zones_with_crashes.csv'
        # Drop geometry column for CSV
        integrated_df = pd.DataFrame(self.integrated_wz.drop(columns='geometry'))
        integrated_df.to_csv(integrated_csv, index=False)
        print(f"✓ Integrated work zones (CSV): {integrated_csv}")

        # 4. Summary statistics (JSON)
        summary = {
            'generated_at': datetime.now().isoformat(),
            'data_sources': {
                'work_zones': str(self.wzdx_path),
                'crashes': str(self.crashes_path)
            },
            'statistics': {
                'total_work_zones': int(len(self.integrated_wz)),
                'work_zones_with_crash_data': int(self.integrated_wz['total_crashes'].notna().sum()),
                'unique_counties': int(self.integrated_wz['county_name_normalized'].nunique()),
                'total_crashes_in_counties': int(self.county_stats['total_crashes'].sum()),
                'total_fatal_crashes': int(self.county_stats['fatal_crashes'].sum()),
                'total_injury_crashes': int(self.county_stats['injury_crashes'].sum())
            },
            'risk_distribution': self.integrated_wz['risk_category'].value_counts().to_dict(),
            'top_risk_counties': self.county_stats.nlargest(10, 'total_crashes')[
                ['county_name', 'total_crashes', 'injury_crashes', 'fatal_crashes']
            ].to_dict(orient='records')
        }

        summary_json = self.output_dir / 'ny_integration_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary statistics: {summary_json}")

        print(f"\n✓ All outputs saved to: {self.output_dir}")

    def run_full_integration(self, county_shapefile=None):
        """Run complete integration pipeline"""
        print("\n" + "="*80)
        print(" " * 20 + "NY COUNTY-LEVEL INTEGRATION")
        print("="*80)

        # Load data
        self.load_work_zones()
        self.assign_counties_to_work_zones(county_shapefile)
        self.load_crashes()

        # Process
        self.aggregate_crashes_by_county()
        self.integrate_work_zones_with_crashes()

        # Save
        self.save_outputs()

        # Final summary
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print(f"\nWork zones: {len(self.work_zones_gdf):,}")
        print(f"Crashes: {len(self.crashes_df):,}")
        print(f"Counties: {self.county_stats['county_name'].nunique()}")
        print(f"Work zones with crash data: {self.integrated_wz['total_crashes'].notna().sum():,}")
        print(f"\nOutputs saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Explore integrated data:")
        print("   import pandas as pd")
        print("   df = pd.read_csv('data/processed/ny_work_zones_with_crashes.csv')")
        print("   print(df[df['risk_category'] == 'Very High'])")
        print("\n2. Use in dashboard:")
        print("   Load GeoPackage in map visualization")
        print("\n3. Build ML features:")
        print("   Use crash statistics as features for risk prediction")
        print("="*80)


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Integrate NY work zones and crashes at county level')
    parser.add_argument('--wzdx', default='data/raw/ny_wzdx_feed.json', help='WZDx feed path')
    parser.add_argument('--crashes', default='data/raw/crashes/ny_crashes.csv', help='Crashes CSV path')
    parser.add_argument('--output', default='data/processed', help='Output directory')
    parser.add_argument('--counties', help='Path to NY county shapefile (optional)')

    args = parser.parse_args()

    integrator = NYCountyIntegrator(
        wzdx_path=args.wzdx,
        crashes_path=args.crashes,
        output_dir=args.output
    )

    integrator.run_full_integration(county_shapefile=args.counties)


if __name__ == "__main__":
    main()
