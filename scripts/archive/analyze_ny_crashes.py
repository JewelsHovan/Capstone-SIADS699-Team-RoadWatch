"""
Analyze New York Crash Data
Comprehensive analysis of NY crash features for ML modeling

This script analyzes:
- Feature distributions and completeness
- Temporal patterns (time, day, seasonality)
- Weather and road condition patterns
- Accident severity and types
- County-level crash rates
- Feature engineering opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json


class NYCrashAnalyzer:
    """Analyze NY crash data for insights and ML feature engineering"""

    def __init__(self, data_path='data/raw/crashes/ny_crashes.csv'):
        """
        Initialize analyzer

        Args:
            data_path: Path to NY crashes CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.analysis_results = {}

    def load_data(self):
        """Load and prepare crash data"""
        print("\n" + "="*70)
        print("LOADING NY CRASH DATA")
        print("="*70)

        self.df = pd.read_csv(self.data_path)

        # Parse dates
        self.df['date_parsed'] = pd.to_datetime(self.df['date'], errors='coerce')

        # Parse time to hour
        self.df['hour'] = pd.to_datetime(self.df['time'], format='%H:%M', errors='coerce').dt.hour

        # Create severity flag
        severity_map = {
            'Property Damage Accident': 0,
            'Property Damage & Injury Accident': 1,
            'Injury Accident': 2,
            'Fatal Accident': 3
        }
        self.df['severity_level'] = self.df['accident_descriptor'].map(severity_map)
        self.df['is_injury'] = self.df['severity_level'] >= 1
        self.df['is_fatal'] = self.df['severity_level'] == 3

        # Create time of day categories
        def categorize_time(hour):
            if pd.isna(hour):
                return 'Unknown'
            elif 6 <= hour < 10:
                return 'Morning Rush (6-10am)'
            elif 10 <= hour < 16:
                return 'Midday (10am-4pm)'
            elif 16 <= hour < 20:
                return 'Evening Rush (4-8pm)'
            elif 20 <= hour < 24 or 0 <= hour < 6:
                return 'Night (8pm-6am)'
            else:
                return 'Unknown'

        self.df['time_category'] = self.df['hour'].apply(categorize_time)

        print(f"✓ Loaded {len(self.df):,} crash records")
        print(f"  Date range: {self.df['date_parsed'].min()} to {self.df['date_parsed'].max()}")
        print(f"  Columns: {len(self.df.columns)}")

        return self.df

    def analyze_completeness(self):
        """Analyze data completeness and missing values"""
        print("\n" + "="*70)
        print("DATA COMPLETENESS ANALYSIS")
        print("="*70)

        total = len(self.df)
        completeness = {}

        print(f"\nTotal records: {total:,}\n")
        print(f"{'Field':<40} {'Non-Null':>12} {'Complete':>10}")
        print("-" * 70)

        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            pct = non_null / total * 100
            completeness[col] = {'count': non_null, 'percent': pct}

            if pct < 100:
                print(f"{col:<40} {non_null:>12,} {pct:>9.1f}%")

        # Store results
        self.analysis_results['completeness'] = completeness

        # Flag potentially useful fields
        print("\n" + "="*70)
        print("KEY FIELDS FOR ML")
        print("="*70)

        key_fields = [
            'year', 'day_of_week', 'time', 'weather_conditions',
            'lighting_conditions', 'road_surface_conditions',
            'road_descriptor', 'traffic_control_device',
            'collision_type_descriptor', 'county_name',
            'number_of_vehicles_involved', 'accident_descriptor'
        ]

        for field in key_fields:
            if field in completeness:
                pct = completeness[field]['percent']
                status = "✓" if pct > 95 else "⚠" if pct > 80 else "✗"
                print(f"{status} {field:<40} {pct:>6.1f}% complete")

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in crashes"""
        print("\n" + "="*70)
        print("TEMPORAL PATTERNS")
        print("="*70)

        # Year distribution
        print("\n1. YEARLY DISTRIBUTION:")
        print("-" * 50)
        year_counts = self.df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count:>8,} crashes")

        # Day of week
        print("\n2. DAY OF WEEK DISTRIBUTION:")
        print("-" * 50)
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = self.df['day_of_week'].value_counts()
        for day in dow_order:
            if day in dow_counts.index:
                count = dow_counts[day]
                pct = count / len(self.df) * 100
                print(f"  {day:<12}: {count:>8,} ({pct:>5.1f}%)")

        # Time of day
        print("\n3. TIME OF DAY DISTRIBUTION:")
        print("-" * 50)
        time_order = ['Morning Rush (6-10am)', 'Midday (10am-4pm)',
                     'Evening Rush (4-8pm)', 'Night (8pm-6am)', 'Unknown']
        time_counts = self.df['time_category'].value_counts()
        for time_cat in time_order:
            if time_cat in time_counts.index:
                count = time_counts[time_cat]
                pct = count / len(self.df) * 100
                print(f"  {time_cat:<25}: {count:>8,} ({pct:>5.1f}%)")

        # Store results
        self.analysis_results['temporal'] = {
            'by_year': year_counts.to_dict(),
            'by_day': dow_counts.to_dict(),
            'by_time': time_counts.to_dict()
        }

    def analyze_severity(self):
        """Analyze crash severity patterns"""
        print("\n" + "="*70)
        print("CRASH SEVERITY ANALYSIS")
        print("="*70)

        # Overall severity
        print("\n1. ACCIDENT SEVERITY:")
        print("-" * 50)
        severity_counts = self.df['accident_descriptor'].value_counts()
        for severity, count in severity_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {severity:<40}: {count:>8,} ({pct:>5.1f}%)")

        # Calculate rates
        injury_rate = self.df['is_injury'].sum() / len(self.df) * 100
        fatal_rate = self.df['is_fatal'].sum() / len(self.df) * 100

        print(f"\n  Total with injuries: {self.df['is_injury'].sum():,} ({injury_rate:.1f}%)")
        print(f"  Total fatal: {self.df['is_fatal'].sum():,} ({fatal_rate:.2f}%)")

        # Vehicles involved
        print("\n2. VEHICLES INVOLVED:")
        print("-" * 50)
        vehicle_counts = self.df['number_of_vehicles_involved'].value_counts().sort_index()
        for num_vehicles, count in vehicle_counts.head(10).items():
            pct = count / len(self.df) * 100
            print(f"  {num_vehicles} vehicle(s): {count:>8,} ({pct:>5.1f}%)")

        # Store results
        self.analysis_results['severity'] = {
            'by_type': severity_counts.to_dict(),
            'injury_rate': injury_rate,
            'fatal_rate': fatal_rate,
            'vehicles_involved': vehicle_counts.to_dict()
        }

    def analyze_environmental_conditions(self):
        """Analyze weather, lighting, and road conditions"""
        print("\n" + "="*70)
        print("ENVIRONMENTAL CONDITIONS")
        print("="*70)

        # Weather
        print("\n1. WEATHER CONDITIONS:")
        print("-" * 50)
        weather_counts = self.df['weather_conditions'].value_counts()
        for weather, count in weather_counts.head(10).items():
            pct = count / len(self.df) * 100
            print(f"  {weather:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Lighting
        print("\n2. LIGHTING CONDITIONS:")
        print("-" * 50)
        lighting_counts = self.df['lighting_conditions'].value_counts()
        for lighting, count in lighting_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {lighting:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Road surface
        print("\n3. ROAD SURFACE CONDITIONS:")
        print("-" * 50)
        surface_counts = self.df['road_surface_conditions'].value_counts()
        for surface, count in surface_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {surface:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Road descriptor
        print("\n4. ROAD CHARACTERISTICS:")
        print("-" * 50)
        road_counts = self.df['road_descriptor'].value_counts()
        for road, count in road_counts.head(10).items():
            pct = count / len(self.df) * 100
            print(f"  {road:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Store results
        self.analysis_results['environmental'] = {
            'weather': weather_counts.to_dict(),
            'lighting': lighting_counts.to_dict(),
            'road_surface': surface_counts.to_dict(),
            'road_descriptor': road_counts.to_dict()
        }

    def analyze_collision_types(self):
        """Analyze collision types and event descriptors"""
        print("\n" + "="*70)
        print("COLLISION CHARACTERISTICS")
        print("="*70)

        # Collision type
        print("\n1. COLLISION TYPES:")
        print("-" * 50)
        collision_counts = self.df['collision_type_descriptor'].value_counts()
        for collision, count in collision_counts.head(15).items():
            pct = count / len(self.df) * 100
            print(f"  {collision:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Event descriptor
        print("\n2. EVENT DESCRIPTORS (Top 15):")
        print("-" * 50)
        event_counts = self.df['event_descriptor'].value_counts()
        for event, count in event_counts.head(15).items():
            pct = count / len(self.df) * 100
            print(f"  {event:<40}: {count:>8,} ({pct:>5.1f}%)")

        # Traffic control
        print("\n3. TRAFFIC CONTROL DEVICES:")
        print("-" * 50)
        control_counts = self.df['traffic_control_device'].value_counts()
        for control, count in control_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {control:<30}: {count:>8,} ({pct:>5.1f}%)")

        # Store results
        self.analysis_results['collision'] = {
            'types': collision_counts.to_dict(),
            'events': event_counts.to_dict(),
            'traffic_control': control_counts.to_dict()
        }

    def analyze_geographic_distribution(self):
        """Analyze crash distribution by county and municipality"""
        print("\n" + "="*70)
        print("GEOGRAPHIC DISTRIBUTION")
        print("="*70)

        # County distribution
        print("\n1. TOP 20 COUNTIES BY CRASH COUNT:")
        print("-" * 50)
        county_counts = self.df['county_name'].value_counts()

        print(f"  Total counties: {self.df['county_name'].nunique()}")
        print(f"\n  {'County':<20} {'Crashes':>10} {'% of Total':>12}")
        print("  " + "-" * 45)

        for county, count in county_counts.head(20).items():
            pct = count / len(self.df) * 100
            print(f"  {county:<20} {count:>10,} {pct:>11.1f}%")

        # Municipality distribution
        print("\n2. TOP 20 MUNICIPALITIES BY CRASH COUNT:")
        print("-" * 50)
        muni_counts = self.df['municipality'].value_counts()

        print(f"  Total municipalities: {self.df['municipality'].nunique()}")
        print(f"\n  {'Municipality':<20} {'Crashes':>10} {'% of Total':>12}")
        print("  " + "-" * 45)

        for muni, count in muni_counts.head(20).items():
            pct = count / len(self.df) * 100
            print(f"  {muni:<20} {count:>10,} {pct:>11.1f}%")

        # Store results
        self.analysis_results['geographic'] = {
            'by_county': county_counts.to_dict(),
            'by_municipality': muni_counts.to_dict(),
            'unique_counties': int(self.df['county_name'].nunique()),
            'unique_municipalities': int(self.df['municipality'].nunique())
        }

    def identify_high_risk_patterns(self):
        """Identify high-risk patterns for crashes"""
        print("\n" + "="*70)
        print("HIGH-RISK PATTERN ANALYSIS")
        print("="*70)

        # Fatal crashes by condition
        fatal_df = self.df[self.df['is_fatal']]

        if len(fatal_df) > 0:
            print(f"\n1. FATAL CRASH CHARACTERISTICS ({len(fatal_df):,} fatal crashes):")
            print("-" * 50)

            # Weather in fatal crashes
            print("\n  Weather conditions in fatal crashes:")
            fatal_weather = fatal_df['weather_conditions'].value_counts()
            total_fatal = len(fatal_df)
            for weather, count in fatal_weather.head(5).items():
                pct = count / total_fatal * 100
                print(f"    {weather:<25}: {count:>6,} ({pct:>5.1f}%)")

            # Lighting in fatal crashes
            print("\n  Lighting conditions in fatal crashes:")
            fatal_lighting = fatal_df['lighting_conditions'].value_counts()
            for lighting, count in fatal_lighting.head(5).items():
                pct = count / total_fatal * 100
                print(f"    {lighting:<25}: {count:>6,} ({pct:>5.1f}%)")

            # Time of day in fatal crashes
            print("\n  Time of day in fatal crashes:")
            fatal_time = fatal_df['time_category'].value_counts()
            for time_cat, count in fatal_time.items():
                pct = count / total_fatal * 100
                print(f"    {time_cat:<25}: {count:>6,} ({pct:>5.1f}%)")

        # Injury crashes
        injury_df = self.df[self.df['is_injury']]
        print(f"\n2. INJURY CRASH PATTERNS ({len(injury_df):,} injury crashes):")
        print("-" * 50)

        # Collision types in injury crashes
        print("\n  Top collision types in injury crashes:")
        injury_collision = injury_df['collision_type_descriptor'].value_counts()
        total_injury = len(injury_df)
        for collision, count in injury_collision.head(5).items():
            pct = count / total_injury * 100
            print(f"    {collision:<25}: {count:>6,} ({pct:>5.1f}%)")

        # Store results
        self.analysis_results['high_risk'] = {
            'fatal_count': int(len(fatal_df)),
            'injury_count': int(len(injury_df)),
            'fatal_weather': fatal_weather.to_dict() if len(fatal_df) > 0 else {},
            'fatal_lighting': fatal_lighting.to_dict() if len(fatal_df) > 0 else {},
            'injury_collision_types': injury_collision.to_dict()
        }

    def generate_ml_recommendations(self):
        """Generate recommendations for ML feature engineering"""
        print("\n" + "="*70)
        print("ML FEATURE ENGINEERING RECOMMENDATIONS")
        print("="*70)

        recommendations = []

        print("\n1. TEMPORAL FEATURES:")
        print("-" * 50)
        temporal_features = [
            "✓ hour (0-23) - Already extracted",
            "✓ day_of_week - Available",
            "✓ time_category (rush hour, night, etc.) - Already created",
            "- month (1-12) - Can extract from date",
            "- is_weekend (boolean) - Can derive from day_of_week",
            "- is_holiday (boolean) - Requires holiday calendar",
            "- season (winter, spring, summer, fall) - Can derive from date"
        ]
        for feature in temporal_features:
            print(f"  {feature}")
        recommendations.extend(temporal_features)

        print("\n2. ENVIRONMENTAL FEATURES:")
        print("-" * 50)
        env_features = [
            "✓ weather_conditions - Available (categorical)",
            "✓ lighting_conditions - Available (categorical)",
            "✓ road_surface_conditions - Available (categorical)",
            "✓ road_descriptor - Available (categorical)",
            "- is_adverse_weather (boolean) - Can derive (rain, snow, fog)",
            "- is_dark (boolean) - Can derive from lighting_conditions",
            "- is_poor_road (boolean) - Can derive (wet, ice, snow)"
        ]
        for feature in env_features:
            print(f"  {feature}")
        recommendations.extend(env_features)

        print("\n3. CRASH CHARACTERISTICS:")
        print("-" * 50)
        crash_features = [
            "✓ collision_type_descriptor - Available (categorical)",
            "✓ traffic_control_device - Available (categorical)",
            "✓ number_of_vehicles_involved - Available (numeric)",
            "✓ severity_level (0-3) - Already created",
            "- is_multi_vehicle (boolean) - Can derive (>1 vehicle)",
            "- has_traffic_control (boolean) - Can derive"
        ]
        for feature in crash_features:
            print(f"  {feature}")
        recommendations.extend(crash_features)

        print("\n4. LOCATION FEATURES:")
        print("-" * 50)
        location_features = [
            "✓ county_name - Available (categorical)",
            "✓ municipality - Available (categorical)",
            "- county_crash_rate - Can calculate (crashes per population)",
            "- is_urban (boolean) - Can derive from municipality type",
            "⚠ lat/lon - NOT AVAILABLE (major limitation)",
            "⚠ distance_to_work_zone - NOT POSSIBLE without coordinates"
        ]
        for feature in location_features:
            print(f"  {feature}")
        recommendations.extend(location_features)

        print("\n5. TARGET VARIABLES FOR ML:")
        print("-" * 50)
        targets = [
            "✓ is_injury (binary classification) - Predict if crash causes injury",
            "✓ is_fatal (binary classification) - Predict if crash is fatal",
            "✓ severity_level (multi-class) - Predict severity (0-3)",
            "- crash_risk_score (regression) - Composite risk metric"
        ]
        for target in targets:
            print(f"  {target}")

        print("\n6. RECOMMENDED ENCODING STRATEGIES:")
        print("-" * 50)
        encoding_tips = [
            "- One-hot encode: weather, lighting, road_surface, collision_type",
            "- Label encode: severity_level (ordinal)",
            "- Target encode: county_name, municipality (high cardinality)",
            "- Binary flags: is_weekend, is_dark, is_adverse_weather",
            "- Time cyclical: hour -> sin/cos features for 24-hour cycle"
        ]
        for tip in encoding_tips:
            print(f"  {tip}")

        self.analysis_results['ml_recommendations'] = {
            'temporal': temporal_features,
            'environmental': env_features,
            'crash_characteristics': crash_features,
            'location': location_features,
            'targets': targets,
            'encoding': encoding_tips
        }

    def save_analysis_report(self, output_path='outputs/ny_crash_analysis.json'):
        """Save analysis results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        # Recursively convert all values
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_types(d)

        analysis_json = recursive_convert(self.analysis_results)

        with open(output_path, 'w') as f:
            json.dump(analysis_json, f, indent=2, default=str)

        print(f"\n✓ Analysis report saved to: {output_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print(" " * 25 + "NY CRASH DATA ANALYSIS")
        print("="*80)

        # Load data
        self.load_data()

        # Run all analyses
        self.analyze_completeness()
        self.analyze_temporal_patterns()
        self.analyze_severity()
        self.analyze_environmental_conditions()
        self.analyze_collision_types()
        self.analyze_geographic_distribution()
        self.identify_high_risk_patterns()
        self.generate_ml_recommendations()

        # Save report
        self.save_analysis_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nTotal crashes analyzed: {len(self.df):,}")
        print(f"Date range: {self.df['date_parsed'].min()} to {self.df['date_parsed'].max()}")
        print(f"Counties covered: {self.df['county_name'].nunique()}")
        print(f"Fatal crashes: {self.df['is_fatal'].sum():,}")
        print(f"Injury crashes: {self.df['is_injury'].sum():,}")
        print("\nNext step: Run county-level integration with work zones")
        print("="*80)


def main():
    """Main execution"""
    analyzer = NYCrashAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
