import pandas as pd
import os

MERGED_PATH = "/work/siads_699_mads_capstone/data/processed/workzone_crash_merged.csv"
OUTPUT_PATH = "/work/siads_699_mads_capstone/data/models/workzone_features.csv"

def aggregate_features():
    print("Loading merged work-zone crash dataset")
    df = pd.read_csv(MERGED_PATH)
    print(f"Loaded {len(df)} merged records from {df['road_event_id'].nunique()} unique work zones.")
    print("Aggregating crash statistics by work zone")
    agg = (
        df.groupby("road_event_id")
          .agg({
              "crash_id": "count",
              "severity": "mean",
              "distance_km": "mean",
              "visibility_mi": "mean",
              "precip_in": "mean",
              "temperature_F": "mean",
              "wind_mph": "mean"
          })
          .rename(columns={
              "crash_id": "crash_count",
              "severity": "avg_severity",
              "distance_km": "avg_distance_km",
              "visibility_mi": "avg_visibility_mi",
              "precip_in": "avg_precip_in",
              "temperature_F": "avg_temp_F",
              "wind_mph": "avg_wind_mph"
          })
          .reset_index()
    )
    meta_cols = ["road_event_id","road_name","duration_hr","vehicle_impact", "latitude","longitude"]
    meta = df[meta_cols].drop_duplicates("road_event_id")
    features = meta.merge(agg, on="road_event_id", how="left")
    # high-risk zones
    q75 = features["crash_count"].quantile(0.75)
    features["high_risk"] = (features["crash_count"] >= q75).astype(int)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved aggregated feature table - {OUTPUT_PATH}")
    print(features.head(10))
    return features
if __name__ == "__main__":
    aggregate_features()
