# Data Rebuild Instructions - Overfitting Fixes

## What Changed

### 1. **Removed Exact Coordinates (Start_Lat, Start_Lng)**
**Why:** These were the top 2 most important features (45% combined importance) but caused severe overfitting:
- Validation AUC: 0.875
- Test AUC: 0.613 (**30% drop!**)

The model was memorizing specific crash locations instead of learning generalizable patterns.

**Replaced with:**
- `lat_zone`: 4 coarse latitude zones (south, south_central, central, north)
- `lng_zone`: 4 coarse longitude zones (west, west_central, central, east)
- `region`: 16 combined zones (e.g., "south_west", "north_east")
- `county_top20`: Top 20 counties by crash frequency + "other"

### 2. **Added HPMS Road Characteristics**
**What we get:**
- `hpms_speed_limit`: Posted speed limit
- `hpms_lanes`: Number of through lanes
- `hpms_functional_class`: Road type (interstate, arterial, etc.)
- `hpms_aadt`: Annual Average Daily Traffic from HPMS

**Derived categorical features:**
- `speed_category`: low (<35), medium (35-50), high (50-65), highway (>65)
- `lane_category`: narrow (1-2), standard (3-4), wide (5+)
- `road_class`: functional classification as string

### 3. **Enhanced Location Features**
- `is_major_city`: Binary indicator for 6 major Texas cities
- `city_size_category`: large (>1000 crashes), medium (100-1000), small (<100)
- Coarse geographic zones instead of exact coordinates

## Expected Improvements

### Before (with lat/lng):
- Val AUC: 0.875, Test AUC: 0.613 (**30% gap = overfitting**)
- Top 2 features: Start_Lat (25.7%), Start_Lng (20.9%)
- Model learned specific locations, not generalizable patterns

### After (with regions + HPMS):
- Expected: **10-15% smaller train/test gap**
- More generalizable features (road characteristics, regions)
- Better feature importance distribution
- Should improve test AUC to ~0.70-0.75

## How to Rebuild

### Step 1: Rebuild Crash-Level Dataset
```bash
cd /Users/julienh/Desktop/MADS/Capstone

# Full rebuild with HPMS integration
python data_engineering/datasets/build_crash_level_dataset.py

# Or with sampling for faster testing
python data_engineering/datasets/build_crash_level_dataset.py --sample 50000
```

**What happens:**
1. Loads crash data (2016-2020)
2. **Integrates HPMS** via spatial join (nearest road within 100m)
3. Creates target variable (high_severity)
4. Engineers features:
   - Removes Start_Lat/Start_Lng
   - Creates coarse regions
   - Adds HPMS road characteristics
   - Creates categorical features
5. Temporal split (2016-2018 train, 2019 val, 2020 test)
6. Saves to `data/gold/ml_datasets/crash_level/`

### Step 2: Verify New Features
```bash
# Check first row to see new features
head -1 data/gold/ml_datasets/crash_level/train_latest.csv | tr ',' '\n' | grep -E "lat_zone|lng_zone|region|hpms|county"
```

You should see:
- `lat_zone`, `lng_zone`, `region`
- `hpms_speed_limit`, `hpms_lanes`, `hpms_functional_class`, `hpms_aadt`
- `speed_category`, `lane_category`, `road_class`
- `county_top20`, `is_major_city`, `city_size_category`

And should NOT see:
- ❌ `Start_Lat`, `Start_Lng` (removed!)

### Step 3: Retrain Models
```bash
# Quick demo to see new features in action
python ml_engineering/demo_quick_start.py

# Full training with MLflow
python -m ml_engineering.train_with_mlflow --dataset crash --model baseline
```

### Step 4: Compare Results

**Key metrics to watch:**

| Metric | Before | Target After |
|--------|--------|--------------|
| **Val AUC** | 0.875 | 0.80-0.85 (slightly lower OK) |
| **Test AUC** | 0.613 | **0.70-0.75** (much better!) |
| **AUC Gap** | **-30%** 🚨 | **-10% to -15%** ✅ |
| Top Feature | Start_Lat (25.7%) | Should be distributed |

**Success criteria:**
✅ Test AUC improves by at least 10%
✅ Train/Test gap reduces from 30% to <15%
✅ Feature importance more evenly distributed
✅ HPMS features appear in top 10

## Troubleshooting

### Issue: HPMS Integration Fails
**Symptom:**
```
⚠️  HPMS integration failed: ...
   Continuing without HPMS features...
```

**Fix:**
Check that HPMS file exists:
```bash
ls -lh data/silver/texas/roadway/hpms_texas_2023.gpkg
```

If missing, you won't get HPMS features but region features will still work.

### Issue: Missing Columns in Training
**Symptom:**
```
Missing numeric features: ['hpms_speed_limit', 'hpms_lanes', ...]
```

**This is OK!** The pipeline will use whatever features are available. If HPMS integration failed, you'll just have fewer features but the model will still train.

### Issue: "Start_Lat not found"
**This is expected!** Start_Lat/Start_Lng are intentionally removed in the feature engineering step to prevent overfitting.

## Next Steps After Rebuild

1. **Run baseline models:**
   ```bash
   python -m ml_engineering.train_with_mlflow --dataset crash --model baseline
   ```

2. **Try XGBoost** (better at handling categorical features):
   ```bash
   python -m ml_engineering.train_with_mlflow --dataset crash --model xgboost
   ```

3. **Hyperparameter tuning:**
   ```bash
   python -m ml_engineering.train_with_mlflow --dataset crash --tune
   ```

4. **View results in MLflow:**
   ```bash
   mlflow ui
   # Open http://localhost:5000
   ```

5. **Compare old vs new:**
   - Old runs: High validation AUC (0.87) but poor test AUC (0.61)
   - New runs: More balanced - smaller gap means better generalization

## Files Modified

1. **`data_engineering/datasets/build_crash_level_dataset.py`**
   - Line 287-379: Added region features and HPMS processing
   - Removes Start_Lat/Start_Lng
   - Creates coarse geographic zones
   - Adds HPMS categorical features

2. **`ml_engineering/preprocessing/feature_lists.py`**
   - Updated CRASH_NUMERIC_FEATURES (removed lat/lng, added HPMS)
   - Updated CRASH_CATEGORICAL_FEATURES (added regions, HPMS categories)
   - Added documentation about overfitting fix

## Expected Timeline

- **Dataset rebuild:** 5-10 minutes (full data) or 1-2 minutes (sampled)
- **Model training:** 2-3 minutes per model
- **Total:** ~15 minutes to rebuild, retrain, and compare

## Questions?

After rebuilding, compare:
1. Feature importances - should be more distributed
2. Train/val/test AUC - gap should be smaller
3. Test performance - should improve significantly

Report back the new metrics and we'll analyze if overfitting is fixed!
