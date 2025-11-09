# Streamlit Prediction Features

## Overview

Added two new prediction pages to the Texas Crash Analysis Dashboard:

1. **Page 6: Segment Risk Map** (🗺️) - Infrastructure planning tool
2. **Page 7: Real-Time Crash Severity Predictor** (🚨) - Emergency response tool

---

## Page 6: 🗺️ Segment Risk Map

### Purpose
Identify high-risk road segments for proactive safety improvements and resource allocation.

### Features

#### 1. Interactive Area Selection
- Draw bounding box or polygon on map
- Select any geographic area in Texas
- Automatically loads HPMS segments in area

#### 2. Risk Assessment
- Predicts crash risk for each road segment
- Uses features: speed_limit, lanes, road_type, AADT, urban/rural
- Color-codes segments: 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk

#### 3. Analysis Tab
- Feature importance visualization
- Risk distribution by road type
- Scatter plots (risk vs traffic volume)

#### 4. Recommendations Tab
- Actionable safety recommendations
- Prioritized by risk level
- Specific interventions for high-risk segments

### How to Use

1. Navigate to "🗺️ Segment Risk Map" page
2. Use drawing tools to select an area
3. Wait for segments to load and predictions to run
4. Explore map, analysis, and recommendations tabs
5. Download results as CSV

### Model Integration

**Currently using:** Baseline heuristic (demo mode)

**To use trained model:**
```python
# 1. Train your segment-level model
from ml_engineering.baseline_models import train_segment_model

model = train_segment_model(...)

# 2. Save model
import pickle
with open('models/segment_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 3. Model will be automatically loaded by app
```

**Expected features:**
- `speed_limit` (float)
- `through_lanes` (float)
- `f_system` (int, 1-7)
- `urban_id` (int)
- `aadt` (float)
- `speed_x_aadt` (float)
- `fsystem_x_urban` (float)
- `lanes_x_aadt` (float)

---

## Page 7: 🚨 Real-Time Crash Severity Predictor

### Purpose
Predict severity of crashes happening right now to prioritize emergency response.

### Features

#### 1. Interactive Location Selection
- Click anywhere on map to mark crash location
- Automatically extracts road characteristics from HPMS
- Shows: road type, speed limit, lanes, traffic volume

#### 2. Current Conditions Input
- **Time:** Hour of day, day of week
- **Weather:** Condition, temperature, visibility
- **Auto-calculated:** is_weekend, is_rush_hour, adverse_weather

#### 3. Quick Presets
- 🌧️ Rainy Rush Hour
- 🌙 Night Clear
- 🌫️ Foggy Morning

#### 4. Severity Prediction
- Probability of high severity (0-100%)
- Color-coded alert level
- Dispatch recommendation
- Confidence gauge

#### 5. What-If Analysis
- "If weather was clear..."
- "If crash at 2 PM..."
- Shows how conditions affect severity

### How to Use

1. Navigate to "🚨 Crash Severity Predictor" page
2. Click on map to select crash location
3. Enter current conditions (or use preset)
4. Click "PREDICT CRASH SEVERITY"
5. See results and dispatch recommendation

### Dispatch Recommendations

**HIGH SEVERITY (>70% confidence):**
- 🚨 PRIORITY RESPONSE
- Multiple ambulances
- Alert trauma center
- Dispatch fire rescue

**MEDIUM SEVERITY (40-70% confidence):**
- ⚠️ Standard response with backup on standby

**LOW SEVERITY (<40% confidence):**
- ✓ Single unit response adequate

### Model Integration

**Currently using:** Baseline heuristic (demo mode)

**To use trained model:**
```python
# 1. Train your crash-level model
from ml_engineering.baseline_models import train_crash_model

model = train_crash_model(...)

# 2. Save model
import pickle
with open('models/crash_severity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 3. Model will be automatically loaded by app
```

**Expected features:**
```python
feature_cols = [
    'hour',              # int, 0-23
    'day_of_week',       # int, 0-6
    'is_weekend',        # binary
    'is_rush_hour',      # binary
    'temperature',       # float, °F
    'visibility',        # float, miles
    'adverse_weather',   # binary
    'low_visibility',    # binary
    'speed_limit',       # float, mph (from HPMS)
    'through_lanes',     # float (from HPMS)
    'f_system',          # int, 1-7 (from HPMS)
    'aadt'               # float, vehicles/day (from HPMS)
]
```

**Target:** `high_severity` (binary: 1 if Severity ≥ 3, else 0)

---

## Technical Architecture

### File Structure

```
app/
├── pages/
│   ├── 6_🗺️_Segment_Risk_Map.py       # Segment risk assessment
│   └── 7_🚨_Crash_Severity_Predictor.py # Real-time severity prediction
├── utils/
│   ├── model_loader.py                  # Model loading & prediction utilities
│   ├── data_loader.py                   # Data loading utilities (existing)
│   └── map_utils.py                     # Map utilities (existing)
└── PREDICTION_FEATURES.md              # This file

models/                                 # Created automatically
├── crash_severity_model.pkl           # Trained crash model (save here)
├── segment_risk_model.pkl             # Trained segment model (save here)
├── crash_severity_model_metadata.json # Optional metadata
└── segment_risk_model_metadata.json   # Optional metadata
```

### Model Loading

Models are automatically loaded with caching:

```python
from app.utils.model_loader import (
    load_crash_severity_model,
    load_segment_risk_model,
    predict_crash_severity,
    predict_segment_risk
)

# Load models (cached)
crash_model = load_crash_severity_model()  # None if not found
segment_model = load_segment_risk_model()   # None if not found

# Make predictions
crash_pred = predict_crash_severity(crash_model, features)
segment_pred = predict_segment_risk(segment_model, segment_features)
```

If no trained model is found, automatically falls back to baseline heuristics.

---

## Integration with Trained Models

### Step 1: Train Your Models

Use your existing training scripts:

```bash
# Train crash-level model
python -m ml_engineering.baseline_models --dataset crash

# Train segment-level model
python -m ml_engineering.baseline_models --dataset segment
```

### Step 2: Save Models for App

```python
import pickle
from pathlib import Path

# After training your models
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Save crash model
with open(MODELS_DIR / "crash_severity_model.pkl", 'wb') as f:
    pickle.dump(crash_model, f)

# Save segment model
with open(MODELS_DIR / "segment_risk_model.pkl", 'wb') as f:
    pickle.dump(segment_model, f)

# Optional: Save metadata
import json
metadata = {
    'model_type': 'RandomForest',
    'training_date': '2025-11-09',
    'performance': {
        'roc_auc': 0.75,
        'accuracy': 0.72
    },
    'features': feature_cols
}

with open(MODELS_DIR / "crash_severity_model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Step 3: Restart Streamlit

```bash
streamlit run app/app.py
```

Models will be automatically detected and loaded!

---

## Customization

### Adjusting Risk Thresholds

In Segment Risk Map sidebar, adjust:
- **High Risk Threshold:** Default 3.0 predicted crashes
- **Medium Risk Threshold:** Default 1.0 predicted crashes

### Modifying Baseline Predictions

Edit `app/utils/model_loader.py`:

```python
def baseline_crash_severity_prediction(features):
    """Customize baseline heuristic here"""
    risk_score = 0.0

    # Add your custom logic
    if features.get('your_condition'):
        risk_score += 0.X

    return risk_score
```

### Adding New Features

1. Update feature extraction in page
2. Update `model_loader.py` preprocessing
3. Update expected features list
4. Retrain model with new features

---

## Performance Considerations

### Caching
- Models cached with `@st.cache_resource`
- HPMS data queries use spatial indexing
- Data loaded only once per session

### Spatial Queries
- HPMS loaded with bounding box filter
- Only segments in selected area loaded
- Typical query: <1 second for 1000 segments

### Recommendations
- Keep selected areas reasonable (< 100 sq miles)
- Use Streamlit Cloud for best performance
- Consider model size (< 100 MB recommended)

---

## Troubleshooting

### "Model not found" Warning

**Cause:** No trained model file exists

**Solution:** Train and save model, or use baseline mode

### "No road segments found"

**Cause:** Selected area outside Texas or too small

**Solution:** Select larger area or different location

### Slow Loading

**Cause:** Large HPMS file (1.1 GB)

**Solution:**
- Use smaller bounding boxes
- Consider pre-filtering HPMS by region
- Use Streamlit caching effectively

### Prediction Errors

**Cause:** Missing features or wrong data types

**Solution:** Check feature preprocessing in `model_loader.py`

---

## Future Enhancements

### Planned Features
- [ ] Model comparison (compare multiple models side-by-side)
- [ ] Batch prediction (upload CSV of locations)
- [ ] Historical validation (compare predictions to actual crashes)
- [ ] Feature importance from trained models
- [ ] Confidence intervals for predictions
- [ ] Integration with work zone data
- [ ] Export predictions to GeoJSON for GIS software
- [ ] API endpoints for programmatic access

### Model Improvements
- [ ] Ensemble models (combine multiple models)
- [ ] Time-series forecasting for temporal patterns
- [ ] Spatial autocorrelation in predictions
- [ ] Active learning from user feedback
- [ ] Model retraining pipeline

---

## Data Dictionary

### Segment-Level Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| speed_limit | float | Posted speed limit (mph) | HPMS |
| through_lanes | float | Number of through lanes | HPMS |
| f_system | int | Functional class (1-7) | HPMS |
| urban_id | int | Urban area code | HPMS |
| aadt | float | Traffic volume (vehicles/day) | HPMS |
| speed_x_aadt | float | Speed × AADT interaction | Engineered |
| fsystem_x_urban | float | Road type × Urban interaction | Engineered |
| lanes_x_aadt | float | Lanes × AADT interaction | Engineered |

### Crash-Level Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| hour | int | Hour of day (0-23) | User input |
| day_of_week | int | Day of week (0-6) | User input |
| is_weekend | binary | Saturday/Sunday flag | Derived |
| is_rush_hour | binary | Peak traffic hours | Derived |
| temperature | float | Temperature (°F) | User input |
| visibility | float | Visibility (miles) | User input |
| adverse_weather | binary | Rain/fog/snow flag | Derived |
| low_visibility | binary | Visibility < 2 miles | Derived |
| speed_limit | float | Posted speed (mph) | HPMS @ location |
| through_lanes | float | Number of lanes | HPMS @ location |
| f_system | int | Road type (1-7) | HPMS @ location |
| aadt | float | Traffic volume | HPMS @ location |

---

## Contact & Support

**Questions?** Open an issue on GitHub

**Contributions?** Pull requests welcome!

**Documentation:** See main README.md and DATA_DICTIONARY.md files

---

## License

Same as main project license.
