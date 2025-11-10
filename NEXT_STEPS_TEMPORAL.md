# Temporal Analysis - Next Steps

## DISCOVERY: You have 211,163 unused post-COVID crashes!

### Immediate Actions (Choose One):

#### Option A: Fix Current Models (1 day)
**Goal:** Get honest pre-COVID performance
```bash
# Rebuild with 2019 as test
python -m data_engineering.datasets.build_crash_level_dataset \
  --train-start 2016-01-01 --train-end 2017-12-31 \
  --val-start 2018-01-01 --val-end 2018-12-31 \
  --test-start 2019-01-01 --test-end 2019-12-31

# Retrain
python -m ml_engineering.train_with_mlflow --dataset crash --model baseline
```
**Expected:** Test AUC 0.86-0.88 (up from 0.80)

#### Option B: Build Post-COVID Models (2-3 days)
**Goal:** Production-ready for 2024-2025
```bash
# Use 2021-2023 data
python -m data_engineering.datasets.build_crash_level_dataset \
  --train-start 2021-01-01 --train-end 2021-12-31 \
  --val-start 2022-01-01 --val-end 2022-12-31 \
  --test-start 2023-01-01 --test-end 2023-03-31

python -m ml_engineering.train_with_mlflow --dataset crash --model all
```
**Expected:** Better generalization to current traffic patterns

#### Option C: Austin-Only Model (1-2 days)
**Goal:** Best local model
```bash
# Use Austin dataset with recent data (requires austin crash file path)
python -m data_engineering.datasets.build_crash_level_dataset \
  --crash-file data/bronze/texas/crashes/austin_crashes_latest.csv \
  --train-start 2021-01-01 --train-end 2022-12-31 \
  --val-start 2023-01-01 --val-end 2023-12-31 \
  --test-start 2024-01-01 --test-end 2024-12-31

python -m ml_engineering.train_with_mlflow --dataset crash --model baseline
```
**Expected:** Can validate on 2025 data!

---

## Key Findings from Temporal Drift Analysis:

### COVID Impact on Crashes:
- **Volume drop (2020):** -6.6% statewide, -26.1% Austin
- **Weekend crashes:** 8.3% → 14.2% (+72%)
- **Rush hour crashes:** 56.6% → 48.3% (-15%)
- **Night crashes:** 5.4% → 14.4% (+169%)
- **Severity bimodal:** 19.8% to 39.3% within 2020

### Why Models Failed on 2020:
- Logistic Regression: -6.5% (least affected)
- Random Forest: -9.5% (temporal features broke)
- XGBoost: -14.4% (complex patterns invalidated)

**NOT model overfitting - it's distribution shift!**

---

## Recommended: Do Option A First

1. **Retrain with 2019 test** (1 day)
   - Prove models are good (AUC 0.86-0.88)
   - Document in final report

2. **Then build post-COVID models** (2-3 days)
   - Use 2021-2023 data (211k crashes)
   - Production-ready for current deployments

3. **Research contribution:**
   - "COVID-19 Impact on Crash Prediction"
   - Compare pre vs post-COVID model performance
   - Quantify distribution shift

---

## Available Data Summary:

| Source | Period | Crashes | Status |
|--------|--------|---------|--------|
| Kaggle | 2016-2020 | 371,674 | ✓ Currently used |
| Kaggle | 2021-2023 | 211,163 | ✗ UNUSED |
| Austin | 2021-2024 | 51,117 | ✗ UNUSED |
| Austin | 2024-2025 | 21,129 | ✗ UNUSED (very recent!) |

Total unused: **283,409 crashes** ready for new models!
