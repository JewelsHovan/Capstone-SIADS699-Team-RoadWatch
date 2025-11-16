# Report Figures

This directory contains scripts to generate all 5 key figures for the capstone project final report.

## Overview

### Figures Plan

1. **Figure 1: Geospatial Distribution** - Shows crash density and work zones across Texas
2. **Figure 2: Feature Importance** - Identifies key risk factors using SHAP and correlations
3. **Figure 3: Model Comparison** - Demonstrates ML model performance (AUC, F1, ROC curves)
4. **Figure 4: Regularization Impact** - Before/after data leakage detection and cleaning
5. **Figure 5: App Deployment** - Screenshots of Streamlit application in action

## Quick Start

### Generate All Figures

```bash
# Navigate to project root
cd /Users/julienh/Desktop/MADS/Capstone

# Run each figure script
python -m analysis.reports.figure_1_geospatial
python -m analysis.reports.figure_2_feature_importance
python -m analysis.reports.figure_3_model_comparison
python -m analysis.reports.figure_4_regularization_comparison
python -m analysis.reports.figure_5_app_screenshots
```

### Generate Individual Figures

```bash
# Figure 1 only
python analysis/reports/figure_1_geospatial.py

# Figure 2 only (requires SHAP: pip install shap)
python analysis/reports/figure_2_feature_importance.py

# Figure 3 only
python analysis/reports/figure_3_model_comparison.py

# Figure 4 only
python analysis/reports/figure_4_regularization_comparison.py

# Figure 5 (requires manual screenshots first)
python analysis/reports/figure_5_app_screenshots.py
```

## Output Location

All generated figures are saved to:
```
analysis/reports/figures/
```

## Requirements

### Python Packages
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- folium (for interactive maps)
- shap (optional, for SHAP analysis)

Install missing packages:
```bash
pip install shap folium
```

### Data Requirements
- Test dataset: `data/gold/ml_datasets/crash_level/test_latest.csv`
- Train dataset: `data/gold/ml_datasets/crash_level/train_latest.csv`
- Trained models in `models/artifacts/`

### For Figure 5 (App Screenshots)
1. Start Streamlit app: `streamlit run app/Home.py`
2. Capture screenshots (see script instructions)
3. Save to: `analysis/reports/figures/screenshots/`

## Figure Details

### Figure 1: Geospatial Distribution
**Purpose**: Visualize the scale and geographic distribution of crash data

**Outputs**:
- `figure_1_geospatial.png` - Static map for report
- `figure_1_interactive_map.html` - Interactive exploration

**Variables**:
- Crash locations (lat/long)
- Severity (binary)
- Geographic aggregations

**Takeaway**:
> "Our integrated dataset spans 500,000+ crashes across Texas (2016-2023), revealing concentrated crash hotspots in major urban corridors where work zones intersect with high-traffic areas."

### Figure 2: Feature Importance
**Purpose**: Identify which factors most strongly predict crash severity

**Outputs**:
- `figure_2a_shap_importance.png` - SHAP summary plot
- `figure_2b_rf_importance.png` - Random Forest importance
- `figure_2c_correlation_heatmap.png` - Feature correlations
- `figure_2_combined.png` - **Recommended for report**

**Variables**:
- Feature importance scores
- SHAP values
- Feature correlations

**Takeaway**:
> "Traffic volume (AADT), speed limit, and temporal factors are the strongest predictors, while adverse weather amplifies risk."

### Figure 3: Model Comparison
**Purpose**: Demonstrate that ML models successfully predict crash severity

**Outputs**:
- `figure_3_model_comparison.png` - **Recommended for report**
- `figure_3_pr_curves.png` - Supplementary

**Variables**:
- AUC-ROC scores
- F1 scores
- ROC curves for 3 models

**Takeaway**:
> "Our optimized Random Forest achieves 93% AUC-ROC and 67% F1, significantly outperforming baseline logistic regression."

### Figure 4: Regularization Impact
**Purpose**: Show importance of data quality and leakage detection

**Outputs**:
- `figure_4_regularization_comparison.png` - **Recommended**
- `figure_4_table.png` - Alternative format

**Variables**:
- Before/after metrics
- Feature importance changes
- Data quality improvements

**Takeaway**:
> "Before regularization, suspiciously high AUC (0.9999) indicated data leakage. After removing 1,434 duplicates and problematic features, performance is realistic (AUC: 0.86)."

### Figure 5: App Deployment
**Purpose**: Showcase practical deployment of predictive models

**Outputs**:
- `figure_5_app_deployment.png` - Composite screenshot
- `figure_5_placeholder.png` - Template/reference

**Variables**:
- User interface (input controls)
- Geographic risk map
- Prediction outputs

**Takeaway**:
> "Our deployed Streamlit application enables real-time crash severity prediction, facilitating data-driven decisions for work zone scheduling and safety resource allocation."

## Customization

### Adjusting Sample Sizes
Edit these parameters in the scripts for performance:
- `SAMPLE_SIZE` in `figure_1_geospatial.py` (default: 50,000)
- `SHAP_SAMPLE` in `figure_2_feature_importance.py` (default: 1,000)

### Color Schemes
All scripts use consistent color palettes:
- Traffic features: `steelblue`
- Weather features: `coral`
- Road features: `lightgreen`
- Temporal features: `mediumpurple`

### Figure Sizes
Adjust `figsize` parameter in each script:
```python
fig, ax = plt.subplots(figsize=(width, height))
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "File not found" errors
Ensure you're running from project root:
```bash
cd /Users/julienh/Desktop/MADS/Capstone
```

### SHAP computation too slow
Reduce sample size in `figure_2_feature_importance.py`:
```python
SHAP_SAMPLE = 500  # Default: 1000
```

### Memory issues
Process smaller data samples or close other applications

## For the Report

### Recommended Figures
1. `figure_1_geospatial.png` - Shows data scale
2. `figure_2_combined.png` - Shows key features
3. `figure_3_model_comparison.png` - Shows model performance
4. `figure_4_regularization_comparison.png` - Shows data quality
5. `figure_5_app_deployment.png` - Shows practical impact

### High-Resolution Output
All figures are saved at 300 DPI for print quality.

### Editing Figures
To modify figures, edit the corresponding `.py` file and re-run:
```bash
python analysis/reports/figure_X_name.py
```

## Contact

Questions or issues? Check:
1. Script comments for detailed documentation
2. Data dictionary: `data/gold/ml_datasets/crash_level/DATA_DICTIONARY.md`
3. Model metadata: `models/artifacts/*/metadata.json`

---

**Author**: Capstone Team
**Date**: 2025-11-15
**Project**: Work Zone Safety and Traffic Impact Prediction
