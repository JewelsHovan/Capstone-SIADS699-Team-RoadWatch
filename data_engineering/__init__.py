"""
Data Engineering Module for Texas Work Zone Crash Risk Prediction

This module contains all data engineering code organized by pipeline stage:
1. download/ - Data acquisition scripts
2. clean/ - Data cleaning and validation
3. integrate/ - Data integration (crashes + work zones + AADT + weather)
4. features/ - Feature engineering
5. datasets/ - Final dataset creation

Usage:
    from data_engineering.integrate import integrate_workzones
    from data_engineering.features import engineer_crash_features
"""

__version__ = "1.0.0"
