"""
Configuration file for Texas Work Zone Dashboard
Colors, constants, and settings
"""

# Path configuration
from config.paths import TEXAS_GOLD_ANALYTICS

# Page Configuration
PAGE_CONFIG = {
    'page_title': 'Texas Work Zone Dashboard',
    'page_icon': '🚧',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Traffic Volume Colors (for maps and charts)
TRAFFIC_COLORS = {
    'very_low': '#2ecc71',      # Green
    'low': '#52c76c',           # Light green
    'medium': '#f39c12',        # Orange
    'high': '#e74c3c',          # Red
    'very_high': '#c0392b'      # Dark red
}

# Traffic Volume Labels
TRAFFIC_LABELS = {
    'very_low': 'Very Low (< 1K)',
    'low': 'Low (1-5K)',
    'medium': 'Medium (5-15K)',
    'high': 'High (15-30K)',
    'very_high': 'Very High (> 30K)'
}

# Chart Color Palette (for non-traffic charts)
CHART_COLORS = [
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#2ecc71',  # Green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Turquoise
    '#34495e',  # Dark gray
    '#e67e22',  # Carrot
]

# Plotly Chart Theme
PLOTLY_THEME = 'plotly_white'

# Map Settings
MAP_CENTER = [31.0, -100.0]  # Center of Texas
MAP_ZOOM = 6

# Data File Path
DATA_PATH = str(TEXAS_GOLD_ANALYTICS / 'texas_work_zones_with_aadt.csv')

# Display Column Names (for data explorer)
DISPLAY_COLUMNS = [
    'road_event_id',
    'road_name',
    'direction',
    'start_date_parsed',
    'end_date_parsed',
    'duration_days',
    'latitude',
    'longitude',
    'total_num_lanes',
    'vehicle_impact',
    'aadt_filled',
    'traffic_volume_category',
    'exposure_score',
    'CNTY_NM',
    'DIST_NM'
]

# Column Display Names (prettier names for tables)
COLUMN_RENAME = {
    'road_event_id': 'Event ID',
    'road_name': 'Road',
    'direction': 'Direction',
    'start_date_parsed': 'Start Date',
    'end_date_parsed': 'End Date',
    'duration_days': 'Duration (days)',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'total_num_lanes': 'Lanes',
    'vehicle_impact': 'Vehicle Impact',
    'aadt_filled': 'AADT',
    'traffic_volume_category': 'Traffic Category',
    'exposure_score': 'Exposure Score',
    'CNTY_NM': 'County',
    'DIST_NM': 'District'
}

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    /* Metric card styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }

    /* Ensure metric text is visible */
    .stMetric label {
        color: #2c3e50 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        color: #2c3e50 !important;
    }

    /* Info box styling */
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }

    /* Warning box styling */
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        margin: 10px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Adjust padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
"""
