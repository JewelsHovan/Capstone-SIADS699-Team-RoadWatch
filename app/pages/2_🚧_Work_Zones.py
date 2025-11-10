"""
Work Zones Analysis Page
Analysis of Texas work zones from WZDx feed
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path (insert at beginning to prioritize app/config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, CUSTOM_CSS
from utils.data_loader import load_work_zones
from utils.map_utils import create_workzone_map
from streamlit_folium import st_folium

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚧 Work Zones Analysis</h1>', unsafe_allow_html=True)
st.markdown("**Analysis of active work zones across Texas**")
st.markdown("---")

# Load data
with st.spinner("Loading work zone data..."):
    df = load_work_zones()

# Summary metrics
st.markdown("## 📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Work Zones", f"{len(df):,}")

with col2:
    if 'county_name' in df.columns:
        st.metric("Counties", f"{df['county_name'].nunique()}")
    else:
        st.metric("Counties", "N/A")

with col3:
    if 'start_date' in df.columns and 'end_date' in df.columns:
        df['duration'] = (pd.to_datetime(df['end_date'], errors='coerce') -
                          pd.to_datetime(df['start_date'], errors='coerce')).dt.days
        avg_duration = df['duration'].mean()
        st.metric("Avg Duration", f"{avg_duration:.0f} days")
    else:
        st.metric("Avg Duration", "N/A")

with col4:
    if 'road_name' in df.columns:
        st.metric("Unique Roads", f"{df['road_name'].nunique()}")
    else:
        st.metric("Unique Roads", "N/A")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📊 Analytics", "📋 Data Table"])

with tab1:
    st.markdown("### Work Zone Locations")

    if 'beginning_latitude' in df.columns and 'beginning_longitude' in df.columns:
        with st.spinner("Generating map..."):
            m = create_workzone_map(df)
            st_folium(m, width=1200, height=600, returned_objects=[])

        st.info(f"Showing {len(df):,} work zones across Texas")
    else:
        st.warning("Geographic coordinates not available")

with tab2:
    st.markdown("### Work Zone Statistics")

    col1, col2 = st.columns(2)

    with col1:
        if 'county_name' in df.columns:
            st.markdown("#### Top Counties by Work Zones")

            county_counts = df['county_name'].value_counts().head(15)

            fig = px.bar(
                x=county_counts.index,
                y=county_counts.values,
                labels={'x': 'County', 'y': 'Number of Work Zones'},
                title='Top 15 Counties'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'road_name' in df.columns:
            st.markdown("#### Top Roads with Work Zones")

            road_counts = df['road_name'].value_counts().head(15)

            fig = px.bar(
                x=road_counts.values,
                y=road_counts.index,
                orientation='h',
                labels={'x': 'Number of Work Zones', 'y': 'Road Name'},
                title='Top 15 Roads'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Duration analysis
    if 'duration' in df.columns:
        st.markdown("#### Work Zone Duration Distribution")

        fig_duration = px.histogram(
            df,
            x='duration',
            nbins=50,
            title='Distribution of Work Zone Durations',
            labels={'duration': 'Duration (days)'}
        )

        fig_duration.update_layout(height=400)
        st.plotly_chart(fig_duration, use_container_width=True)

with tab3:
    st.markdown("### Work Zones Data Table")

    # Display columns
    display_cols = [col for col in ['road_name', 'county_name', 'direction',
                                      'start_date', 'end_date', 'duration',
                                      'beginning_latitude', 'beginning_longitude']
                    if col in df.columns]

    if display_cols:
        st.dataframe(df[display_cols], use_container_width=True, height=600)
    else:
        st.dataframe(df, use_container_width=True, height=600)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Complete Dataset",
        data=csv,
        file_name="texas_work_zones.csv",
        mime="text/csv"
    )

st.markdown("---")

# Data source info
st.markdown("""
### 📝 Data Source
**Source**: Texas Department of Transportation (TxDOT) Work Zone Data Exchange (WZDx) Feed
**Format**: CSV export from JSON feed
**Update Frequency**: Real-time
**Coverage**: Statewide Texas work zones
""")
