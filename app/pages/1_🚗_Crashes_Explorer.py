"""
Crashes Explorer Page
Comprehensive analysis of raw crash data from Kaggle and Austin
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path for imports (insert at beginning to prioritize app/config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, CUSTOM_CSS, DEFAULT_SAMPLE_SIZES, SEVERITY_COLORS
from utils.data_loader import load_crash_data, load_austin_crashes
from utils.visualizations import (
    create_severity_pie_chart,
    create_temporal_line_chart,
    create_county_bar_chart,
    create_feature_histogram,
    create_box_plot
)
from utils.map_utils import create_crash_map, create_heatmap
from streamlit_folium import st_folium

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Crashes Explorer</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive analysis of Texas traffic crashes (2016-2023)**")
st.markdown("---")

# Sidebar - Data Selection
with st.sidebar:
    st.markdown("##  Data Controls")

    data_source = st.selectbox(
        "Select Dataset",
        ["Kaggle US Accidents", "Austin Crashes", "Combined"],
        help="Choose which crash dataset to analyze"
    )

    sample_size = st.slider(
        "Sample Size",
        min_value=1000,
        max_value=100000,
        value=50000,
        step=1000,
        help="Number of crashes to load (for performance)"
    )

    st.markdown("---")

    date_filter = st.checkbox("Enable Date Filter", value=False)

    if date_filter:
        st.markdown("###  Date Range")
        start_year = st.selectbox("Start Year", list(range(2016, 2024)), index=0)
        end_year = st.selectbox("End Year", list(range(2016, 2024)), index=7)

    severity_filter = st.checkbox("Filter by Severity", value=False)

    if severity_filter:
        st.markdown("### ⚠️ Severity Levels")
        selected_severities = st.multiselect(
            "Select Severities",
            [1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Level {x}"
        )

# Load data based on selection
with st.spinner(f"Loading {data_source} data..."):
    if data_source == "Kaggle US Accidents":
        df = load_crash_data(sample_size=sample_size)
        date_col = 'Start_Time'
        county_col = 'County'
    elif data_source == "Austin Crashes":
        df = load_austin_crashes(sample_size=sample_size)
        date_col = 'crash_date' if 'crash_date' in df.columns else 'Start_Time'
        county_col = 'county' if 'county' in df.columns else 'County'
    else:  # Combined
        df_kaggle = load_crash_data(sample_size=sample_size//2)
        df_austin = load_austin_crashes(sample_size=sample_size//2)
        # For simplicity, use kaggle data
        df = df_kaggle
        date_col = 'Start_Time'
        county_col = 'County'

# Apply filters
if date_filter:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[
        (df[date_col].dt.year >= start_year) &
        (df[date_col].dt.year <= end_year)
    ]

if severity_filter and 'Severity' in df.columns:
    df = df[df['Severity'].isin(selected_severities)]

# Summary metrics
st.markdown("##  Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Crashes",
        f"{len(df):,}",
        help="Number of crashes in current view"
    )

with col2:
    if 'Severity' in df.columns:
        avg_severity = df['Severity'].mean()
        st.metric(
            "Avg Severity",
            f"{avg_severity:.2f}",
            help="Average crash severity (1-4 scale)"
        )
    else:
        st.metric("Avg Severity", "N/A")

with col3:
    if county_col in df.columns:
        unique_counties = df[county_col].nunique()
        st.metric(
            "Counties",
            f"{unique_counties}",
            help="Number of unique counties"
        )
    else:
        st.metric("Counties", "N/A")

with col4:
    if date_col in df.columns:
        date_range = (df[date_col].max() - df[date_col].min()).days
        st.metric(
            "Date Range",
            f"{date_range} days",
            help="Span of crash records"
        )
    else:
        st.metric("Date Range", "N/A")

st.markdown("---")

# Visualization tabs
tab1, tab2, tab3, tab4 = st.tabs([" Temporal Analysis", " Geographic Analysis", "⚠️ Severity Analysis", " Feature Distributions"])

with tab1:
    st.markdown("### Crashes Over Time")

    if date_col in df.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            freq = st.selectbox(
                "Aggregation Frequency",
                ["Daily", "Weekly", "Monthly", "Yearly"],
                index=2
            )

            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}

            fig = create_temporal_line_chart(
                df,
                date_col=date_col,
                freq=freq_map[freq],
                title=f"Crashes Over Time ({freq})"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("###  Insights")

            # Parse dates
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')

            # Most crashes by day of week
            if not df_temp[date_col].isna().all():
                df_temp['day_name'] = df_temp[date_col].dt.day_name()
                day_counts = df_temp['day_name'].value_counts()

                st.markdown(f"""
                <div class="info-box">
                <b>Peak Day:</b> {day_counts.index[0]}<br>
                <b>Crashes:</b> {day_counts.values[0]:,}<br>
                <br>
                <b>Quietest Day:</b> {day_counts.index[-1]}<br>
                <b>Crashes:</b> {day_counts.values[-1]:,}
                </div>
                """, unsafe_allow_html=True)

        # Hour of day analysis (if available)
        if 'hour' in df.columns or date_col in df.columns:
            st.markdown("### ⏰ Hourly Distribution")

            if 'hour' not in df.columns:
                df['hour'] = pd.to_datetime(df[date_col], errors='coerce').dt.hour

            hour_counts = df['hour'].value_counts().sort_index()

            fig_hour = go.Figure(data=[go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                marker_color='#3498db'
            )])

            fig_hour.update_layout(
                title='Crashes by Hour of Day',
                xaxis_title='Hour',
                yaxis_title='Number of Crashes',
                height=350
            )

            st.plotly_chart(fig_hour, use_container_width=True)

    else:
        st.warning(f"Date column '{date_col}' not found in dataset")

with tab2:
    st.markdown("### Geographic Distribution")

    map_type = st.radio(
        "Map Type",
        ["Heatmap", "Point Map"],
        horizontal=True,
        help="Choose visualization style"
    )

    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
        with st.spinner("Generating map..."):
            if map_type == "Heatmap":
                m = create_heatmap(df, lat_col='Start_Lat', lon_col='Start_Lng', max_points=10000)
            else:
                m = create_crash_map(df, lat_col='Start_Lat', lon_col='Start_Lng', max_points=1000)

            st_folium(m, width=1200, height=600, returned_objects=[])

        # County distribution
        if county_col in df.columns:
            st.markdown("###  Top Counties by Crash Count")

            fig_county = create_county_bar_chart(df, county_col=county_col, top_n=15)
            st.plotly_chart(fig_county, use_container_width=True)

    else:
        st.warning("Geographic coordinates not found in dataset")

with tab3:
    st.markdown("### Severity Analysis")

    if 'Severity' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Severity pie chart
            fig_sev = create_severity_pie_chart(df, severity_col='Severity')
            st.plotly_chart(fig_sev, use_container_width=True)

        with col2:
            # Severity statistics
            st.markdown("###  Severity Stats")

            sev_counts = df['Severity'].value_counts().sort_index()

            st.markdown(f"""
            <div class="info-box">
            <b>Severity Distribution:</b><br>
            - Level 1 (Minor): {sev_counts.get(1, 0):,} ({sev_counts.get(1, 0)/len(df)*100:.1f}%)<br>
            - Level 2 (Moderate): {sev_counts.get(2, 0):,} ({sev_counts.get(2, 0)/len(df)*100:.1f}%)<br>
            - Level 3 (Serious): {sev_counts.get(3, 0):,} ({sev_counts.get(3, 0)/len(df)*100:.1f}%)<br>
            - Level 4 (Fatal): {sev_counts.get(4, 0):,} ({sev_counts.get(4, 0)/len(df)*100:.1f}%)<br>
            </div>
            """, unsafe_allow_html=True)

        # Severity by time of day (if hour available)
        if 'hour' in df.columns or date_col in df.columns:
            st.markdown("### ⏰ Severity by Hour")

            if 'hour' not in df.columns:
                df['hour'] = pd.to_datetime(df[date_col], errors='coerce').dt.hour

            severity_by_hour = df.groupby(['hour', 'Severity']).size().unstack(fill_value=0)

            fig_sev_hour = go.Figure()

            for severity in sorted(severity_by_hour.columns):
                fig_sev_hour.add_trace(go.Bar(
                    name=f'Severity {severity}',
                    x=severity_by_hour.index,
                    y=severity_by_hour[severity],
                    marker_color=SEVERITY_COLORS.get(severity, '#95a5a6')
                ))

            fig_sev_hour.update_layout(
                title='Crash Severity by Hour of Day',
                xaxis_title='Hour',
                yaxis_title='Number of Crashes',
                barmode='stack',
                height=400
            )

            st.plotly_chart(fig_sev_hour, use_container_width=True)

    else:
        st.warning("Severity data not available in this dataset")

with tab4:
    st.markdown("### Feature Distributions")

    # Select numeric columns for distribution analysis
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            numeric_cols,
            help="Choose a numeric feature to visualize"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Histogram
            fig_hist = create_feature_histogram(df, selected_feature, bins=50)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Statistics
            st.markdown(f"###  {selected_feature} Stats")

            feature_data = df[selected_feature].dropna()

            st.markdown(f"""
            <div class="info-box">
            <b>Count:</b> {len(feature_data):,}<br>
            <b>Mean:</b> {feature_data.mean():.2f}<br>
            <b>Median:</b> {feature_data.median():.2f}<br>
            <b>Std Dev:</b> {feature_data.std():.2f}<br>
            <b>Min:</b> {feature_data.min():.2f}<br>
            <b>Max:</b> {feature_data.max():.2f}<br>
            </div>
            """, unsafe_allow_html=True)

        # Box plot by severity (if available)
        if 'Severity' in df.columns and selected_feature in df.columns:
            st.markdown(f"### {selected_feature} by Severity Level")

            fig_box = create_box_plot(df, 'Severity', selected_feature)
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.warning("No numeric features available for distribution analysis")

st.markdown("---")

# Data preview
with st.expander(" View Raw Data Sample"):
    st.dataframe(df.head(100), use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Filtered Data (CSV)",
        data=csv,
        file_name=f"texas_crashes_{data_source.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )

# Sidebar - Current view info
with st.sidebar:
    st.markdown("---")
    st.markdown("###  Current View")
    st.markdown(f"**Showing:** {len(df):,} crashes")
    st.markdown(f"**Source:** {data_source}")
    if date_filter:
        st.markdown(f"**Years:** {start_year}-{end_year}")
    if severity_filter:
        st.markdown(f"**Severities:** {selected_severities}")
