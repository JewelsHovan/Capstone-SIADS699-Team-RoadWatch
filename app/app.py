"""
Texas Crash Analysis Dashboard - Home Page
SIADS 699 Capstone Project
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import PAGE_CONFIG, CUSTOM_CSS, TEXAS_CENTER
from utils.data_loader import get_data_summary, get_file_sizes
from streamlit_folium import st_folium
import folium

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header"> Texas Crash Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;'>
    <b>SIADS 699 Capstone Project</b> | Comprehensive analysis of Texas traffic crashes, work zones, and ML datasets for crash risk prediction
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load summary statistics
with st.spinner("Loading summary statistics..."):
    summary = get_data_summary()
    file_sizes = get_file_sizes()

# Hero Metrics Row
st.markdown("##  Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Crashes",
        value=f"{summary.get('total_crashes', 0):,}",
        help="Combined Kaggle US Accidents + Austin crashes"
    )

with col2:
    st.metric(
        label="Work Zones",
        value=f"{summary.get('work_zones', 0):,}",
        help="Active work zones from WZDx feed"
    )

with col3:
    st.metric(
        label="Unique Road Segments",
        value=f"{summary.get('unique_segments', 0):,}",
        help="Unique road segments in ML dataset"
    )

with col4:
    st.metric(
        label="ML Features (Crash)",
        value=f"{summary.get('ml_crash_features', 0)}",
        help="Features in crash-level ML dataset"
    )

st.markdown("---")

# Project Overview Section
st.markdown("##  Project Goals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
    <h3> Objectives</h3>
    <ul>
        <li><b>Crash Analysis</b>: Understand patterns in Texas traffic crashes (2016-2023)</li>
        <li><b>Work Zone Safety</b>: Analyze work zone locations and crash risk</li>
        <li><b>ML Modeling</b>: Predict crash severity and segment-level risk</li>
        <li><b>Interactive Tool</b>: Build work zone risk prediction application</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <h3> Available Datasets</h3>
    <ul>
        <li><b>Raw Crashes</b>: Kaggle US Accidents (213 MB) + Austin (81 MB)</li>
        <li><b>Work Zones</b>: TxDOT WZDx feed (2,180 zones)</li>
        <li><b>Weather</b>: NOAA daily weather (2016-2023)</li>
        <li><b>Traffic</b>: TxDOT AADT traffic volumes</li>
        <li><b>ML Datasets</b>: Crash-level (1.1M rows) + Segment-level (303K rows)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Dataset Summary Cards
st.markdown("##  Dataset Summary")

tab1, tab2, tab3 = st.tabs([" Raw Data", " ML Datasets", " File Sizes"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4> Crash Data</h4>
        <ul>
            <li><b>Kaggle Crashes</b>: {:,}</li>
            <li><b>Austin Crashes</b>: {:,}</li>
            <li><b>Total</b>: {:,}</li>
            <li><b>Date Range</b>: 2016-2023</li>
        </ul>
        </div>
        """.format(
            summary.get('kaggle_crashes', 0),
            summary.get('austin_crashes', 0),
            summary.get('total_crashes', 0)
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h4> Work Zones</h4>
        <ul>
            <li><b>Total Zones</b>: {:,}</li>
            <li><b>Counties</b>: {}</li>
            <li><b>Source</b>: TxDOT WZDx</li>
            <li><b>Status</b>: Active feed</li>
        </ul>
        </div>
        """.format(
            summary.get('work_zones', 0),
            summary.get('wz_counties', 0)
        ), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="success-box">
        <h4> Weather & Traffic</h4>
        <ul>
            <li><b>Weather</b>: NOAA daily data</li>
            <li><b>Traffic</b>: TxDOT AADT</li>
            <li><b>Stations</b>: 41,467</li>
            <li><b>Coverage</b>: Statewide</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h4> Crash-Level ML Dataset</h4>
        <ul>
            <li><b>Total Rows</b>: 1,135,762 crashes</li>
            <li><b>Features</b>: {} (78 engineered)</li>
            <li><b>Target</b>: high_severity (binary)</li>
            <li><b>Splits</b>: Train (2016-2021) / Val (2022) / Test (2023)</li>
            <li><b>Use Case</b>: Individual crash severity prediction</li>
        </ul>
        </div>
        """.format(summary.get('ml_crash_features', 81)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4> Segment-Level ML Dataset</h4>
        <ul>
            <li><b>Total Rows</b>: 303,281 segment-quarters</li>
            <li><b>Features</b>: {} (39 aggregated)</li>
            <li><b>Segments</b>: {:,} unique road segments</li>
            <li><b>Targets</b>: crash_count, severity_rate, risk_score</li>
            <li><b>Use Case</b>: Work zone risk prediction</li>
        </ul>
        </div>
        """.format(
            summary.get('ml_segment_features', 39),
            summary.get('unique_segments', 0)
        ), unsafe_allow_html=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Raw Data Files")
        st.markdown(f"- **Kaggle Crashes**: {file_sizes.get('kaggle_crashes', 'N/A')}")
        st.markdown(f"- **Austin Crashes**: {file_sizes.get('austin_crashes', 'N/A')}")
        st.markdown(f"- **Work Zones**: 2.7 MB (JSON)")
        st.markdown(f"- **Weather**: 4.2 MB")
        st.markdown(f"- **Traffic (AADT)**: 9.0 MB (GeoPackage)")

    with col2:
        st.markdown("###  ML Dataset Files")
        st.markdown(f"- **Crash ML (train)**: {file_sizes.get('crash_ml_train', 'N/A')}")
        st.markdown(f"- **Crash ML (val)**: 90 MB")
        st.markdown(f"- **Crash ML (test)**: 14 MB")
        st.markdown(f"- **Segment ML (train)**: {file_sizes.get('segment_ml_train', 'N/A')}")
        st.markdown(f"- **Segment ML (val+test)**: <1 MB")

st.markdown("---")

# Quick Visualization Section
st.markdown("##  Quick Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("###  Texas Overview")

    # Create simple Texas map
    m = folium.Map(
        location=TEXAS_CENTER,
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Add a simple marker for Texas center
    folium.Marker(
        location=TEXAS_CENTER,
        popup="Texas Center",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    st_folium(m, width=500, height=400, returned_objects=[])

    st.markdown("""
    <div class="info-box">
    <small>
    <b>Coverage Area</b>: Statewide Texas<br>
    <b>Focus</b>: Major highways and urban areas<br>
    <b>Data Density</b>: High in Dallas, Houston, Austin, San Antonio
    </small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("###  Dataset Composition")

    # Create dataset composition chart
    dataset_data = {
        'Dataset': ['Kaggle Crashes', 'Austin Crashes', 'Work Zones', 'Road Segments'],
        'Count': [
            summary.get('kaggle_crashes', 0),
            summary.get('austin_crashes', 0),
            summary.get('work_zones', 0),
            summary.get('unique_segments', 0)
        ]
    }

    fig = px.bar(
        dataset_data,
        x='Dataset',
        y='Count',
        title='Dataset Sizes',
        color='Dataset',
        text='Count'
    )

    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_title='Record Count'
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Navigation Guide
st.markdown("##  Navigation Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
    <h4> Raw Data Pages</h4>
    <ul>
        <li><b>Crashes Explorer</b>: Temporal & geographic analysis</li>
        <li><b>Work Zones</b>: Active zones visualization</li>
        <li><b>Weather Patterns</b>: Climate analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <h4> ML Dataset Pages</h4>
    <ul>
        <li><b>Crash-Level</b>: Feature analysis & distributions</li>
        <li><b>Segment-Level</b>: Risk patterns & aggregates</li>
        <li><b>Model Ready</b>: Train/val/test splits</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
    <h4> Future Pages</h4>
    <ul>
        <li><b>Risk Prediction</b>: Work zone risk tool</li>
        <li><b>Model Performance</b>: ML model results</li>
        <li><b>Insights</b>: Key findings & recommendations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #95a5a6; margin-top: 3rem;'>
    <p><b>SIADS 699 Capstone Project</b> | University of Michigan School of Information</p>
    <p>Texas Crash Analysis & Work Zone Risk Prediction | 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("##  Dashboard Info")
    st.markdown("---")

    st.markdown("###  Datasets")
    st.markdown(f"✅ Crashes: {summary.get('total_crashes', 0):,}")
    st.markdown(f"✅ Work Zones: {summary.get('work_zones', 0):,}")
    st.markdown(f"✅ Segments: {summary.get('unique_segments', 0):,}")

    st.markdown("---")

    st.markdown("###  Pages")
    st.markdown("-  Home (current)")
    st.markdown("-  Raw Data Analysis")
    st.markdown("-  ML Datasets")
    st.markdown("-  Risk Prediction (coming)")

    st.markdown("---")

    st.markdown("###  Resources")
    st.markdown("[GitHub Repository](https://github.com/JewelsHovan/Capstone-SIADS699)")
    st.markdown("[Google Drive Data](https://drive.google.com/drive/folders/1xVGXbxUFHSdSawo2C9wnmABj15wPEX3A)")

    st.markdown("---")
    st.markdown("### ℹ About")
    st.markdown("""
    This dashboard provides comprehensive analysis of Texas traffic crashes
    and ML datasets for crash risk prediction.
    """)
