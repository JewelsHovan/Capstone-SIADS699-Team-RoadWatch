"""
Segment-Level ML Dataset Page
Analysis and visualization of the segment-level machine learning dataset
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, CUSTOM_CSS, SEGMENT_ML_FEATURES, RISK_COLORS
from utils.data_loader import load_segment_ml_dataset
from utils.visualizations import (
    create_risk_distribution_chart,
    create_correlation_heatmap,
    create_feature_histogram,
    create_time_series_with_trend
)
from utils.map_utils import create_segment_map
from streamlit_folium import st_folium

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ Segment-Level ML Dataset</h1>', unsafe_allow_html=True)
st.markdown("**Machine Learning dataset for road segment risk prediction**")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.markdown("## 🔧 Dataset Controls")

    split = st.selectbox(
        "Select Split",
        ["train", "val", "test"],
        help="Choose which dataset split to analyze"
    )

    sample_size = st.slider(
        "Sample Size",
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help="Number of rows to load"
    )

# Load data
with st.spinner(f"Loading {split} dataset..."):
    df = load_segment_ml_dataset(split=split, sample_size=sample_size)

# Overview metrics
st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Segment-Quarters", f"{len(df):,}")

with col2:
    if 'segment_id' in df.columns:
        st.metric("Unique Segments", f"{df['segment_id'].nunique():,}")
    else:
        st.metric("Unique Segments", "N/A")

with col3:
    if 'crash_count' in df.columns:
        avg_crashes = df['crash_count'].mean()
        st.metric("Avg Crashes/Quarter", f"{avg_crashes:.2f}")
    else:
        st.metric("Avg Crashes/Quarter", "N/A")

with col4:
    if 'risk_category' in df.columns:
        high_risk_pct = (df['risk_category'].isin(['HIGH', 'VERY_HIGH'])).mean()
        st.metric("High Risk Segments", f"{high_risk_pct:.1%}")
    else:
        st.metric("High Risk Segments", "N/A")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset Info",
    "🎯 Target Variables",
    "📊 Feature Analysis",
    "🗺️ Geographic View",
    "📈 Temporal Patterns"
])

with tab1:
    st.markdown("### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📁 Dataset Details</h4>
        <ul>
            <li><b>Purpose</b>: Road segment risk prediction</li>
            <li><b>Granularity</b>: Quarterly aggregation</li>
            <li><b>Total Features</b>: 39 features</li>
            <li><b>Rows (full)</b>: 303,281 segment-quarters</li>
            <li><b>Unique Segments</b>: 75,650</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Target Variables</h4>
        <ul>
            <li><b>crash_count</b>: Number of crashes</li>
            <li><b>severity_rate</b>: Proportion high-severity</li>
            <li><b>traffic_impact</b>: AADT × severity_rate</li>
            <li><b>risk_score_simple</b>: Composite score</li>
            <li><b>risk_category</b>: LOW/MED/HIGH/VERY_HIGH</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature categories
    st.markdown("### 📂 Feature Categories")

    feature_cats = {
        "Temporal": SEGMENT_ML_FEATURES['temporal'],
        "Location": SEGMENT_ML_FEATURES['location'],
        "Road": SEGMENT_ML_FEATURES['road'],
        "Aggregates": SEGMENT_ML_FEATURES['aggregates'],
        "Traffic": SEGMENT_ML_FEATURES['traffic'],
        "Weather": SEGMENT_ML_FEATURES['weather'],
        "Targets": SEGMENT_ML_FEATURES['targets']
    }

    for category, features in feature_cats.items():
        existing = [f for f in features if f in df.columns]
        if existing:
            st.markdown(f"**{category}** ({len(existing)} features): {', '.join(existing)}")

with tab2:
    st.markdown("### Target Variable Analysis")

    if 'risk_category' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Risk category distribution
            fig_risk = create_risk_distribution_chart(df, risk_col='risk_category')
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            # Risk category stats
            risk_counts = df['risk_category'].value_counts()

            st.markdown("""
            <div class="info-box">
            <h4>📊 Risk Distribution</h4>
            """, unsafe_allow_html=True)

            for risk_level in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
                count = risk_counts.get(risk_level, 0)
                pct = (count / len(df)) * 100
                st.markdown(f"- **{risk_level}**: {count:,} ({pct:.1f}%)")

            st.markdown("</div>", unsafe_allow_html=True)

    # Target variable statistics
    st.markdown("### 🎯 Target Variable Statistics")

    target_vars = ['crash_count', 'severity_rate', 'traffic_impact', 'risk_score_simple']
    available_targets = [t for t in target_vars if t in df.columns]

    if available_targets:
        col1, col2, col3, col4 = st.columns(len(available_targets))
        cols = [col1, col2, col3, col4]

        for i, target in enumerate(available_targets):
            with cols[i]:
                st.markdown(f"**{target}**")
                st.markdown(f"Mean: {df[target].mean():.2f}")
                st.markdown(f"Median: {df[target].median():.2f}")
                st.markdown(f"Std: {df[target].std():.2f}")

        # Distribution plots
        selected_target = st.selectbox("Select Target for Distribution", available_targets)

        col1, col2 = st.columns(2)

        with col1:
            fig_dist = create_feature_histogram(df, selected_target, bins=50)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            if 'risk_category' in df.columns:
                fig_box = px.box(
                    df,
                    x='risk_category',
                    y=selected_target,
                    title=f'{selected_target} by Risk Category',
                    color='risk_category',
                    color_discrete_map=RISK_COLORS
                )
                st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.markdown("### Feature Analysis")

    # Select feature
    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in ['segment_id']]

    if numeric_features:
        selected_feature = st.selectbox("Select Feature", numeric_features)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_hist = create_feature_histogram(df, selected_feature, bins=50)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown(f"### 📈 {selected_feature}")

            feature_data = df[selected_feature].dropna()

            st.markdown(f"""
            <div class="info-box">
            <b>Count:</b> {len(feature_data):,}<br>
            <b>Mean:</b> {feature_data.mean():.2f}<br>
            <b>Median:</b> {feature_data.median():.2f}<br>
            <b>Std:</b> {feature_data.std():.2f}<br>
            <b>Min:</b> {feature_data.min():.2f}<br>
            <b>Max:</b> {feature_data.max():.2f}
            </div>
            """, unsafe_allow_html=True)

        # Feature correlations
        st.markdown("### 🔗 Feature Correlations with Targets")

        if available_targets:
            corr_data = []

            for target in available_targets:
                corr = df[[selected_feature, target]].corr().iloc[0, 1]
                corr_data.append({'Target': target, 'Correlation': corr})

            corr_df = pd.DataFrame(corr_data)

            fig_corr_bar = px.bar(
                corr_df,
                x='Target',
                y='Correlation',
                title=f'Correlation of {selected_feature} with Target Variables',
                color='Correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1]
            )

            st.plotly_chart(fig_corr_bar, use_container_width=True)

with tab4:
    st.markdown("### Geographic Distribution")

    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns and 'risk_category' in df.columns:
        map_sample = st.slider(
            "Map Sample Size",
            min_value=100,
            max_value=1000,
            value=500,
            help="Number of segments to show on map"
        )

        with st.spinner("Generating segment map..."):
            m = create_segment_map(
                df,
                lat_col='Start_Lat',
                lon_col='Start_Lng',
                risk_col='risk_category',
                max_points=map_sample
            )

            st_folium(m, width=1200, height=600, returned_objects=[])

        st.info(f"Showing {map_sample} segments colored by risk level")

    else:
        st.warning("Geographic data not available")

    # Geographic statistics
    if 'City' in df.columns:
        st.markdown("### 📍 Top Cities by Crash Count")

        city_crashes = df.groupby('City')['crash_count'].sum().sort_values(ascending=False).head(15)

        fig_cities = px.bar(
            x=city_crashes.index,
            y=city_crashes.values,
            title='Top 15 Cities by Total Crashes',
            labels={'x': 'City', 'y': 'Total Crashes'}
        )

        fig_cities.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cities, use_container_width=True)

with tab5:
    st.markdown("### Temporal Patterns")

    if 'year_quarter' in df.columns and 'crash_count' in df.columns:
        # Aggregate by time
        temporal_agg = df.groupby('year_quarter').agg({
            'crash_count': 'sum',
            'severity_rate': 'mean',
            'segment_id': 'nunique'
        }).reset_index()

        temporal_agg.columns = ['year_quarter', 'total_crashes', 'avg_severity_rate', 'num_segments']

        # Crashes over time
        col1, col2 = st.columns(2)

        with col1:
            fig_temporal = px.line(
                temporal_agg,
                x='year_quarter',
                y='total_crashes',
                title='Total Crashes by Quarter',
                markers=True
            )

            st.plotly_chart(fig_temporal, use_container_width=True)

        with col2:
            fig_severity = px.line(
                temporal_agg,
                x='year_quarter',
                y='avg_severity_rate',
                title='Average Severity Rate by Quarter',
                markers=True
            )

            st.plotly_chart(fig_severity, use_container_width=True)

        # Seasonality analysis
        if 'quarter' in df.columns:
            st.markdown("### 📅 Seasonal Patterns")

            seasonal = df.groupby('quarter')['crash_count'].mean().reset_index()

            fig_seasonal = px.bar(
                seasonal,
                x='quarter',
                y='crash_count',
                title='Average Crashes by Quarter (Seasonality)',
                labels={'quarter': 'Quarter', 'crash_count': 'Avg Crashes'}
            )

            st.plotly_chart(fig_seasonal, use_container_width=True)

    else:
        st.warning("Temporal data not available")

st.markdown("---")

# Data preview
with st.expander("📋 View Data Sample"):
    st.dataframe(df.head(100), use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Sample Data (CSV)",
        data=csv,
        file_name=f"segment_ml_{split}_sample.csv",
        mime="text/csv"
    )

# Sidebar - Current view
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Current View")
    st.markdown(f"**Split:** {split}")
    st.markdown(f"**Segment-Quarters:** {len(df):,}")
    if 'segment_id' in df.columns:
        st.markdown(f"**Segments:** {df['segment_id'].nunique():,}")
    if 'crash_count' in df.columns:
        st.markdown(f"**Avg Crashes:** {df['crash_count'].mean():.2f}")
