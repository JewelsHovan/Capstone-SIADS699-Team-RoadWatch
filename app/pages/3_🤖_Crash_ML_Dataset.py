"""
Crash-Level ML Dataset Page
Analysis and visualization of the crash-level machine learning dataset
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path (insert at beginning to prioritize app/config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, CUSTOM_CSS, CRASH_ML_FEATURES, DEFAULT_SAMPLE_SIZES
from utils.data_loader import load_crash_ml_dataset
from utils.visualizations import (
    create_severity_pie_chart,
    create_correlation_heatmap,
    create_feature_histogram,
    create_box_plot
)

# Page config
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Crash-Level ML Dataset</h1>', unsafe_allow_html=True)
st.markdown("**Machine Learning dataset for individual crash severity prediction**")
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
        max_value=100000,
        value=50000,
        step=1000,
        help="Number of rows to load (for performance)"
    )

    st.markdown("---")
    st.markdown(f"### 📊 Split Info")
    split_info = {
        "train": "2016-2021 (Train)",
        "val": "2022 (Validation)",
        "test": "2023 (Test)"
    }
    st.info(split_info[split])

# Load data
with st.spinner(f"Loading {split} dataset..."):
    df = load_crash_ml_dataset(split=split, sample_size=sample_size)

# Overview metrics
st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", f"{len(df):,}")

with col2:
    st.metric("Total Features", f"{len(df.columns)}")

with col3:
    if 'high_severity' in df.columns:
        high_sev_rate = df['high_severity'].mean()
        st.metric("High Severity Rate", f"{high_sev_rate:.2%}")
    else:
        st.metric("High Severity Rate", "N/A")

with col4:
    # Calculate missing data percentage
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    st.metric("Missing Data", f"{missing_pct:.1f}%")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset Info",
    "🎯 Target Analysis",
    "📊 Feature Distributions",
    "🔗 Correlations",
    "🔍 Data Quality"
])

with tab1:
    st.markdown("### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📁 Dataset Details</h4>
        <ul>
            <li><b>Purpose</b>: Individual crash severity prediction</li>
            <li><b>Target Variable</b>: high_severity (binary)</li>
            <li><b>Total Features</b>: 78 engineered features</li>
            <li><b>Rows (full)</b>: 1,135,762 crashes</li>
            <li><b>Date Range</b>: 2016-2023</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Use Cases</h4>
        <ul>
            <li>Predict crash severity at time of occurrence</li>
            <li>Identify high-risk conditions</li>
            <li>Allocate emergency resources</li>
            <li>Understand severity factors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature categories
    st.markdown("### 📂 Feature Categories")

    feature_cats = {
        "Temporal": CRASH_ML_FEATURES['temporal'],
        "Location": CRASH_ML_FEATURES['location'],
        "Road Characteristics": CRASH_ML_FEATURES['road'],
        "Traffic": CRASH_ML_FEATURES['traffic'],
        "Weather": CRASH_ML_FEATURES['weather'],
        "Infrastructure": CRASH_ML_FEATURES['infrastructure'],
        "Target": CRASH_ML_FEATURES['target']
    }

    for category, features in feature_cats.items():
        # Count how many features exist in df
        existing = [f for f in features if f in df.columns]
        st.markdown(f"**{category}** ({len(existing)} features): {', '.join(existing[:10])}...")

with tab2:
    st.markdown("### Target Variable Analysis")

    if 'high_severity' in df.columns and 'Severity' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # High severity distribution
            high_sev_counts = df['high_severity'].value_counts()

            fig_target = px.pie(
                values=high_sev_counts.values,
                names=['Low Severity (1-2)', 'High Severity (3-4)'],
                title='Target Variable Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )

            st.plotly_chart(fig_target, use_container_width=True)

            # Class balance
            st.markdown(f"""
            <div class="info-box">
            <b>Class Balance:</b><br>
            - Low Severity: {high_sev_counts.get(0, 0):,} ({high_sev_counts.get(0, 0)/len(df)*100:.1f}%)<br>
            - High Severity: {high_sev_counts.get(1, 0):,} ({high_sev_counts.get(1, 0)/len(df)*100:.1f}%)<br>
            <br>
            <b>Imbalance Ratio:</b> {high_sev_counts.get(0, 0) / high_sev_counts.get(1, 1):.2f}:1
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Detailed severity distribution
            fig_sev = create_severity_pie_chart(df, severity_col='Severity')
            st.plotly_chart(fig_sev, use_container_width=True)

            # Severity stats
            sev_counts = df['Severity'].value_counts().sort_index()
            st.markdown(f"""
            <div class="info-box">
            <b>Severity Breakdown:</b><br>
            - Level 1: {sev_counts.get(1, 0):,}<br>
            - Level 2: {sev_counts.get(2, 0):,}<br>
            - Level 3: {sev_counts.get(3, 0):,}<br>
            - Level 4: {sev_counts.get(4, 0):,}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Target variables not found in dataset")

with tab3:
    st.markdown("### Feature Distributions")

    # Select feature category
    feature_category = st.selectbox(
        "Select Feature Category",
        list(feature_cats.keys())[:-1]  # Exclude target
    )

    available_features = [f for f in feature_cats[feature_category] if f in df.columns]

    if available_features:
        selected_feature = st.selectbox(
            "Select Feature",
            available_features
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Histogram
            if df[selected_feature].dtype in ['float64', 'int64']:
                fig_hist = create_feature_histogram(df, selected_feature, bins=50)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                # For categorical features, use value counts
                value_counts = df[selected_feature].value_counts().head(20)

                fig_cat = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Distribution of {selected_feature}',
                    labels={'x': selected_feature, 'y': 'Count'}
                )

                st.plotly_chart(fig_cat, use_container_width=True)

        with col2:
            # Statistics
            st.markdown(f"### 📈 {selected_feature}")

            if df[selected_feature].dtype in ['float64', 'int64']:
                feature_data = df[selected_feature].dropna()

                st.markdown(f"""
                <div class="info-box">
                <b>Count:</b> {len(feature_data):,}<br>
                <b>Missing:</b> {df[selected_feature].isnull().sum():,}<br>
                <b>Mean:</b> {feature_data.mean():.2f}<br>
                <b>Median:</b> {feature_data.median():.2f}<br>
                <b>Std:</b> {feature_data.std():.2f}<br>
                <b>Min:</b> {feature_data.min():.2f}<br>
                <b>Max:</b> {feature_data.max():.2f}
                </div>
                """, unsafe_allow_html=True)
            else:
                unique_count = df[selected_feature].nunique()
                missing_count = df[selected_feature].isnull().sum()

                st.markdown(f"""
                <div class="info-box">
                <b>Type:</b> Categorical<br>
                <b>Unique Values:</b> {unique_count}<br>
                <b>Missing:</b> {missing_count:,}<br>
                <b>Most Common:</b> {df[selected_feature].mode().values[0] if len(df[selected_feature].mode()) > 0 else 'N/A'}
                </div>
                """, unsafe_allow_html=True)

        # Feature by target (if target available)
        if 'high_severity' in df.columns and df[selected_feature].dtype in ['float64', 'int64']:
            st.markdown(f"### {selected_feature} by Target Variable")

            fig_box = create_box_plot(df, 'high_severity', selected_feature,
                                      title=f'{selected_feature} by High Severity')
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.warning(f"No features available in {feature_category} category")

with tab4:
    st.markdown("### Feature Correlations")

    # Select features for correlation
    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in ['ID']]

    if len(numeric_features) > 1:
        num_features = st.slider(
            "Number of Features to Display",
            min_value=5,
            max_value=min(20, len(numeric_features)),
            value=10
        )

        # Select top features by variance or correlation with target
        if 'high_severity' in df.columns:
            # Features most correlated with target
            correlations = df[numeric_features].corrwith(df['high_severity']).abs().sort_values(ascending=False)
            top_features = correlations.head(num_features).index.tolist()
        else:
            # Features with highest variance
            variances = df[numeric_features].var().sort_values(ascending=False)
            top_features = variances.head(num_features).index.tolist()

        # Create correlation heatmap
        fig_corr = create_correlation_heatmap(df, features=top_features)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations with target
        if 'high_severity' in df.columns:
            st.markdown("### 🎯 Features Most Correlated with Target")

            top_corrs = correlations.head(10)

            col1, col2 = st.columns(2)

            with col1:
                for i, (feature, corr) in enumerate(list(top_corrs.items())[:5]):
                    st.markdown(f"{i+1}. **{feature}**: {corr:.3f}")

            with col2:
                for i, (feature, corr) in enumerate(list(top_corrs.items())[5:10]):
                    st.markdown(f"{i+6}. **{feature}**: {corr:.3f}")

    else:
        st.warning("Not enough numeric features for correlation analysis")

with tab5:
    st.markdown("### Data Quality Assessment")

    # Missing data analysis
    st.markdown("#### Missing Data by Feature")

    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing %', ascending=False)

    # Only show features with missing data
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) > 0:
        fig_missing = px.bar(
            missing_df.head(20),
            x='Feature',
            y='Missing %',
            title='Top 20 Features by Missing Data Percentage',
            labels={'Missing %': 'Missing (%)'}
        )

        fig_missing.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)

        # Missing data table
        st.dataframe(missing_df, use_container_width=True, height=300)
    else:
        st.success("No missing data found!")

    # Data completeness by category
    st.markdown("#### Completeness by Feature Category")

    completeness_data = []

    for category, features in feature_cats.items():
        existing_features = [f for f in features if f in df.columns]
        if existing_features:
            category_completeness = (1 - df[existing_features].isnull().sum().sum() /
                                      (len(df) * len(existing_features))) * 100
            completeness_data.append({
                'Category': category,
                'Completeness %': category_completeness
            })

    if completeness_data:
        completeness_df = pd.DataFrame(completeness_data)

        fig_completeness = px.bar(
            completeness_df,
            x='Category',
            y='Completeness %',
            title='Data Completeness by Feature Category',
            color='Completeness %',
            color_continuous_scale='RdYlGn'
        )

        fig_completeness.update_layout(height=400)
        st.plotly_chart(fig_completeness, use_container_width=True)

st.markdown("---")

# Data preview
with st.expander("📋 View Data Sample"):
    st.dataframe(df.head(100), use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Sample Data (CSV)",
        data=csv,
        file_name=f"crash_ml_{split}_sample.csv",
        mime="text/csv"
    )

# Sidebar - Current view
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Current View")
    st.markdown(f"**Split:** {split}")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Features:** {len(df.columns)}")
    if 'high_severity' in df.columns:
        st.markdown(f"**High Severity:** {df['high_severity'].mean():.2%}")
