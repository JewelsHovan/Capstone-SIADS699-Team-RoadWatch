"""
Segment-Level Risk Map - Predict crash risk for road segments in selected areas
Uses trained segment-level ML model to identify high-risk road segments
"""

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path (insert at beginning to prioritize app/config.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, CUSTOM_CSS, TEXAS_CENTER, HPMS_TEXAS_2023, SEGMENT_LEVEL_ML_DIR
from utils.hpms_loader import load_hpms_full, load_hpms_for_location, get_segment_at_location
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import box as bbox_geom

# Page configuration
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Paths
HPMS_FILE = HPMS_TEXAS_2023
ML_DATASETS_DIR = SEGMENT_LEVEL_ML_DIR

# Title
st.markdown('<h1 class="main-header">Road Segment Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1rem; color: #7f8c8d; margin-bottom: 2rem;'>
    <b>Infrastructure Planning Tool</b> | Identify high-risk road segments for proactive safety improvements
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar: Model selection and parameters
with st.sidebar:
    st.markdown("## Model Settings")
    st.markdown("---")

    st.markdown("### Model Type")
    model_type = st.selectbox(
        "Select Model",
        ["Simple Baseline (Demo)", "Random Forest", "XGBoost", "Zero-Inflated NB"],
        help="Choose prediction model. Baseline uses simple heuristics for demo."
    )

    st.markdown("---")
    st.markdown("### Risk Thresholds")

    high_risk_threshold = st.slider(
        "High Risk Threshold",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Segments with predicted crashes > this value are marked high risk"
    )

    medium_risk_threshold = st.slider(
        "Medium Risk Threshold",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Segments with predicted crashes > this value are marked medium risk"
    )

    st.markdown("---")
    st.markdown("### Display Options")

    show_segment_labels = st.checkbox("Show Segment IDs", value=False)
    show_feature_values = st.checkbox("Show Feature Values", value=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool predicts crash risk for road segments based on:
    - Speed limit
    - Number of lanes
    - Road type (Interstate, Arterial, Local)
    - Traffic volume (AADT)
    - Urban/Rural location
    """)


# Main content tabs
tab1, tab2, tab3 = st.tabs(["Risk Map", "Analysis", "Recommendations"])

with tab1:
    st.markdown("### Select Area of Interest")
    st.info(" Draw a box or polygon on the map to select an area for risk assessment")

    # Create map with drawing tools
    m = folium.Map(
        location=TEXAS_CENTER,
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Add drawing tools
    draw = Draw(
        export=True,
        draw_options={
            'polyline': False,
            'polygon': True,
            'circle': False,
            'rectangle': True,
            'marker': False,
            'circlemarker': False,
        }
    )
    draw.add_to(m)

    # Display map with key to prevent re-rendering
    map_data = st_folium(
        m,
        width=900,
        height=600,
        returned_objects=["all_drawings"],
        key="segment_risk_map"
    )

    # Process selected area - only if we have valid drawings
    if map_data and map_data.get("all_drawings") and len(map_data["all_drawings"]) > 0:
        drawings = map_data["all_drawings"]

        if drawings and len(drawings) > 0:
            # Check if this is a new drawing (different from last time)
            need_to_process = False
            if 'last_bbox' not in st.session_state or st.session_state['last_bbox'] != drawings:
                st.session_state['last_bbox'] = drawings
                need_to_process = True

            # Process segments if this is a new drawing
            if need_to_process:
                st.success(f"✅ Area selected! Analyzing road segments...")

                # Extract bounding box
                drawing = drawings[0]
                geometry = drawing.get("geometry", {})
                coordinates = geometry.get("coordinates", [])[0]

                lats = [point[1] for point in coordinates]
                lons = [point[0] for point in coordinates]

                bbox = {
                    "north": max(lats),
                    "south": min(lats),
                    "east": max(lons),
                    "west": min(lons)
                }

                # Load HPMS segments in area (using cached loader for performance)
                try:
                    # Load full HPMS (cached - instant after first load)
                    hpms_full = load_hpms_full()

                    # Filter to the drawn area
                    bbox_shape = bbox_geom(bbox['west'], bbox['south'], bbox['east'], bbox['north'])
                    hpms = hpms_full[hpms_full.intersects(bbox_shape)].copy()

                    if len(hpms) == 0:
                        st.warning("No road segments found in selected area. Try selecting a larger area.")
                    else:
                        st.success(f"Found {len(hpms):,} road segments in selected area")

                        # Prepare features for prediction
                        hpms['segment_id'] = hpms.index.astype(str)

                        # Convert HPMS string types to numeric (fix pandas warnings)
                        for col in ['speed_limit', 'through_lanes', 'f_system', 'urban_id', 'aadt']:
                            if col in hpms.columns:
                                # Replace NULL strings with NaN, then convert to numeric
                                hpms[col] = pd.to_numeric(
                                    hpms[col].replace({'NULL': np.nan}).infer_objects(copy=False),
                                    errors='coerce'
                                )

                        # Create interaction features
                        if 'speed_limit' in hpms.columns and 'aadt' in hpms.columns:
                            hpms['speed_x_aadt'] = hpms['speed_limit'] * hpms['aadt']
                        if 'f_system' in hpms.columns and 'urban_id' in hpms.columns:
                            hpms['fsystem_x_urban'] = hpms['f_system'] * hpms['urban_id']
                        if 'through_lanes' in hpms.columns and 'aadt' in hpms.columns:
                            hpms['lanes_x_aadt'] = hpms['through_lanes'] * hpms['aadt']

                        # Simple baseline prediction (replace with trained model)
                        st.info(" Using baseline prediction model (demo). Replace with trained model for production.")

                        # Baseline Risk Model (Deterministic):
                        # - Speed limit: 30% weight (normalized 0-1)
                        # - AADT traffic: 40% weight (normalized 0-1)
                        # - Lane count: 30% weight
                        # Note: This is a simple rule-based model for demo purposes.
                        # For production, replace with trained ML model.
                        hpms['predicted_risk'] = (
                            hpms['speed_limit'].fillna(0) / 10 * 0.3 +
                            hpms['aadt'].fillna(0) / 10000 * 0.4 +
                            hpms['through_lanes'].fillna(0) * 0.3
                        ).clip(lower=0)

                        # Classify risk levels
                        hpms['risk_level'] = pd.cut(
                            hpms['predicted_risk'],
                            bins=[-np.inf, medium_risk_threshold, high_risk_threshold, np.inf],
                            labels=['Low', 'Medium', 'High']
                        )

                        # Store in session state so results persist across reruns
                        st.session_state['hpms'] = hpms
                        st.session_state['bbox'] = bbox

                except Exception as e:
                    st.error(f"Error loading segments: {e}")
                    st.write("Make sure HPMS data file exists at:", HPMS_FILE)

            # Display results if we have processed data (either just now or from previous run)
            if 'hpms' in st.session_state:
                hpms = st.session_state['hpms']
                bbox = st.session_state.get('bbox', {})

                # Display risk distribution
                col1, col2, col3 = st.columns(3)
                with col1:
                    low_count = (hpms['risk_level'] == 'Low').sum()
                    st.metric("Low Risk Segments", f"{low_count:,}",
                             f"{low_count/len(hpms)*100:.1f}%")
                with col2:
                    med_count = (hpms['risk_level'] == 'Medium').sum()
                    st.metric("Medium Risk Segments", f"{med_count:,}",
                             f"{med_count/len(hpms)*100:.1f}%")
                with col3:
                    high_count = (hpms['risk_level'] == 'High').sum()
                    st.metric("High Risk Segments", f"{high_count:,}",
                             f"{high_count/len(hpms)*100:.1f}%")

                # Create risk map
                st.markdown("### Risk Map with Color-Coded Segments")

                risk_map = folium.Map(
                    location=[(bbox['north'] + bbox['south'])/2, (bbox['east'] + bbox['west'])/2],
                    zoom_start=12,
                    tiles='OpenStreetMap'
                )

                # Add segments colored by risk
                for idx, row in hpms.iterrows():
                    if row['risk_level'] == 'High':
                        color = '#e74c3c'  # Red
                    elif row['risk_level'] == 'Medium':
                        color = '#f39c12'  # Orange
                    else:
                        color = '#27ae60'  # Green

                    # Create popup text
                    popup_text = f"""
                    <b>Segment ID:</b> {row['segment_id']}<br>
                    <b>Predicted Risk:</b> {row['predicted_risk']:.2f} crashes/5yrs<br>
                    <b>Risk Level:</b> {row['risk_level']}<br>
                    <b>Speed Limit:</b> {row['speed_limit']:.0f} mph<br>
                    <b>Lanes:</b> {row['through_lanes']:.0f}<br>
                    <b>AADT:</b> {row['aadt']:.0f} vehicles/day<br>
                    <b>Road Type:</b> {row['f_system']:.0f}
                    """

                    # Add to map
                    folium.GeoJson(
                        row['geometry'],
                        style_function=lambda x, color=color: {
                            'color': color,
                            'weight': 3,
                            'opacity': 0.8
                        },
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(risk_map)

                # Display map
                st_folium(risk_map, width=900, height=600, key="risk_results_map")

                # Top 10 highest risk segments
                st.markdown("### Top 10 Highest Risk Segments")
                top_10 = hpms.nlargest(10, 'predicted_risk')[[
                    'segment_id', 'predicted_risk', 'risk_level',
                    'speed_limit', 'through_lanes', 'aadt', 'f_system'
                ]].copy()

                top_10.columns = ['Segment ID', 'Predicted Crashes', 'Risk Level',
                                 'Speed Limit', 'Lanes', 'AADT', 'Road Type']

                st.dataframe(
                    top_10.style.background_gradient(subset=['Predicted Crashes'], cmap='Reds'),
                    use_container_width=True
                )

                # Download options
                st.markdown("---")
                st.download_button(
                    label="Download Risk Assessment (CSV)",
                    data=hpms[['segment_id', 'predicted_risk', 'risk_level', 'speed_limit',
                              'through_lanes', 'aadt', 'f_system']].to_csv(index=False),
                    file_name="segment_risk_assessment.csv",
                    mime="text/csv"
                )
        else:
            st.info(" Draw an area on the map to start risk assessment")
    else:
        st.info(" Draw an area on the map to start risk assessment")

with tab2:
    st.markdown("### Risk Analysis")

    # Check if we have processed data in session state
    if 'hpms' in st.session_state:
        hpms = st.session_state['hpms']

        # Feature importance (mock - replace with actual model feature importance)
        st.markdown("#### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['AADT (Traffic Volume)', 'Speed Limit', 'Number of Lanes',
                       'Speed × AADT', 'Road Type × Urban'],
            'Importance': [0.35, 0.25, 0.20, 0.12, 0.08]
        })

        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Risk Prediction'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk distribution by road type
        st.markdown("#### Risk Distribution by Road Type")
        if 'f_system' in hpms.columns:
            road_type_map = {
                1: 'Interstate',
                2: 'Freeway',
                3: 'Principal Arterial',
                4: 'Minor Arterial',
                5: 'Major Collector',
                6: 'Minor Collector',
                7: 'Local'
            }
            hpms_copy = hpms.copy()
            hpms_copy['road_type_name'] = hpms_copy['f_system'].map(road_type_map).fillna('Unknown')

            # Fix pandas warning by adding observed parameter
            risk_by_type = hpms_copy.groupby(['road_type_name', 'risk_level'], observed=False).size().unstack(fill_value=0)

            fig = px.bar(
                risk_by_type,
                title='Risk Distribution by Road Type',
                labels={'value': 'Number of Segments', 'road_type_name': 'Road Type'},
                color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Risk vs AADT scatter - only show if we have variation in risk levels
        st.markdown("#### Risk vs Traffic Volume")
        if len(hpms['risk_level'].unique()) > 1:
            # Only include categories that exist in the data
            available_colors = {}
            available_categories = []
            if 'Low' in hpms['risk_level'].values:
                available_colors['Low'] = '#27ae60'
                available_categories.append('Low')
            if 'Medium' in hpms['risk_level'].values:
                available_colors['Medium'] = '#f39c12'
                available_categories.append('Medium')
            if 'High' in hpms['risk_level'].values:
                available_colors['High'] = '#e74c3c'
                available_categories.append('High')

            fig = px.scatter(
                hpms,
                x='aadt',
                y='predicted_risk',
                color='risk_level',
                title='Predicted Risk vs Traffic Volume (AADT)',
                labels={'aadt': 'AADT (vehicles/day)', 'predicted_risk': 'Predicted Crashes (5 years)'},
                color_discrete_map=available_colors,
                category_orders={'risk_level': available_categories}  # Use dynamic list
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough variation in risk levels to show scatter plot. All segments are in the same risk category.")

    else:
        st.info("Select an area in the Risk Map tab to see analysis")

with tab3:
    st.markdown("### Safety Recommendations")

    # Check if we have processed data in session state
    if 'hpms' in st.session_state:
        hpms = st.session_state['hpms']
        high_risk_segments = hpms[hpms['risk_level'] == 'High']

        if len(high_risk_segments) > 0:
            st.markdown("#### Priority Actions for High-Risk Segments")

            # Speed limit reduction candidates
            high_speed = high_risk_segments[high_risk_segments['speed_limit'] > 55]
            if len(high_speed) > 0:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Speed Limit Reduction</h4>
                <p><b>{len(high_speed)} segments</b> have speed limits > 55 mph and high predicted risk.</p>
                <p><b>Recommendation:</b> Consider reducing speed limits by 5-10 mph on these segments.</p>
                </div>
                """, unsafe_allow_html=True)

            # High traffic segments
            high_traffic = high_risk_segments[high_risk_segments['aadt'] > 50000]
            if len(high_traffic) > 0:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Congestion Management</h4>
                <p><b>{len(high_traffic)} segments</b> have very high traffic (>50K AADT) and high risk.</p>
                <p><b>Recommendation:</b> Implement intelligent traffic management, add lanes, or improve signalization.</p>
                </div>
                """, unsafe_allow_html=True)

            # Narrow roads with high risk
            narrow = high_risk_segments[high_risk_segments['through_lanes'] <= 2]
            if len(narrow) > 0:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Capacity Expansion</h4>
                <p><b>{len(narrow)} segments</b> have ≤2 lanes and high risk.</p>
                <p><b>Recommendation:</b> Evaluate feasibility of adding lanes or creating bypass routes.</p>
                </div>
                """, unsafe_allow_html=True)

            # Infrastructure improvements
            st.markdown("""
            <div class="info-box">
            <h4>General Infrastructure Improvements</h4>
            <ul>
                <li>Improve road surface quality (reduce IRI)</li>
                <li>Add or improve shoulders</li>
                <li>Install rumble strips</li>
                <li>Enhance lighting at high-risk segments</li>
                <li>Add median barriers on high-speed segments</li>
                <li>Install variable speed limit signs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            # Enforcement recommendations
            st.markdown("""
            <div class="success-box">
            <h4>Enforcement Priorities</h4>
            <ul>
                <li>Increase patrol presence on high-risk segments</li>
                <li>Deploy speed cameras in top 10 highest risk areas</li>
                <li>Focus enforcement during peak traffic hours</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("✅ No high-risk segments identified in this area!")
    else:
        st.info("Select an area in the Risk Map tab to see recommendations")
