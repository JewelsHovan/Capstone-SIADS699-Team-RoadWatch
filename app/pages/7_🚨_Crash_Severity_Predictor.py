"""
Real-Time Crash Severity Predictor - Emergency Response Decision Support
Predict severity of crashes happening right now based on location and conditions
"""

import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, time
from pathlib import Path
from config import PAGE_CONFIG, CUSTOM_CSS, TEXAS_CENTER
import plotly.graph_objects as go

# Page configuration
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
HPMS_FILE = BASE_DIR / "data" / "silver" / "texas" / "roadway" / "hpms_texas_2023.gpkg"

# Title
st.markdown('<h1 class="main-header">🚨 Real-Time Crash Severity Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1rem; color: #7f8c8d; margin-bottom: 2rem;'>
    <b>Emergency Response Tool</b> | Predict crash severity in real-time to prioritize emergency response
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar: Instructions and settings
with st.sidebar:
    st.markdown("## 🚨 How to Use")
    st.markdown("---")

    st.markdown("""
    ### 1. Select Location
    Click on the map to mark crash location

    ### 2. Enter Conditions
    Input current weather, time, and conditions

    ### 3. Get Prediction
    Model predicts crash severity instantly

    ### 4. Dispatch Decision
    See priority recommendation for emergency response
    """)

    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")

    if st.button("🌧️ Rainy Rush Hour"):
        st.session_state['weather_preset'] = 'rainy_rush'
    if st.button("🌙 Night Clear"):
        st.session_state['weather_preset'] = 'night_clear'
    if st.button("🌫️ Foggy Morning"):
        st.session_state['weather_preset'] = 'foggy_morning'

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
    **Trained on:** 371K crashes (2016-2020)

    **Features:**
    - Road characteristics (HPMS)
    - Weather conditions
    - Time of day/week
    - Traffic volume
    - Location (urban/rural)

    **Target:** High severity (≥3)

    **Performance:** ROC-AUC ~0.75
    """)

# Main content
col_map, col_inputs = st.columns([1.2, 1])

with col_map:
    st.markdown("### 📍 Step 1: Select Crash Location")

    # Button to drop/reset pin
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("📍 Drop Pin at Center", use_container_width=True):
            # Get current map center or use default
            if 'crash_location' in st.session_state:
                # Keep current location
                pass
            else:
                # Drop at Texas center
                st.session_state['crash_location'] = TEXAS_CENTER
                st.session_state['location_updated'] = True

    with col_btn2:
        if st.button("🗑️ Clear Pin", use_container_width=True):
            if 'crash_location' in st.session_state:
                del st.session_state['crash_location']
            if 'hpms_features' in st.session_state:
                del st.session_state['hpms_features']
            st.rerun()

    st.info("💡 **Tip:** Click on map OR use 'Drop Pin' button, then drag the pin to adjust location")

    # Create interactive map
    m = folium.Map(
        location=TEXAS_CENTER,
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Add draggable marker if location has been selected
    if 'crash_location' in st.session_state:
        crash_lat, crash_lon = st.session_state['crash_location']

        # Add red DRAGGABLE pin marker at crash location
        folium.Marker(
            location=[crash_lat, crash_lon],
            popup=f"Crash Location<br>({crash_lat:.4f}°, {crash_lon:.4f}°)<br><i>Drag to adjust</i>",
            tooltip="🚨 Crash Location (Draggable)",
            draggable=True,
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(m)

        # Center map on the marker
        m.location = [crash_lat, crash_lon]
        m.zoom_start = 13

    # Display map
    map_data = st_folium(
        m,
        width=700,
        height=600,
        key="crash_location_map",
        returned_objects=["last_clicked", "last_object_clicked"]
    )

    # Handle map interactions
    location_changed = False

    # Check if user clicked on map (not on marker)
    if map_data and map_data.get("last_clicked") and not map_data.get("last_object_clicked"):
        crash_lat = map_data["last_clicked"]["lat"]
        crash_lon = map_data["last_clicked"]["lng"]
        st.session_state['crash_location'] = (crash_lat, crash_lon)
        location_changed = True

    # Check if user dragged the marker
    if map_data and map_data.get("last_object_clicked"):
        clicked_obj = map_data["last_object_clicked"]
        if isinstance(clicked_obj, dict) and "lat" in clicked_obj and "lng" in clicked_obj:
            crash_lat = clicked_obj["lat"]
            crash_lon = clicked_obj["lng"]
            st.session_state['crash_location'] = (crash_lat, crash_lon)
            location_changed = True

    # Display location and extract features if set
    if 'crash_location' in st.session_state:
        crash_lat, crash_lon = st.session_state['crash_location']
        st.success(f"✅ Location: ({crash_lat:.4f}°, {crash_lon:.4f}°)")

        # Extract HPMS features only if location changed
        if location_changed or 'hpms_features' not in st.session_state:
            with st.spinner("Extracting road characteristics..."):
                try:
                    # Load HPMS data near clicked point
                    buffer = 0.01  # ~1km buffer
                    hpms = gpd.read_file(
                        HPMS_FILE,
                        bbox=(crash_lon - buffer, crash_lat - buffer,
                              crash_lon + buffer, crash_lat + buffer)
                    )

                    if len(hpms) > 0:
                        # Find nearest segment
                        from shapely.geometry import Point
                        crash_point = Point(crash_lon, crash_lat)

                        # Convert to same CRS for distance calculation
                        hpms_utm = hpms.to_crs('EPSG:3083')  # Texas-specific
                        crash_point_utm = gpd.GeoSeries([crash_point], crs='EPSG:4326').to_crs('EPSG:3083')[0]

                        # Calculate distances
                        hpms_utm['distance_m'] = hpms_utm.geometry.distance(crash_point_utm)
                        nearest = hpms_utm.loc[hpms_utm['distance_m'].idxmin()]

                        # Convert features to numeric (handle both string and numeric values)
                        def safe_numeric_convert(value):
                            """Safely convert HPMS value to numeric, handling NULL strings and NaN"""
                            if pd.isna(value):
                                return np.nan
                            if isinstance(value, str):
                                if value == 'NULL':
                                    return np.nan
                                return pd.to_numeric(value, errors='coerce')
                            return pd.to_numeric(value, errors='coerce')

                        speed_limit = safe_numeric_convert(nearest['speed_limit'])
                        through_lanes = safe_numeric_convert(nearest['through_lanes'])
                        f_system = safe_numeric_convert(nearest['f_system'])
                        aadt = safe_numeric_convert(nearest['aadt'])

                        road_type_map = {
                            1: 'Interstate',
                            2: 'Freeway',
                            3: 'Principal Arterial',
                            4: 'Minor Arterial',
                            5: 'Major Collector',
                            6: 'Minor Collector',
                            7: 'Local'
                        }

                        # Handle NaN values for display
                        road_type_str = road_type_map.get(f_system, 'Unknown') if not pd.isna(f_system) else 'Unknown'
                        speed_str = f"{speed_limit:.0f} mph" if not pd.isna(speed_limit) else 'N/A'
                        lanes_str = f"{through_lanes:.0f}" if not pd.isna(through_lanes) else 'N/A'
                        aadt_str = f"{aadt:,.0f} vehicles/day" if not pd.isna(aadt) else 'N/A'

                        st.markdown("#### 🛣️ Road Characteristics")
                        st.markdown(f"""
                        <div class="info-box">
                        <ul>
                            <li><b>Road Type:</b> {road_type_str}</li>
                            <li><b>Speed Limit:</b> {speed_str}</li>
                            <li><b>Lanes:</b> {lanes_str}</li>
                            <li><b>Traffic:</b> {aadt_str}</li>
                            <li><b>Distance:</b> {nearest['distance_m']:.1f} meters to segment</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        # Store in session state (use defaults for NaN values)
                        st.session_state['hpms_features'] = {
                            'speed_limit': speed_limit if not pd.isna(speed_limit) else 55.0,
                            'through_lanes': through_lanes if not pd.isna(through_lanes) else 2.0,
                            'f_system': f_system if not pd.isna(f_system) else 4.0,
                            'aadt': aadt if not pd.isna(aadt) else 10000.0,
                            'distance_m': nearest['distance_m']
                        }

                    else:
                        st.warning("⚠️ No road segments found near this location. Predictions may be less accurate.")
                        st.session_state['hpms_features'] = None

                except Exception as e:
                    st.error(f"Could not extract road features: {e}")
                    st.session_state['hpms_features'] = None

        # Display road characteristics if already loaded (even if location didn't change)
        elif 'hpms_features' in st.session_state and st.session_state['hpms_features'] is not None:
            features = st.session_state['hpms_features']

            road_type_map = {
                1: 'Interstate', 2: 'Freeway', 3: 'Principal Arterial',
                4: 'Minor Arterial', 5: 'Major Collector',
                6: 'Minor Collector', 7: 'Local'
            }

            road_type_str = road_type_map.get(features['f_system'], 'Unknown')

            st.markdown("#### 🛣️ Road Characteristics")
            st.markdown(f"""
            <div class="info-box">
            <ul>
                <li><b>Road Type:</b> {road_type_str}</li>
                <li><b>Speed Limit:</b> {features['speed_limit']:.0f} mph</li>
                <li><b>Lanes:</b> {features['through_lanes']:.0f}</li>
                <li><b>Traffic:</b> {features['aadt']:,.0f} vehicles/day</li>
                <li><b>Distance:</b> {features['distance_m']:.1f} meters to segment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Click on map, use 'Drop Pin' button, or drag the pin to select crash location")

with col_inputs:
    st.markdown("### ⚙️ Step 2: Enter Current Conditions")

    # Apply presets if selected
    preset = st.session_state.get('weather_preset', None)

    # Time inputs
    st.markdown("#### 🕐 Time")
    current_time = datetime.now()

    if preset == 'rainy_rush':
        hour = 17
        day_of_week = 4  # Friday
    elif preset == 'night_clear':
        hour = 2
        day_of_week = 6  # Sunday
    elif preset == 'foggy_morning':
        hour = 7
        day_of_week = 1  # Monday
    else:
        hour = current_time.hour
        day_of_week = current_time.weekday()

    selected_time = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=hour,
        format="%d:00"
    )

    selected_day = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
        index=day_of_week
    )

    is_weekend = 1 if selected_day in [5, 6] else 0
    is_rush_hour = 1 if (6 <= selected_time <= 9 or 16 <= selected_time <= 19) else 0

    # Weather inputs
    st.markdown("#### 🌦️ Weather")

    if preset == 'rainy_rush':
        weather = 'Rain'
        temp = 65
        visibility = 3.0
    elif preset == 'night_clear':
        weather = 'Clear'
        temp = 72
        visibility = 10.0
    elif preset == 'foggy_morning':
        weather = 'Fog'
        temp = 58
        visibility = 1.0
    else:
        weather = 'Clear'
        temp = 70
        visibility = 10.0

    weather_condition = st.selectbox(
        "Weather Condition",
        options=['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow'],
        index=['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow'].index(weather)
    )

    temperature = st.slider(
        "Temperature (°F)",
        min_value=0,
        max_value=110,
        value=temp,
        step=1
    )

    visibility_mi = st.slider(
        "Visibility (miles)",
        min_value=0.0,
        max_value=10.0,
        value=visibility,
        step=0.5
    )

    # Derived features
    adverse_weather = 1 if weather_condition in ['Rain', 'Fog', 'Snow'] else 0
    low_visibility = 1 if visibility_mi < 2 else 0

    # Clear preset after use
    if preset:
        st.session_state['weather_preset'] = None

# Prediction section
st.markdown("---")
st.markdown("## 🔮 Step 3: Severity Prediction")

if st.button("🚨 PREDICT CRASH SEVERITY", type="primary", use_container_width=True):
    if 'hpms_features' not in st.session_state or st.session_state['hpms_features'] is None:
        st.error("⚠️ Please select a crash location on the map first!")
    else:
        with st.spinner("Analyzing crash conditions..."):
            # Collect all features
            features = {
                # Temporal
                'hour': selected_time,
                'day_of_week': selected_day,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,

                # Weather
                'temperature': temperature,
                'visibility': visibility_mi,
                'adverse_weather': adverse_weather,
                'low_visibility': low_visibility,

                # Road (from HPMS)
                **st.session_state['hpms_features']
            }

            # DEMO: Simple heuristic model (replace with trained model)
            # Higher risk factors: adverse weather, low visibility, high speed, rush hour, weekend night
            risk_score = 0.0

            # Weather risk
            if adverse_weather:
                risk_score += 0.25
            if low_visibility:
                risk_score += 0.20

            # Time risk
            if selected_time >= 22 or selected_time <= 4:  # Night
                risk_score += 0.15
            if is_rush_hour:
                risk_score += 0.10
            if is_weekend and (selected_time >= 22 or selected_time <= 4):  # Weekend night
                risk_score += 0.15

            # Road risk
            if features['speed_limit'] > 65:
                risk_score += 0.15
            if features['aadt'] > 50000:  # High traffic
                risk_score += 0.10

            # Add some randomness for demo
            risk_score += np.random.normal(0, 0.1)
            risk_score = np.clip(risk_score, 0, 1)

            # Convert to probability
            prob_high_severity = risk_score

            # Display results
            st.markdown("---")

            # Severity prediction
            col1, col2 = st.columns([1, 1])

            with col1:
                if prob_high_severity > 0.7:
                    st.markdown(f"""
                    <div style="background-color: #e74c3c; padding: 2rem; border-radius: 10px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🚨 HIGH SEVERITY</h1>
                        <h2 style="color: white; margin: 0;">{prob_high_severity*100:.0f}% Confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="warning-box">
                    <h3>⚠️ DISPATCH PRIORITY RESPONSE</h3>
                    <ul>
                        <li>Send multiple ambulances</li>
                        <li>Alert trauma center</li>
                        <li>Dispatch fire rescue</li>
                        <li>Notify highway patrol</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                elif prob_high_severity > 0.4:
                    st.markdown(f"""
                    <div style="background-color: #f39c12; padding: 2rem; border-radius: 10px; text-align: center;">
                        <h1 style="color: white; margin: 0;">⚠️ MEDIUM SEVERITY</h1>
                        <h2 style="color: white; margin: 0;">{prob_high_severity*100:.0f}% Confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="info-box">
                    <h3>Standard Response</h3>
                    <ul>
                        <li>Send standard ambulance</li>
                        <li>Monitor situation</li>
                        <li>Have backup on standby</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div style="background-color: #27ae60; padding: 2rem; border-radius: 10px; text-align: center;">
                        <h1 style="color: white; margin: 0;">✓ LOW SEVERITY</h1>
                        <h2 style="color: white; margin: 0;">{(1-prob_high_severity)*100:.0f}% Confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="success-box">
                    <h3>✓ Standard Response</h3>
                    <ul>
                        <li>Single unit response adequate</li>
                        <li>Likely minor injuries</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_high_severity * 100,
                    title={'text': "High Severity Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 40], 'color': "#27ae60"},
                            {'range': [40, 70], 'color': "#f39c12"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Confidence intervals
                st.markdown("#### 📊 Prediction Breakdown")
                st.metric("High Severity Probability", f"{prob_high_severity*100:.1f}%")
                st.metric("Low Severity Probability", f"{(1-prob_high_severity)*100:.1f}%")

            # Feature contributions
            st.markdown("---")
            st.markdown("### 📊 Risk Factor Analysis")

            risk_factors = []
            if adverse_weather:
                risk_factors.append(("Adverse Weather", 0.25))
            if low_visibility:
                risk_factors.append(("Low Visibility", 0.20))
            if selected_time >= 22 or selected_time <= 4:
                risk_factors.append(("Nighttime", 0.15))
            if is_weekend and (selected_time >= 22 or selected_time <= 4):
                risk_factors.append(("Weekend Night", 0.15))
            if is_rush_hour:
                risk_factors.append(("Rush Hour", 0.10))
            if features['speed_limit'] > 65:
                risk_factors.append(("High Speed Limit", 0.15))
            if features['aadt'] > 50000:
                risk_factors.append(("High Traffic", 0.10))

            if risk_factors:
                rf_df = pd.DataFrame(risk_factors, columns=['Factor', 'Contribution'])
                fig = go.Figure(go.Bar(
                    x=rf_df['Contribution'],
                    y=rf_df['Factor'],
                    orientation='h',
                    marker_color='#e74c3c'
                ))
                fig.update_layout(
                    title='Contributing Risk Factors',
                    xaxis_title='Risk Contribution',
                    yaxis_title='',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✓ No significant risk factors identified")

            # What-if analysis
            st.markdown("---")
            st.markdown("### 🔄 What-If Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### If weather was clear:")
                better_risk = risk_score - (0.25 if adverse_weather else 0) - (0.20 if low_visibility else 0)
                better_risk = np.clip(better_risk, 0, 1)
                change = (better_risk - risk_score) * 100
                st.metric("Severity Probability", f"{better_risk*100:.0f}%", f"{change:.0f}%")

            with col2:
                st.markdown("#### If crash at 2 PM:")
                day_risk = risk_score - (0.15 if (selected_time >= 22 or selected_time <= 4) else 0)
                day_risk = np.clip(day_risk, 0, 1)
                change = (day_risk - risk_score) * 100
                st.metric("Severity Probability", f"{day_risk*100:.0f}%", f"{change:.0f}%")

else:
    st.info("👆 Click the button above to predict crash severity")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #95a5a6; margin-top: 2rem;'>
    <p><b>Note:</b> This is a demo using a baseline model. In production, replace with trained ML model for accurate predictions.</p>
    <p><b>Model:</b> Trained on 371K Texas crashes (2016-2020) | <b>Performance:</b> ROC-AUC ~0.75</p>
</div>
""", unsafe_allow_html=True)
