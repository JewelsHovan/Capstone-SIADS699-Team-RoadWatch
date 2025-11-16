"""
Map visualization utilities for Texas Crash Analysis Dashboard
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
from typing import Optional, Tuple


def create_crash_map(df: pd.DataFrame, lat_col: str = 'Start_Lat', lon_col: str = 'Start_Lng',
                      center: Optional[Tuple[float, float]] = None,
                      zoom_start: int = 6, max_points: int = 1000) -> folium.Map:
    """
    Create a map with crash locations

    Args:
        df: DataFrame with crash data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        center: Map center (lat, lon). If None, use mean of data
        zoom_start: Initial zoom level
        max_points: Maximum number of points to plot

    Returns:
        Folium map object
    """
    # Sample if too many points
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df.copy()

    # Remove missing coordinates
    df_sample = df_sample.dropna(subset=[lat_col, lon_col])

    # Determine map center
    if center is None:
        center = (df_sample[lat_col].mean(), df_sample[lon_col].mean())

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )

    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers
    for idx, row in df_sample.iterrows():
        # Create popup text
        popup_text = f"<b>Crash ID:</b> {row.get('ID', 'N/A')}<br>"
        if 'Severity' in row:
            popup_text += f"<b>Severity:</b> {row['Severity']}<br>"
        if 'Start_Time' in row:
            popup_text += f"<b>Date:</b> {row['Start_Time']}<br>"

        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)

    return m


def create_heatmap(df: pd.DataFrame, lat_col: str = 'Start_Lat', lon_col: str = 'Start_Lng',
                   center: Optional[Tuple[float, float]] = None,
                   zoom_start: int = 6, max_points: int = 10000) -> folium.Map:
    """
    Create a heatmap of crash density

    Args:
        df: DataFrame with crash data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        center: Map center (lat, lon). If None, use mean of data
        zoom_start: Initial zoom level
        max_points: Maximum number of points to include

    Returns:
        Folium map object
    """
    # Sample if too many points
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df.copy()

    # Remove missing coordinates
    df_sample = df_sample.dropna(subset=[lat_col, lon_col])

    # Determine map center
    if center is None:
        center = (df_sample[lat_col].mean(), df_sample[lon_col].mean())

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )

    # Prepare heatmap data
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in df_sample.iterrows()]

    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=15,
        blur=25,
        max_zoom=13,
        gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1.0: 'red'}
    ).add_to(m)

    return m


def create_workzone_map(df: pd.DataFrame, lat_col: str = 'beginning_latitude',
                        lon_col: str = 'beginning_longitude',
                        center: Optional[Tuple[float, float]] = None,
                        zoom_start: int = 6) -> folium.Map:
    """
    Create a map with work zone locations

    Args:
        df: DataFrame with work zone data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        center: Map center (lat, lon). If None, use Texas center
        zoom_start: Initial zoom level

    Returns:
        Folium map object
    """
    # Remove missing coordinates
    df_clean = df.dropna(subset=[lat_col, lon_col])

    # Determine map center
    if center is None:
        if len(df_clean) > 0:
            center = (df_clean[lat_col].mean(), df_clean[lon_col].mean())
        else:
            center = (31.0, -100.0)  # Texas center

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )

    # Add work zone markers
    for idx, row in df_clean.iterrows():
        # Create popup text
        popup_text = f"<b>Road:</b> {row.get('road_name', 'N/A')}<br>"
        if 'county_name' in row:
            popup_text += f"<b>County:</b> {row['county_name']}<br>"
        if 'start_date' in row:
            popup_text += f"<b>Start:</b> {row['start_date']}<br>"
        if 'end_date' in row:
            popup_text += f"<b>End:</b> {row['end_date']}<br>"

        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='orange', icon='wrench')
        ).add_to(m)

    return m


def create_segment_map(df: pd.DataFrame, lat_col: str = 'Start_Lat', lon_col: str = 'Start_Lng',
                       risk_col: str = 'risk_category',
                       center: Optional[Tuple[float, float]] = None,
                       zoom_start: int = 6, max_points: int = 500) -> folium.Map:
    """
    Create a map showing road segments colored by risk level

    Args:
        df: DataFrame with segment data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        risk_col: Name of risk category column
        center: Map center (lat, lon). If None, use mean of data
        zoom_start: Initial zoom level
        max_points: Maximum number of segments to plot

    Returns:
        Folium map object
    """
    # Sample if too many points
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df.copy()

    # Remove missing coordinates
    df_sample = df_sample.dropna(subset=[lat_col, lon_col])

    # Determine map center
    if center is None:
        center = (df_sample[lat_col].mean(), df_sample[lon_col].mean())

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )

    # Define color mapping for risk levels
    risk_colors = {
        'LOW': 'green',
        'MEDIUM': 'orange',
        'HIGH': 'red',
        'VERY_HIGH': 'darkred'
    }

    # Add segment markers
    for idx, row in df_sample.iterrows():
        risk = row.get(risk_col, 'MEDIUM')
        color = risk_colors.get(risk, 'gray')

        # Create popup text
        popup_text = f"<b>Segment:</b> {row.get('segment_id', 'N/A')}<br>"
        if 'crash_count' in row:
            popup_text += f"<b>Crashes:</b> {row['crash_count']}<br>"
        if 'severity_rate' in row:
            popup_text += f"<b>Severity Rate:</b> {row['severity_rate']*100:.1f}%<br>"
        if risk_col in row:
            popup_text += f"<b>Risk:</b> {row[risk_col]}<br>"

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    return m
