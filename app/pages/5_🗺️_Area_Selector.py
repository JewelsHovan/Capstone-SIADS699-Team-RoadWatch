"""
Texas Area Selector - Interactive Map with Drawing Tools
"""

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from config import PAGE_CONFIG, CUSTOM_CSS, TEXAS_CENTER

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ Texas Area Selector</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1rem; color: #7f8c8d; margin-bottom: 2rem;'>
    Draw boxes or polygons on the map to select areas of interest in Texas
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Instructions
st.markdown("""
### 📝 Instructions
1. Use the **drawing tools** on the left side of the map
2. Click the **rectangle** or **polygon** tool to draw areas
3. Draw your area of interest on the map
4. The selected area coordinates will appear below
5. Use the **delete** tool to remove drawn shapes
""")

st.markdown("---")

# Create the map
st.markdown("### 🗺️ Interactive Map")

# Initialize the map centered on Texas
m = folium.Map(
    location=TEXAS_CENTER,
    zoom_start=6,
    tiles='OpenStreetMap'
)

# Add drawing tools
draw = Draw(
    export=True,
    draw_options={
        'polyline': False,
        'polygon': {
            'allowIntersection': False,
            'drawError': {
                'color': '#e74c3c',
                'message': 'Error: shapes cannot intersect!'
            },
            'shapeOptions': {
                'color': '#3498db',
                'fillOpacity': 0.3
            }
        },
        'circle': False,
        'rectangle': {
            'shapeOptions': {
                'color': '#e74c3c',
                'fillOpacity': 0.3
            }
        },
        'marker': False,
        'circlemarker': False,
    },
    edit_options={
        'edit': True,
        'remove': True
    }
)

draw.add_to(m)

# Display the map and capture drawn objects
st.markdown("#### Draw your area below:")
map_data = st_folium(
    m,
    width=800,
    height=600,
    returned_objects=["all_drawings"]
)

st.markdown("---")

# Display selected area information
st.markdown("### 📊 Selected Area Information")

if map_data and map_data.get("all_drawings"):
    drawings = map_data["all_drawings"]

    if drawings:
        st.success(f"✅ {len(drawings)} area(s) selected")

        # Display each drawn shape
        for idx, drawing in enumerate(drawings):
            with st.expander(f"Area {idx + 1} Details", expanded=True):
                geometry = drawing.get("geometry", {})
                geometry_type = geometry.get("type", "Unknown")
                coordinates = geometry.get("coordinates", [])

                st.markdown(f"**Type**: {geometry_type}")

                if geometry_type == "Polygon":
                    # Extract bounding box
                    if coordinates and len(coordinates) > 0:
                        points = coordinates[0]
                        lats = [point[1] for point in points]
                        lons = [point[0] for point in points]

                        bbox = {
                            "north": max(lats),
                            "south": min(lats),
                            "east": max(lons),
                            "west": min(lons)
                        }

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("North Latitude", f"{bbox['north']:.4f}°")
                            st.metric("South Latitude", f"{bbox['south']:.4f}°")
                        with col2:
                            st.metric("East Longitude", f"{bbox['east']:.4f}°")
                            st.metric("West Longitude", f"{bbox['west']:.4f}°")

                        # Calculate approximate area
                        lat_diff = abs(bbox['north'] - bbox['south'])
                        lon_diff = abs(bbox['east'] - bbox['west'])

                        # Rough estimation (1 degree ≈ 69 miles at Texas latitude)
                        area_sq_miles = (lat_diff * 69) * (lon_diff * 54)  # 54 miles/degree longitude at ~30°N

                        st.metric("Approximate Area", f"{area_sq_miles:,.0f} sq miles")

                        # Show raw coordinates
                        st.markdown("**Raw Coordinates:**")
                        st.json({"bounding_box": bbox, "coordinates": coordinates})

                elif geometry_type == "Rectangle":
                    if coordinates and len(coordinates) > 0:
                        points = coordinates[0]
                        lats = [point[1] for point in points]
                        lons = [point[0] for point in points]

                        bbox = {
                            "north": max(lats),
                            "south": min(lats),
                            "east": max(lons),
                            "west": min(lons)
                        }

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("North Latitude", f"{bbox['north']:.4f}°")
                            st.metric("South Latitude", f"{bbox['south']:.4f}°")
                        with col2:
                            st.metric("East Longitude", f"{bbox['east']:.4f}°")
                            st.metric("West Longitude", f"{bbox['west']:.4f}°")

                        # Calculate area
                        lat_diff = abs(bbox['north'] - bbox['south'])
                        lon_diff = abs(bbox['east'] - bbox['west'])
                        area_sq_miles = (lat_diff * 69) * (lon_diff * 54)

                        st.metric("Approximate Area", f"{area_sq_miles:,.0f} sq miles")

                        st.markdown("**Bounding Box:**")
                        st.json({"bounding_box": bbox})

        # Option to download coordinates
        st.markdown("---")
        st.download_button(
            label="📥 Download Area Data (JSON)",
            data=str(drawings),
            file_name="selected_areas.json",
            mime="application/json"
        )
    else:
        st.info("👆 Draw an area on the map above to see details")
else:
    st.info("👆 Draw an area on the map above to see details")

# Sidebar
with st.sidebar:
    st.markdown("## 🗺️ Area Selector")
    st.markdown("---")

    st.markdown("### 🎨 Drawing Tools")
    st.markdown("- **Rectangle**: Draw rectangular areas")
    st.markdown("- **Polygon**: Draw custom shapes")
    st.markdown("- **Edit**: Modify drawn shapes")
    st.markdown("- **Delete**: Remove shapes")

    st.markdown("---")

    st.markdown("### 💡 Use Cases")
    st.markdown("""
    - Select work zone locations
    - Define geographic regions
    - Identify high-risk areas
    - Filter data by area
    """)

    st.markdown("---")

    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool allows you to interactively select geographic areas in Texas
    for further analysis or filtering.
    """)
