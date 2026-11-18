"""
EstateIQ AI - Phase 6.3: Geospatial Intelligence Explorer UI
------------------------------------------------------------
Renders OpenStreetMap-enriched locality proximity metrics and 2D/3D interactive
PyDeck cartographic visualizations for metropolitan Bengaluru.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.services.geo_service import (
    get_locality_proximity,
    build_pydeck_map
)


def render_geospatial_explorer(location: str):
    """
    Renders proximity metrics and an interactive PyDeck map for the selected locality.
    """
    st.markdown("### 🗺️ Locality Geospatial Intelligence & Amenity Proximity")

    proximity = get_locality_proximity(location)

    if not proximity.get("is_geocoded", False):
        st.warning(
            f"📍 **Geospatial Context for '{location}':**\n\n"
            f"{proximity.get('message', 'Geographic coordinates unavailable for this locality.')}"
        )
        st.caption(
            "Note: The valuation model continues to operate with high precision using historical categorical "
            "embeddings even when spatial coordinates are omitted."
        )
        return

    # Proximity Metric Cards
    metro = proximity.get("metro", {})
    tech = proximity.get("tech_park", {})
    hosp = proximity.get("hospital", {})

    m_cols = st.columns(3)

    with m_cols[0]:
        st.markdown(
            f"""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #166534;">🚇 Nearest Metro Station</span>
                <div style="font-size: 1.6rem; font-weight: 700; color: #15803d; margin: 6px 0;">
                    {metro.get('distance_km', 'N/A')} km
                </div>
                <div style="font-size: 0.85rem; color: #166534;">
                    <strong>{metro.get('name', 'N/A')}</strong><br/>
                    <span style="color: #4b5563;">OSM ID: {metro.get('osm_id', 'N/A')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_cols[1]:
        st.markdown(
            f"""
            <div style="background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 12px 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #3730a3;">🏢 Nearest Tech Park / IT Hub</span>
                <div style="font-size: 1.6rem; font-weight: 700; color: #4338ca; margin: 6px 0;">
                    {tech.get('distance_km', 'N/A')} km
                </div>
                <div style="font-size: 0.85rem; color: #3730a3;">
                    <strong>{tech.get('name', 'N/A')}</strong><br/>
                    <span style="color: #4b5563;">OSM ID: {tech.get('osm_id', 'N/A')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_cols[2]:
        st.markdown(
            f"""
            <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px 16px;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #92400e;">🏥 Nearest Healthcare / Hospital</span>
                <div style="font-size: 1.6rem; font-weight: 700; color: #b45309; margin: 6px 0;">
                    {hosp.get('distance_km', 'N/A')} km
                </div>
                <div style="font-size: 0.85rem; color: #92400e;">
                    <strong>{hosp.get('name', 'N/A')}</strong><br/>
                    <span style="color: #4b5563;">OSM ID: {hosp.get('osm_id', 'N/A')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Interactive Map Section
    lat = proximity.get("latitude")
    lon = proximity.get("longitude")

    deck = build_pydeck_map(
        locality_name=location,
        latitude=lat,
        longitude=lon,
        radius_km=4.5
    )

    if deck:
        st.markdown("##### 📍 Interactive Micro-Market Map")
        st.pydeck_chart(deck, use_container_width=True)

        # Map Legend & Attribution
        legend_cols = st.columns([1, 1, 1, 1])
        with legend_cols[0]:
            st.markdown("🔴 **Red**: Locality Centroid")
        with legend_cols[1]:
            st.markdown("🟢 **Green**: Namma Metro")
        with legend_cols[2]:
            st.markdown("🔵 **Indigo**: Tech Park")
        with legend_cols[3]:
            st.markdown("🟡 **Amber**: Hospital")

    st.caption(
        "📐 **Spatial Methodology & Precision Notice**: Distances are geodesic displacements calculated on the "
        "WGS84 ellipsoid via OpenStreetMap Overpass queries. Measurements represent proximity from the locality reference "
        "centroid and do not reflect street-level driving itineraries or exact apartment doorstep locations. "
        "Map data © OpenStreetMap contributors (ODbL 1.0)."
    )
