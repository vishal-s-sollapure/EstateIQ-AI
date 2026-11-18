"""
EstateIQ AI - Phase 6.3: Locality Proximity Catalog UI
------------------------------------------------------
Searchable, interactive directory of all 234 Bengaluru localities enriched with
geodesic proximities to Namma Metro stations, Tech Parks, and Hospitals.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from src.services.geo_service import load_locality_features


def render_locality_catalog():
    """
    Renders an interactive, filterable catalog of all 234 Bengaluru localities.
    """
    st.markdown("## 🗺️ Bengaluru Locality Proximity Catalog")
    st.markdown(
        "Explore and filter all 234 recognized residential localities across metropolitan Bengaluru. "
        "Distances are calculated as WGS84 geodesic displacements to OpenStreetMap-verified transit and infrastructure POIs."
    )

    df_geo = load_locality_features()
    if df_geo.empty:
        st.error("Locality geospatial database could not be loaded.")
        return

    # Filter Controls
    f_cols = st.columns([2, 1, 1, 1])

    with f_cols[0]:
        search_query = st.text_input("🔍 Search Locality Name", placeholder="e.g. Whitefield, Indiranagar, HSR...")

    with f_cols[1]:
        status_filter = st.selectbox(
            "Geocoding Status",
            options=["All", "SUCCESS", "UNRESOLVED", "EXCLUDED_AGGREGATE", "OUT_OF_BOUNDS"]
        )

    with f_cols[2]:
        max_metro = st.slider("Max Metro Dist (km)", min_value=0.5, max_value=25.0, value=15.0, step=0.5)

    with f_cols[3]:
        max_tech = st.slider("Max Tech Park Dist (km)", min_value=0.5, max_value=30.0, value=20.0, step=0.5)

    # Apply Filters
    df_filtered = df_geo.copy()

    if search_query:
        df_filtered = df_filtered[df_filtered["locality"].str.contains(search_query.strip(), case=False, na=False)]

    if status_filter != "All":
        df_filtered = df_filtered[df_filtered["geocoding_status"] == status_filter]

    # Filter numerical distance columns where valid
    if "nearest_metro_km" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["nearest_metro_km"].isna()) |
            (df_filtered["nearest_metro_km"] <= max_metro)
        ]

    if "nearest_tech_park_km" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["nearest_tech_park_km"].isna()) |
            (df_filtered["nearest_tech_park_km"] <= max_tech)
        ]

    st.markdown(f"**Showing {len(df_filtered)} of {len(df_geo)} Localities**")

    # Format Display Table
    display_cols = [
        "locality",
        "geocoding_status",
        "frequency",
        "nearest_metro_name",
        "nearest_metro_km",
        "nearest_tech_park_name",
        "nearest_tech_park_km",
        "nearest_hospital_name",
        "nearest_hospital_km"
    ]
    existing_cols = [c for c in display_cols if c in df_filtered.columns]

    df_display = df_filtered[existing_cols].copy()
    rename_dict = {
        "locality": "Locality",
        "geocoding_status": "Status",
        "frequency": "Listings",
        "nearest_metro_name": "Nearest Metro",
        "nearest_metro_km": "Metro Dist (km)",
        "nearest_tech_park_name": "Nearest Tech Park",
        "nearest_tech_park_km": "Tech Park Dist (km)",
        "nearest_hospital_name": "Nearest Hospital",
        "nearest_hospital_km": "Hosp Dist (km)"
    }
    df_display = df_display.rename(columns=rename_dict)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Data Source: OpenStreetMap Overpass API extraction & GeoPy Nominatim geocoding. "
        "Distances computed via geopy.distance.geodesic on WGS84 ellipsoid."
    )
