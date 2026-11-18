"""
EstateIQ AI - Phase 6.3: Geospatial Intelligence Service
-------------------------------------------------------
Provides high-performance, cached spatial queries and interactive PyDeck
map visualizations for Bengaluru residential localities and nearby amenities
(Metro stations, Technology parks, Hospitals).
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import pydeck as pdk

# Optional Streamlit cache decorator
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    def cache_resource(f):
        return f
    def cache_data(f):
        return f

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger("EstateIQ-GeoService")

DATA_DIR = project_root / "data" / "geospatial"
FEATURES_PATH = DATA_DIR / "locality_geospatial_features.csv"
METRO_PATH = DATA_DIR / "metro_stations.csv"
TECH_PARKS_PATH = DATA_DIR / "tech_parks.csv"
HOSPITALS_PATH = DATA_DIR / "hospitals.csv"


@cache_data
def load_locality_features() -> pd.DataFrame:
    """Loads and caches the 234 Bengaluru locality geospatial features."""
    if not FEATURES_PATH.exists():
        logger.warning(f"Locality features file not found at: {FEATURES_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_PATH)
    return df


@cache_data
def load_amenity_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and caches Metro stations, Tech parks, and Hospitals."""
    df_metro = pd.read_csv(METRO_PATH) if METRO_PATH.exists() else pd.DataFrame()
    df_tech = pd.read_csv(TECH_PARKS_PATH) if TECH_PARKS_PATH.exists() else pd.DataFrame()
    df_hosp = pd.read_csv(HOSPITALS_PATH) if HOSPITALS_PATH.exists() else pd.DataFrame()
    return df_metro, df_tech, df_hosp


def get_locality_proximity(locality_name: str) -> Dict[str, Any]:
    """
    Retrieves geospatial proximity metrics for a specific locality.

    Returns:
        Dict with status, coordinates, nearest amenities (metro, tech park, hospital),
        and spatial disclaimers.
    """
    df = load_locality_features()
    if df.empty:
        return {
            "locality": locality_name,
            "is_geocoded": False,
            "status": "DATA_UNAVAILABLE",
            "message": "Geospatial database not loaded."
        }

    match = df[df["locality"] == locality_name]
    if match.empty:
        # Try case-insensitive or stripped match
        match = df[df["locality"].str.strip().str.lower() == locality_name.strip().lower()]

    if match.empty:
        return {
            "locality": locality_name,
            "is_geocoded": False,
            "status": "UNKNOWN_LOCALITY",
            "message": f"Locality '{locality_name}' not found in spatial registry."
        }

    row = match.iloc[0]
    status = str(row["geocoding_status"])
    lat = row["latitude"]
    lon = row["longitude"]

    is_geocoded = status == "SUCCESS" and pd.notna(lat) and pd.notna(lon)

    if not is_geocoded:
        if status == "EXCLUDED_AGGREGATE":
            msg = "Synthetic aggregate category 'other' represents miscellaneous dispersed locations across Bengaluru. Centroid coordinates and proximity metrics are deliberately omitted to avoid spatial fabrication."
        elif status == "OUT_OF_BOUNDS":
            msg = f"Locality '{locality_name}' lies outside metropolitan Bengaluru's administrative bounding box."
        else:
            msg = f"Locality '{locality_name}' could not be resolved to a definitive geographic reference point in OpenStreetMap / Nominatim."

        return {
            "locality": locality_name,
            "is_geocoded": False,
            "status": status,
            "message": msg,
            "latitude": None,
            "longitude": None
        }

    return {
        "locality": locality_name,
        "is_geocoded": True,
        "status": status,
        "latitude": float(lat),
        "longitude": float(lon),
        "frequency": int(row["frequency"]) if pd.notna(row.get("frequency")) else 0,
        "metro": {
            "name": str(row["nearest_metro_name"]) if pd.notna(row.get("nearest_metro_name")) else "N/A",
            "distance_km": round(float(row["nearest_metro_km"]), 2) if pd.notna(row.get("nearest_metro_km")) else None,
            "osm_id": str(int(row["nearest_metro_osm_id"])) if pd.notna(row.get("nearest_metro_osm_id")) else "N/A"
        },
        "tech_park": {
            "name": str(row["nearest_tech_park_name"]) if pd.notna(row.get("nearest_tech_park_name")) else "N/A",
            "distance_km": round(float(row["nearest_tech_park_km"]), 2) if pd.notna(row.get("nearest_tech_park_km")) else None,
            "osm_id": str(int(row["nearest_tech_park_osm_id"])) if pd.notna(row.get("nearest_tech_park_osm_id")) else "N/A"
        },
        "hospital": {
            "name": str(row["nearest_hospital_name"]) if pd.notna(row.get("nearest_hospital_name")) else "N/A",
            "distance_km": round(float(row["nearest_hospital_km"]), 2) if pd.notna(row.get("nearest_hospital_km")) else None,
            "osm_id": str(int(row["nearest_hospital_osm_id"])) if pd.notna(row.get("nearest_hospital_osm_id")) else "N/A"
        },
        "geodesic_method": str(row.get("geodesic_method", "WGS84 Ellipsoid (geopy.distance.geodesic)")),
        "disclaimer": str(row.get("distance_disclaimer", "Approximate locality-reference-to-POI geodesic displacement; not property doorstep or road navigation distance."))
    }


def get_surrounding_pois(
    center_lat: float,
    center_lon: float,
    radius_km: float = 5.0,
    max_per_category: int = 15
) -> Dict[str, pd.DataFrame]:
    """
    Filters surrounding Metro stations, Tech parks, and Hospitals within a radius.
    """
    df_metro, df_tech, df_hosp = load_amenity_datasets()
    result = {}

    # Fast approximate bounding box (1 deg latitude ~ 111 km, 1 deg lon ~ 108 km in Bengaluru)
    lat_deg_delta = radius_km / 111.0
    lon_deg_delta = radius_km / 108.0

    for name, df in [("metro", df_metro), ("tech_park", df_tech), ("hospital", df_hosp)]:
        if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            result[name] = pd.DataFrame()
            continue

        # Bounding box filter
        mask = (
            (df["latitude"] >= center_lat - lat_deg_delta) &
            (df["latitude"] <= center_lat + lat_deg_delta) &
            (df["longitude"] >= center_lon - lon_deg_delta) &
            (df["longitude"] <= center_lon + lon_deg_delta)
        )
        sub_df = df[mask].copy()

        if not sub_df.empty:
            # Approximate distance calculation in km
            d_lat = (sub_df["latitude"] - center_lat) * 111.0
            d_lon = (sub_df["longitude"] - center_lon) * 108.0
            sub_df["approx_dist_km"] = np.sqrt(d_lat**2 + d_lon**2)
            sub_df = sub_df[sub_df["approx_dist_km"] <= radius_km]
            sub_df = sub_df.sort_values("approx_dist_km").head(max_per_category)

        result[name] = sub_df

    return result


def build_pydeck_map(
    locality_name: str,
    latitude: Optional[float],
    longitude: Optional[float],
    radius_km: float = 4.0
) -> Optional[pdk.Deck]:
    """
    Creates an interactive PyDeck map centered on the locality centroid,
    highlighting the property pin and surrounding amenities.
    """
    if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
        return None

    # Surrounding POIs
    pois = get_surrounding_pois(latitude, longitude, radius_km=radius_km)

    layers = []

    # 1. Locality Centroid Pin (Red / Coral, Large radius, pulsing effect)
    property_df = pd.DataFrame([{
        "name": f"Property Locality: {locality_name}",
        "category": "Property Centroid",
        "latitude": latitude,
        "longitude": longitude,
        "color": [239, 68, 68, 220],  # Tailwind Red-500
        "radius": 220
    }])

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=property_df,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )
    )

    # 2. Metro Stations Layer (Emerald Green)
    if not pois["metro"].empty:
        metro_df = pois["metro"].copy()
        metro_df["color"] = [[16, 185, 129, 200] for _ in range(len(metro_df))]  # Green-500
        metro_df["radius"] = 140
        metro_df["category"] = "Metro Station"
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=metro_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    # 3. Technology Parks Layer (Royal Blue / Indigo)
    if not pois["tech_park"].empty:
        tech_df = pois["tech_park"].copy()
        tech_df["color"] = [[99, 102, 241, 200] for _ in range(len(tech_df))]  # Indigo-500
        tech_df["radius"] = 150
        tech_df["category"] = "Tech Park"
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=tech_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    # 4. Hospitals Layer (Amber / Orange)
    if not pois["hospital"].empty:
        hosp_df = pois["hospital"].copy()
        hosp_df["color"] = [[245, 158, 11, 180] for _ in range(len(hosp_df))]  # Amber-500
        hosp_df["radius"] = 120
        hosp_df["category"] = "Hospital"
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=hosp_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    # View State centered on property
    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=13.0,
        pitch=35,
        bearing=0
    )

    tooltip = {
        "html": "<b>{name}</b><br/>Category: {category}<br/>Coordinates: [{latitude}, {longitude}]",
        "style": {
            "backgroundColor": "#1e293b",
            "color": "#f8fafc",
            "fontSize": "12px",
            "borderRadius": "6px",
            "padding": "6px 10px"
        }
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip=tooltip
    )

    return deck
