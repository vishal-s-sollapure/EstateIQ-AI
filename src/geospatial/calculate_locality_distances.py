"""
EstateIQ AI - Phase 6.2: Locality Proximity & Geodesic Distance Engine
----------------------------------------------------------------------
Computes rigorous geodesic distances (in kilometres) from each Bengaluru
locality reference point to the nearest:
1. Operational Metro Station
2. Major Technology Park / IT Corridor
3. Hospital / Healthcare Facility

Methodology:
- Utilizes WGS84 ellipsoid geodesic distance via geopy.distance.geodesic.
- Does NOT apply arbitrary circuity factors; preserves true geodesic displacement.
- Explicitly flags that distances are approximate locality-reference-to-POI distances,
  NOT property-doorstep or road navigation distances.
- Leaves distances as NaN for unresolved localities and 'other'.
- Output: 'data/geospatial/locality_geospatial_features.csv'
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EstateIQ-Distances")


def find_nearest_poi(
    loc_lat: float,
    loc_lon: float,
    poi_df: pd.DataFrame
) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """
    Finds the nearest POI using exact geodesic distance calculation.
    Returns (distance_km, poi_name, osm_id).
    """
    if np.isnan(loc_lat) or np.isnan(loc_lon) or poi_df.empty:
        return None, None, None

    loc_point = (loc_lat, loc_lon)
    min_dist = float("inf")
    nearest_name = None
    nearest_id = None

    for _, row in poi_df.iterrows():
        poi_point = (row["latitude"], row["longitude"])
        # Calculate geodesic distance in km
        dist = geodesic(loc_point, poi_point).km
        if dist < min_dist:
            min_dist = dist
            nearest_name = row["name"]
            nearest_id = row.get("osm_id")

    if min_dist != float("inf"):
        return round(min_dist, 2), nearest_name, nearest_id
    return None, None, None


def compute_all_locality_distances(
    localities_csv: Path,
    metro_csv: Path,
    tech_csv: Path,
    hosp_csv: Path,
    output_csv: Path
) -> pd.DataFrame:
    """
    Orchestrates nearest-amenity distance computation across all localities.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not localities_csv.exists():
        raise FileNotFoundError(f"Localities file not found: {localities_csv}")
    if not metro_csv.exists():
        raise FileNotFoundError(f"Metro stations file not found: {metro_csv}")
    if not tech_csv.exists():
        raise FileNotFoundError(f"Tech parks file not found: {tech_csv}")
    if not hosp_csv.exists():
        raise FileNotFoundError(f"Hospitals file not found: {hosp_csv}")

    df_loc = pd.read_csv(localities_csv)
    df_metro = pd.read_csv(metro_csv)
    df_tech = pd.read_csv(tech_csv)
    df_hosp = pd.read_csv(hosp_csv)

    logger.info(f"Loaded {len(df_loc)} localities, {len(df_metro)} metro stations, {len(df_tech)} tech parks, {len(df_hosp)} hospitals.")

    results = []

    for _, row in df_loc.iterrows():
        locality = row["locality"]
        clean_loc = row.get("clean_locality", locality)
        freq = row.get("frequency", 0)
        status = row.get("geocoding_status", "UNKNOWN")
        lat = row.get("latitude")
        lon = row.get("longitude")

        record = {
            "locality": locality,
            "clean_locality": clean_loc,
            "frequency": freq,
            "geocoding_status": status,
            "latitude": lat,
            "longitude": lon,
            "nearest_metro_km": np.nan,
            "nearest_metro_name": None,
            "nearest_metro_osm_id": None,
            "nearest_tech_park_km": np.nan,
            "nearest_tech_park_name": None,
            "nearest_tech_park_osm_id": None,
            "nearest_hospital_km": np.nan,
            "nearest_hospital_name": None,
            "nearest_hospital_osm_id": None,
            "geodesic_method": "WGS84 Ellipsoid (geopy.distance.geodesic)",
            "distance_disclaimer": "Approximate locality-reference-to-POI geodesic displacement; not property doorstep or road navigation distance."
        }

        # If coordinates are valid and resolved
        if status == "SUCCESS" and pd.notnull(lat) and pd.notnull(lon):
            # 1. Nearest Metro Station
            d_metro, name_metro, id_metro = find_nearest_poi(lat, lon, df_metro)
            record["nearest_metro_km"] = d_metro
            record["nearest_metro_name"] = name_metro
            record["nearest_metro_osm_id"] = id_metro

            # 2. Nearest Technology Park
            d_tech, name_tech, id_tech = find_nearest_poi(lat, lon, df_tech)
            record["nearest_tech_park_km"] = d_tech
            record["nearest_tech_park_name"] = name_tech
            record["nearest_tech_park_osm_id"] = id_tech

            # 3. Nearest Hospital
            d_hosp, name_hosp, id_hosp = find_nearest_poi(lat, lon, df_hosp)
            record["nearest_hospital_km"] = d_hosp
            record["nearest_hospital_name"] = name_hosp
            record["nearest_hospital_osm_id"] = id_hosp

        results.append(record)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved proximity features for {len(df_out)} localities to: {output_csv}")

    # Summary table
    resolved = df_out[df_out["geocoding_status"] == "SUCCESS"]
    print("\n" + "=" * 75)
    print("📏 BENGALURU LOCALITY PROXIMITY SUMMARY (RESOLVED LOCALITIES)")
    print("=" * 75)
    print(f"Total Localities               : {len(df_out)}")
    print(f"Successfully Resolved Localities: {len(resolved)}")
    print("\nDistance Distributions (in km):")
    print(resolved[["nearest_metro_km", "nearest_tech_park_km", "nearest_hospital_km"]].describe().round(2).to_string())
    print("=" * 75 + "\n")

    return df_out


if __name__ == "__main__":
    data_dir = project_root / "data" / "geospatial"
    compute_all_locality_distances(
        localities_csv=data_dir / "locality_coordinates.csv",
        metro_csv=data_dir / "metro_stations.csv",
        tech_csv=data_dir / "tech_parks.csv",
        hosp_csv=data_dir / "hospitals.csv",
        output_csv=data_dir / "locality_geospatial_features.csv"
    )
