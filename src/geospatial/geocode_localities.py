"""
EstateIQ AI - Phase 6.2: Bengaluru Locality Geocoding Pipeline
--------------------------------------------------------------
Geocodes the 234 unique localities from 'data/processed/bengaluru_cleaned.csv'
using OpenStreetMap's Nominatim geocoder (via GeoPy):
1. Ingests cleaned dataset and extracts unique localities and frequencies.
2. Identifies the application with a descriptive academic User-Agent.
3. Adheres strictly to Nominatim usage policies (>= 1.0s delay between requests).
4. Persists an incremental on-disk cache so lookups are never repeated unnecessarily.
5. Employs geographic bounding constraints for the Bengaluru metropolitan region
   (Latitude: [12.70, 13.35], Longitude: [77.30, 77.95]).
6. Treats 'other' as an aggregated cluster (zero synthetic coordinates).
7. Unresolved localities are recorded with NaN coordinates and status 'UNRESOLVED'.
8. Outputs to 'data/geospatial/locality_coordinates.csv'.
"""

import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

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
logger = logging.getLogger("EstateIQ-Geocode")

# Bengaluru Metropolitan Plausible Bounding Box
BENGALURU_BBOX = {
    "min_lat": 12.70,
    "max_lat": 13.35,
    "min_lon": 77.30,
    "max_lon": 77.95
}

USER_AGENT = "EstateIQ-AI/1.0 (academic real estate research; contact: estateiq-research@bengaluru.edu)"


def normalize_locality_name(locality: str) -> str:
    """Normalizes whitespace and formatting while preserving locality identities."""
    cleaned = " ".join(locality.strip().split())
    return cleaned


def is_within_bengaluru_bounds(lat: float, lon: float) -> bool:
    """Validates if a coordinate falls within the metropolitan Bengaluru extent."""
    return (
        BENGALURU_BBOX["min_lat"] <= lat <= BENGALURU_BBOX["max_lat"] and
        BENGALURU_BBOX["min_lon"] <= lon <= BENGALURU_BBOX["max_lon"]
    )


def geocode_all_localities(
    input_csv: Path,
    output_csv: Path,
    cache_json: Path,
    delay_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Executes rate-limited geocoding with incremental disk caching.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_json.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Source dataset not found at: {input_csv}")

    df = pd.read_csv(input_csv)
    locality_counts = df["location"].value_counts()
    unique_localities = locality_counts.index.tolist()
    total_unique = len(unique_localities)

    logger.info(f"Loaded processed dataset: {len(df):,} listings, {total_unique} unique localities")

    # Load existing cache if present
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_json.exists():
        try:
            with open(cache_json, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded existing geocoding cache with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load cache ({e}), starting fresh.")

    geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
    rate_limited_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay_seconds)

    results = []

    for idx, raw_locality in enumerate(unique_localities, 1):
        clean_name = normalize_locality_name(raw_locality)
        frequency = int(locality_counts[raw_locality])

        # Handle 'other' category without making external API calls
        if clean_name.lower() == "other":
            results.append({
                "locality": raw_locality,
                "clean_locality": clean_name,
                "frequency": frequency,
                "latitude": np.nan,
                "longitude": np.nan,
                "geocoding_status": "EXCLUDED_AGGREGATE",
                "display_name": "Aggregated tail category (<10 listings in raw data)",
                "osm_type": None,
                "osm_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "precision_note": "No geographic coordinates assigned to aggregated cluster."
            })
            continue

        # Check cache first
        if raw_locality in cache:
            cached_entry = cache[raw_locality]
            cached_entry["frequency"] = frequency
            results.append(cached_entry)
            continue

        # Execute live query
        query_str = f"{clean_name}, Bengaluru, Karnataka, India"
        logger.info(f"[{idx}/{total_unique}] Geocoding: '{query_str}'...")

        record: Dict[str, Any] = {
            "locality": raw_locality,
            "clean_locality": clean_name,
            "frequency": frequency,
            "latitude": np.nan,
            "longitude": np.nan,
            "geocoding_status": "UNRESOLVED",
            "display_name": None,
            "osm_type": None,
            "osm_id": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "precision_note": "Approximate locality reference point via OpenStreetMap Nominatim; not a doorstep centroid."
        }

        try:
            location_res = rate_limited_geocode(
                query_str,
                exactly_one=True,
                addressdetails=True
            )

            # If specific query failed, try secondary fallback query without 'Karnataka'
            if location_res is None:
                fallback_query = f"{clean_name}, Bangalore, India"
                location_res = rate_limited_geocode(fallback_query, exactly_one=True, addressdetails=True)

            if location_res is not None:
                lat, lon = float(location_res.latitude), float(location_res.longitude)
                if is_within_bengaluru_bounds(lat, lon):
                    raw_data = location_res.raw
                    record["latitude"] = lat
                    record["longitude"] = lon
                    record["geocoding_status"] = "SUCCESS"
                    record["display_name"] = str(location_res.address)
                    record["osm_type"] = raw_data.get("osm_type")
                    record["osm_id"] = raw_data.get("osm_id")
                    logger.info(f" -> SUCCESS: ({lat:.5f}, {lon:.5f}) - {record['display_name'][:60]}...")
                else:
                    record["geocoding_status"] = "OUT_OF_BOUNDS"
                    record["display_name"] = str(location_res.address)
                    logger.warning(f" -> OUT OF BOUNDS: ({lat:.5f}, {lon:.5f}) outside Bengaluru bbox.")
            else:
                logger.warning(f" -> UNRESOLVED: No result returned for '{clean_name}'.")

        except (GeocoderTimedOut, GeocoderServiceError) as err:
            logger.error(f" -> API ERROR for '{clean_name}': {err}")
            record["geocoding_status"] = "API_ERROR"

        # Update cache & results
        cache[raw_locality] = record
        results.append(record)

        # Save cache every 10 queries
        if idx % 10 == 0:
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

    # Final cache save
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved locality coordinates to: {output_csv}")

    # Summary Statistics
    status_counts = df_out["geocoding_status"].value_counts()
    print("\n" + "=" * 70)
    print("📍 BENGALURU LOCALITY GEOCODING SUMMARY")
    print("=" * 70)
    print(f"Total Localities Evaluated: {len(df_out)}")
    for status, count in status_counts.items():
        pct = (count / len(df_out)) * 100
        print(f" - {status:<20}: {count:>4} ({pct:>5.1f}%)")
    print("=" * 70 + "\n")

    return df_out


if __name__ == "__main__":
    data_dir = project_root / "data"
    input_path = data_dir / "processed" / "bengaluru_cleaned.csv"
    output_path = data_dir / "geospatial" / "locality_coordinates.csv"
    cache_path = data_dir / "geospatial" / "cache" / "nominatim_cache.json"

    geocode_all_localities(input_path, output_path, cache_path, delay_seconds=1.0)
