"""
EstateIQ AI - Phase 6.2: Geospatial Unit & Integration Tests
------------------------------------------------------------
Validates geospatial calculations, bounding box constraints,
offline nearest-POI searches, and schema integrity without requiring live API calls.
"""

import sys
import unittest
from pathlib import Path

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from geopy.distance import geodesic

from src.geospatial.geocode_localities import (
    normalize_locality_name,
    is_within_bengaluru_bounds,
    BENGALURU_BBOX
)
from src.geospatial.calculate_locality_distances import find_nearest_poi


class TestGeospatialLogic(unittest.TestCase):

    def test_normalize_locality_name(self):
        """Tests whitespace trimming and multi-space collapse."""
        self.assertEqual(normalize_locality_name("  Sarjapur   Road  "), "Sarjapur Road")
        self.assertEqual(normalize_locality_name("Electronic City Phase 1"), "Electronic City Phase 1")
        self.assertEqual(normalize_locality_name("other"), "other")

    def test_within_bengaluru_bounds(self):
        """Verifies bounding box filter accepts Bengaluru and rejects distant cities."""
        # Genuine Bengaluru coordinates
        self.assertTrue(is_within_bengaluru_bounds(12.9716, 77.5946)) # MG Road
        self.assertTrue(is_within_bengaluru_bounds(12.8452, 77.6602)) # Electronic City
        self.assertTrue(is_within_bengaluru_bounds(13.0475, 77.6200)) # Manyata

        # Out-of-bounds coordinates
        self.assertFalse(is_within_bengaluru_bounds(19.0760, 72.8777)) # Mumbai
        self.assertFalse(is_within_bengaluru_bounds(28.6139, 77.2090)) # New Delhi
        self.assertFalse(is_within_bengaluru_bounds(0.0, 0.0))         # Null Island

    def test_find_nearest_poi_offline(self):
        """Controlled unit test verifying minimum geodesic distance selection without network calls."""
        # Reference point: MG Road, Bengaluru (12.9756, 77.6066)
        loc_lat = 12.9756
        loc_lon = 77.6066

        # Controlled mock POIs
        mock_pois = pd.DataFrame([
            {"osm_id": 101, "name": "Trinity Metro Station", "latitude": 12.9729, "longitude": 77.6171},
            {"osm_id": 102, "name": "Indiranagar Metro Station", "latitude": 12.9783, "longitude": 77.6408},
            {"osm_id": 103, "name": "Baiyappanahalli Metro Station", "latitude": 12.9912, "longitude": 77.6521}
        ])

        dist_km, nearest_name, nearest_id = find_nearest_poi(loc_lat, loc_lon, mock_pois)

        # Expected nearest is Trinity (~1.18 km away)
        self.assertEqual(nearest_name, "Trinity Metro Station")
        self.assertEqual(nearest_id, 101)
        self.assertGreater(dist_km, 0.0)
        self.assertLess(dist_km, 2.0)

        # Manual direct geodesic check
        direct_dist = round(geodesic((loc_lat, loc_lon), (12.9729, 77.6171)).km, 2)
        self.assertEqual(dist_km, direct_dist)

    def test_nan_coordinates_handling(self):
        """Verifies that missing or NaN coordinates safely return None without throwing exceptions."""
        mock_pois = pd.DataFrame([
            {"osm_id": 101, "name": "Test POI", "latitude": 12.9729, "longitude": 77.6171}
        ])

        dist, name, osm_id = find_nearest_poi(np.nan, 77.6066, mock_pois)
        self.assertIsNone(dist)
        self.assertIsNone(name)
        self.assertIsNone(osm_id)

        dist, name, osm_id = find_nearest_poi(12.9756, np.nan, mock_pois)
        self.assertIsNone(dist)
        self.assertIsNone(name)
        self.assertIsNone(osm_id)

    def test_empty_poi_dataframe(self):
        """Verifies behavior when POI dataframe is empty."""
        empty_df = pd.DataFrame(columns=["osm_id", "name", "latitude", "longitude"])
        dist, name, osm_id = find_nearest_poi(12.9756, 77.6066, empty_df)
        self.assertIsNone(dist)
        self.assertIsNone(name)
        self.assertIsNone(osm_id)


class TestGeneratedDataIntegrity(unittest.TestCase):

    def test_dataset_if_exists(self):
        """Checks data integrity of generated CSV files if already executed."""
        features_csv = project_root / "data" / "geospatial" / "locality_geospatial_features.csv"
        if not features_csv.exists():
            self.skipTest("locality_geospatial_features.csv not yet generated.")

        df = pd.read_csv(features_csv)
        self.assertEqual(len(df), 234, "Must contain all 234 unique localities.")

        # Check 'other' is unassigned
        other_row = df[df["locality"] == "other"].iloc[0]
        self.assertEqual(other_row["geocoding_status"], "EXCLUDED_AGGREGATE")
        self.assertTrue(np.isnan(other_row["latitude"]))
        self.assertTrue(np.isnan(other_row["nearest_metro_km"]))

        # Check resolved localities
        resolved = df[df["geocoding_status"] == "SUCCESS"]
        self.assertGreater(len(resolved), 100, "Should resolve at least 100+ standard localities.")

        for _, r in resolved.iterrows():
            # Check bounding box
            self.assertTrue(is_within_bengaluru_bounds(r["latitude"], r["longitude"]))
            # Check distances are non-negative and finite
            self.assertGreaterEqual(r["nearest_metro_km"], 0.0)
            self.assertGreaterEqual(r["nearest_tech_park_km"], 0.0)
            self.assertGreaterEqual(r["nearest_hospital_km"], 0.0)


if __name__ == "__main__":
    unittest.main()
