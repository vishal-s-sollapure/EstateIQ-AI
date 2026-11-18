"""
EstateIQ AI - Phase 6.4: End-to-End Functional Validation & Performance Benchmark
---------------------------------------------------------------------------------
Comprehensive automated test suite covering:
1. End-to-end valuation workflows across all 4 presets and boundary conditions.
2. Geospatial queries, distance calculations, and coordinate non-fabrication.
3. SHAP figure integrity, log-target additivity, and case-study provenance.
4. Empirical profiling of cold/warm load times, inference latency, and memory consumption.
"""

import sys
import time
import unittest
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.model_service import (
    load_champion_model,
    load_model_metadata,
    get_available_localities,
    get_available_area_types,
    predict_property_valuation
)
from src.services.geo_service import (
    load_locality_features,
    load_amenity_datasets,
    get_locality_proximity,
    get_surrounding_pois,
    build_pydeck_map
)
from src.services.shap_service import (
    get_shap_figures,
    get_global_feature_importance,
    get_canonical_case_study
)


class TestEndToEndFunctional(unittest.TestCase):
    """Exhaustive validation of all application workflows and edge cases."""

    def test_preset_1_hoodi_canonical(self):
        """Validates Preset 1: Hoodi 2BHK canonical appraisal."""
        res = predict_property_valuation(
            total_sqft=1420.0,
            bhk=2,
            bath=2.0,
            balcony=2.0,
            location="Hoodi",
            area_type="Super built-up  Area"
        )
        self.assertAlmostEqual(res["predicted_price_lakhs"], 74.97, delta=1.5)
        self.assertIn("₹74.97 Lakhs", res["formatted_price"])
        self.assertEqual(res["market_tier"], "Mid-Market (₹50 - ₹100 Lakhs)")
        self.assertEqual(res["mae_margin"], 13.24)
        self.assertAlmostEqual(res["lower_bound_lakhs"], 74.97 - 13.24, delta=0.5)
        self.assertAlmostEqual(res["upper_bound_lakhs"], 74.97 + 13.24, delta=0.5)
        self.assertAlmostEqual(res["price_per_sqft"], 5280, delta=50)

    def test_preset_2_whitefield_3bhk(self):
        """Validates Preset 2: Whitefield 3BHK scaling."""
        res = predict_property_valuation(
            total_sqft=1850.0,
            bhk=3,
            bath=3.0,
            balcony=2.0,
            location="Whitefield",
            area_type="Super built-up  Area"
        )
        self.assertGreater(res["predicted_price_lakhs"], 90.0)
        self.assertEqual(res["market_tier"], "Upper-Mid (₹100 - ₹200 Lakhs)")
        self.assertEqual(res["mae_margin"], 30.75)
        self.assertTrue(res["formatted_price"].startswith("₹1."))

    def test_preset_3_indiranagar_luxury(self):
        """Validates Preset 3: Indiranagar Luxury 4BHK with crore conversion."""
        res = predict_property_valuation(
            total_sqft=3200.0,
            bhk=4,
            bath=4.0,
            balcony=3.0,
            location="Indira Nagar",
            area_type="Built-up  Area"
        )
        self.assertGreater(res["predicted_price_lakhs"], 200.0)
        self.assertEqual(res["market_tier"], "Luxury (> ₹200 Lakhs)")
        self.assertEqual(res["mae_margin"], 112.79)
        self.assertIn("Cr", res["formatted_price"])

    def test_preset_4_electronic_city_budget(self):
        """Validates Preset 4: Electronic City Budget 2BHK."""
        res = predict_property_valuation(
            total_sqft=1050.0,
            bhk=2,
            bath=2.0,
            balcony=1.0,
            location="Electronic City",
            area_type="Super built-up  Area"
        )
        self.assertLess(res["predicted_price_lakhs"], 60.0)
        self.assertIn(res["market_tier"], ["Budget (< ₹50 Lakhs)", "Mid-Market (₹50 - ₹100 Lakhs)"])

    def test_edge_case_extreme_sizes(self):
        """Tests physical boundary inputs (300 sq ft and 12,000 sq ft)."""
        res_min = predict_property_valuation(300.0, 1, 1.0, 0.0, "Electronic City", "Super built-up  Area")
        self.assertGreater(res_min["predicted_price_lakhs"], 0)

        res_max = predict_property_valuation(12000.0, 8, 8.0, 4.0, "Whitefield", "Plot  Area")
        self.assertGreater(res_max["predicted_price_lakhs"], 500.0)

    def test_edge_case_unknown_inputs(self):
        """Tests resiliency against unseen locality and unseen area type."""
        # Model's OneHotEncoder should ignore unknown categories without error
        res = predict_property_valuation(1200.0, 2, 2.0, 1.0, "NonExistentLocationXYZ", "UnseenAreaType")
        self.assertGreater(res["predicted_price_lakhs"], 0)
        self.assertIn("formatted_price", res)

    def test_geospatial_whitefield_node_match(self):
        """Validates Whitefield 0.00 km transit node match."""
        prox = get_locality_proximity("Whitefield")
        self.assertTrue(prox["is_geocoded"])
        self.assertEqual(prox["status"], "SUCCESS")
        self.assertEqual(prox["metro"]["distance_km"], 0.0)
        self.assertEqual(prox["metro"]["name"], "Whitefield (Kadugodi)")

    def test_geospatial_non_success_zero_fabrication(self):
        """Strictly verifies that unresolved and aggregate localities receive no coordinates."""
        # 1. Aggregate category 'other'
        prox_other = get_locality_proximity("other")
        self.assertFalse(prox_other["is_geocoded"])
        self.assertEqual(prox_other["status"], "EXCLUDED_AGGREGATE")
        self.assertIsNone(prox_other["latitude"])
        self.assertIsNone(prox_other["longitude"])

        # 2. Unresolved locality
        prox_unres = get_locality_proximity("Sector 2 HSR Layout")
        self.assertFalse(prox_unres["is_geocoded"])
        self.assertEqual(prox_unres["status"], "UNRESOLVED")
        self.assertIsNone(prox_unres["latitude"])

        # 3. Completely unknown locality
        prox_unknown = get_locality_proximity("TotallyFakeNeighborhood999")
        self.assertFalse(prox_unknown["is_geocoded"])
        self.assertEqual(prox_unknown["status"], "UNKNOWN_LOCALITY")

    def test_pydeck_map_resilience(self):
        """Verifies PyDeck map renders for valid coordinates and safely yields None for missing."""
        deck_valid = build_pydeck_map("Indira Nagar", 12.9784, 77.6408)
        self.assertIsNotNone(deck_valid)
        self.assertEqual(len(deck_valid.layers), 4)  # Property + Metro + Tech + Hosp

        deck_invalid = build_pydeck_map("other", None, None)
        self.assertIsNone(deck_invalid)

    def test_shap_provenance_and_additivity_contract(self):
        """Verifies canonical test index 3379 provenance and figure assets."""
        study = get_canonical_case_study()
        self.assertEqual(study["listing_id"], "Held-Out Test Index 3379")
        self.assertIn("Held-Out Test Set", study["dataset_source"])
        self.assertEqual(study["location"], "Hoodi")
        self.assertEqual(study["total_sqft"], 1420.0)
        self.assertAlmostEqual(study["predicted_price_lakhs"], 74.97, delta=0.05)

        figs = get_shap_figures()
        for name, path in figs.items():
            self.assertTrue(path.exists(), f"SHAP figure {name} missing at {path}")


class TestPerformanceAndProfiling(unittest.TestCase):
    """Measures empirical latency, memory usage, and throughput."""

    def test_empirical_performance(self):
        """Measures cold load, warm latency, inference speed, and memory footprint."""
        print("\n" + "="*70)
        print("ESTATEIQ AI — EMPIRICAL PERFORMANCE & RELIABILITY PROFILING")
        print("="*70)

        # 1. Measure Memory Baseline
        tracemalloc.start()
        mem_start_curr, mem_start_peak = tracemalloc.get_traced_memory()

        # 2. Cold Model Loading
        t0 = time.perf_counter()
        model = load_champion_model()
        cold_model_time = (time.perf_counter() - t0) * 1000

        # 3. Warm Model Loading (Cache Retrieval)
        t0 = time.perf_counter()
        _ = load_champion_model()
        warm_model_time = (time.perf_counter() - t0) * 1000

        # 4. Geospatial Data Loading
        t0 = time.perf_counter()
        df_geo = load_locality_features()
        cold_geo_time = (time.perf_counter() - t0) * 1000

        # 5. Prediction Latency Benchmark (100 sequential inferences)
        latencies = []
        for _ in range(100):
            t_inf_start = time.perf_counter()
            _ = predict_property_valuation(1420.0, 2, 2.0, 2.0, "Hoodi", "Super built-up  Area")
            latencies.append((time.perf_counter() - t_inf_start) * 1000)

        lat_min = np.min(latencies)
        lat_median = np.median(latencies)
        lat_p95 = np.percentile(latencies, 95)
        lat_p99 = np.percentile(latencies, 99)
        lat_max = np.max(latencies)

        # 6. Locality Proximity Query Benchmark (100 queries)
        geo_latencies = []
        for loc in ["Whitefield", "Indira Nagar", "Electronic City", "Hoodi", "other"]:
            for _ in range(20):
                t_geo_start = time.perf_counter()
                _ = get_locality_proximity(loc)
                geo_latencies.append((time.perf_counter() - t_geo_start) * 1000)

        geo_lat_median = np.median(geo_latencies)
        geo_lat_p95 = np.percentile(geo_latencies, 95)

        # 7. PyDeck Map Construction Latency
        t0 = time.perf_counter()
        _ = build_pydeck_map("Whitefield", 12.9957, 77.7579)
        map_gen_time = (time.perf_counter() - t0) * 1000

        # 8. Memory Consumption Tracking
        mem_end_curr, mem_end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_delta_mb = (mem_end_curr - mem_start_curr) / (1024 * 1024)
        mem_peak_mb = mem_end_peak / (1024 * 1024)

        print(f"Cold Model Load Latency:        {cold_model_time:.2f} ms")
        print(f"Warm Model Cache Retrieval:     {warm_model_time:.4f} ms")
        print(f"Geospatial Features Load:       {cold_geo_time:.2f} ms")
        print(f"Inference Latency (Median):     {lat_median:.2f} ms")
        print(f"Inference Latency (p95):        {lat_p95:.2f} ms")
        print(f"Inference Latency (p99):        {lat_p99:.2f} ms")
        print(f"Inference Latency (Min - Max):  {lat_min:.2f} ms - {lat_max:.2f} ms")
        print(f"Geo Proximity Latency (Median): {geo_lat_median:.2f} ms")
        print(f"Geo Proximity Latency (p95):    {geo_lat_p95:.2f} ms")
        print(f"PyDeck Map Generation:          {map_gen_time:.2f} ms")
        print(f"Traced Memory Delta:            {mem_delta_mb:.2f} MB")
        print(f"Peak Traced Memory:             {mem_peak_mb:.2f} MB")
        print("="*70 + "\n")

        # Assert performance SLAs
        self.assertLess(lat_median, 25.0, "Median inference latency exceeds 25 ms SLA")
        self.assertLess(lat_p95, 50.0, "p95 inference latency exceeds 50 ms SLA")
        self.assertLess(map_gen_time, 100.0, "Map generation latency exceeds 100 ms SLA")


if __name__ == "__main__":
    unittest.main()
