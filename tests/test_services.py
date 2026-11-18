"""
EstateIQ AI - Test Suite for Services & Streamlit Backing Layer
--------------------------------------------------------------
Automated regression tests verifying model inference, geospatial queries,
and SHAP metadata services.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.model_service import (
    load_champion_model,
    get_available_localities,
    get_available_area_types,
    predict_property_valuation
)
from src.services.geo_service import (
    get_locality_proximity,
    get_surrounding_pois,
    build_pydeck_map
)
from src.services.shap_service import (
    get_shap_figures,
    get_global_feature_importance,
    get_canonical_case_study
)


class TestModelService(unittest.TestCase):
    """Verifies champion model loading, inference contract, and confidence bands."""

    def test_model_loading(self):
        model = load_champion_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_localities_and_area_types(self):
        localities = get_available_localities()
        self.assertGreaterEqual(len(localities), 200)
        self.assertIn("Whitefield", localities)
        self.assertIn("other", localities)

        area_types = get_available_area_types()
        self.assertEqual(len(area_types), 4)
        self.assertIn("Super built-up  Area", area_types)

    def test_canonical_prediction(self):
        res = predict_property_valuation(
            total_sqft=1420.0,
            bhk=2,
            bath=2.0,
            balcony=2.0,
            location="Hoodi",
            area_type="Super built-up  Area"
        )
        self.assertIn("predicted_price_lakhs", res)
        # Expected around ~74.97 Lakhs
        self.assertAlmostEqual(res["predicted_price_lakhs"], 74.97, delta=5.0)
        self.assertIn("formatted_price", res)
        self.assertIn("lower_bound_lakhs", res)
        self.assertIn("upper_bound_lakhs", res)
        self.assertLess(res["lower_bound_lakhs"], res["predicted_price_lakhs"])
        self.assertGreater(res["upper_bound_lakhs"], res["predicted_price_lakhs"])
        self.assertGreater(res["price_per_sqft"], 0)

    def test_other_locality_prediction(self):
        res = predict_property_valuation(
            total_sqft=1200.0,
            bhk=2,
            bath=2.0,
            balcony=1.0,
            location="other",
            area_type="Super built-up  Area"
        )
        self.assertGreater(res["predicted_price_lakhs"], 0)


class TestGeoService(unittest.TestCase):
    """Verifies geospatial queries, fallback behaviors, and PyDeck map construction."""

    def test_valid_locality_proximity(self):
        prox = get_locality_proximity("Whitefield")
        self.assertTrue(prox["is_geocoded"])
        self.assertEqual(prox["status"], "SUCCESS")
        self.assertIsNotNone(prox["latitude"])
        self.assertIsNotNone(prox["longitude"])
        self.assertIsNotNone(prox["metro"]["distance_km"])
        self.assertIsNotNone(prox["tech_park"]["distance_km"])
        self.assertIsNotNone(prox["hospital"]["distance_km"])

    def test_other_locality_excluded(self):
        prox = get_locality_proximity("other")
        self.assertFalse(prox["is_geocoded"])
        self.assertEqual(prox["status"], "EXCLUDED_AGGREGATE")
        self.assertIsNone(prox["latitude"])

    def test_pydeck_map_generation(self):
        # Valid coords
        deck = build_pydeck_map("Whitefield", 12.9957, 77.7579)
        self.assertIsNotNone(deck)
        self.assertGreaterEqual(len(deck.layers), 1)

        # Invalid coords
        deck_none = build_pydeck_map("other", None, None)
        self.assertIsNone(deck_none)


class TestSHAPService(unittest.TestCase):
    """Verifies SHAP interpretability metadata and figure assets."""

    def test_figures_exist(self):
        figs = get_shap_figures()
        self.assertTrue(figs["summary"].exists(), "06_shap_summary.png missing")
        self.assertTrue(figs["bar_importance"].exists(), "07_shap_bar_importance.png missing")
        self.assertTrue(figs["waterfall"].exists(), "08_shap_waterfall.png missing")
        self.assertTrue(figs["dependence"].exists(), "09_shap_dependence_sqft.png missing")

    def test_feature_importance_top(self):
        imp = get_global_feature_importance()
        self.assertEqual(len(imp), 15)
        self.assertEqual(imp[0]["feature"], "total_sqft")

    def test_canonical_case_study(self):
        study = get_canonical_case_study()
        self.assertEqual(study["location"], "Hoodi")
        self.assertEqual(study["actual_price_lakhs"], 75.00)
        self.assertEqual(study["predicted_price_lakhs"], 74.97)


if __name__ == "__main__":
    unittest.main()
