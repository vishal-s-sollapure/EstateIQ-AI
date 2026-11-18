"""
EstateIQ AI - Phase 6.3: Model Inference & Valuation Service
------------------------------------------------------------
Provides cached access to the Phase 5 Champion XGBoost model
(models/estateiq_advanced_model.joblib) and executes production-grade
single-listing real estate appraisals with confidence intervals.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib

# Optional Streamlit cache decorator
try:
    import streamlit as st
    cache_resource = st.cache_resource
except ImportError:
    def cache_resource(f):
        return f

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger("EstateIQ-ModelService")

MODEL_PATH = project_root / "models" / "estateiq_advanced_model.joblib"
METADATA_PATH = project_root / "models" / "model_metadata.json"
DATA_PATH = project_root / "data" / "processed" / "bengaluru_cleaned.csv"

# Empirical MAE bounds by price bracket (from Phase 5 residual diagnostics on held-out test set)
# Note: In historical benchmarking, brackets were binned on actual prices (y_true).
# In production inference, brackets are selected conditioning on predicted price (y_pred).
# These margins represent empirical residual scales, NOT calibrated confidence intervals.
TIERED_MAE = {
    "budget": 8.55,       # < 50 Lakhs
    "mid": 13.24,         # 50 - 100 Lakhs
    "upper_mid": 30.75,   # 100 - 200 Lakhs
    "luxury": 112.79      # > 200 Lakhs
}


@cache_resource
def load_champion_model():
    """Loads and caches the Phase 5 Champion XGBoost model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Champion model artifact not found at: {MODEL_PATH}")
    logger.info(f"Loading champion model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


@cache_resource
def load_model_metadata() -> Dict[str, Any]:
    """Loads and caches model metadata and performance metrics."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@cache_resource
def get_available_localities() -> List[str]:
    """Retrieves sorted list of 234 unique localities."""
    geo_csv = project_root / "data" / "geospatial" / "locality_geospatial_features.csv"
    if geo_csv.exists():
        df_geo = pd.read_csv(geo_csv)
        locs = df_geo["locality"].tolist()
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        locs = sorted(df["location"].unique().tolist())
    else:
        locs = ["Whitefield", "Sarjapur Road", "Electronic City", "Hebbal", "other"]

    # Place top high-demand hubs first, then alphabetical
    popular_hubs = ["Whitefield", "Sarjapur  Road", "Electronic City", "Kanakpura Road", "Thanisandra", "Yelahanka", "Hebbal", "Indira Nagar", "Rajaji Nagar"]
    existing_popular = [h for h in popular_hubs if h in locs]
    remaining = sorted([loc for loc in locs if loc not in existing_popular and loc != "other"])
    
    sorted_locs = existing_popular + remaining
    if "other" in locs:
        sorted_locs.append("other")
    return sorted_locs


@cache_resource
def get_available_area_types() -> List[str]:
    """Returns valid area types matching training data contract."""
    return [
        "Super built-up  Area",
        "Built-up  Area",
        "Plot  Area",
        "Carpet  Area"
    ]


def predict_property_valuation(
    total_sqft: float,
    bhk: int,
    bath: float,
    balcony: float,
    location: str,
    area_type: str
) -> Dict[str, Any]:
    """
    Executes fair market valuation appraisal for a single residential property.

    Returns:
        Dict containing predicted price in Lakhs & Crores, confidence band,
        price per sq ft, and inference metadata.
    """
    model = load_champion_model()

    input_df = pd.DataFrame([{
        "total_sqft": float(total_sqft),
        "bhk": int(bhk),
        "bath": float(bath),
        "balcony": float(balcony),
        "location": str(location).strip(),
        "area_type": str(area_type)
    }])

    # Model predicts in ₹ Lakhs (TransformedTargetRegressor automatically inverts log1p)
    raw_pred_lakhs = float(model.predict(input_df)[0])
    pred_lakhs = max(1.0, round(raw_pred_lakhs, 2))

    # Determine empirical confidence interval based on price tier
    if pred_lakhs < 50.0:
        mae_margin = TIERED_MAE["budget"]
        market_tier = "Budget (< ₹50 Lakhs)"
    elif pred_lakhs < 100.0:
        mae_margin = TIERED_MAE["mid"]
        market_tier = "Mid-Market (₹50 - ₹100 Lakhs)"
    elif pred_lakhs < 200.0:
        mae_margin = TIERED_MAE["upper_mid"]
        market_tier = "Upper-Mid (₹100 - ₹200 Lakhs)"
    else:
        mae_margin = TIERED_MAE["luxury"]
        market_tier = "Luxury (> ₹200 Lakhs)"

    lower_bound = max(1.0, round(pred_lakhs - mae_margin, 2))
    upper_bound = round(pred_lakhs + mae_margin, 2)

    # Calculate estimated price per sqft
    price_per_sqft = round((pred_lakhs * 100_000) / max(1.0, total_sqft))

    # Dual currency formatting
    if pred_lakhs >= 100.0:
        formatted_primary = f"₹{pred_lakhs / 100:.2f} Cr"
        formatted_secondary = f"(₹{pred_lakhs:.2f} Lakhs)"
    else:
        formatted_primary = f"₹{pred_lakhs:.2f} Lakhs"
        formatted_secondary = f"(₹{pred_lakhs * 100_000:,.0f})"

    return {
        "predicted_price_lakhs": pred_lakhs,
        "formatted_price": formatted_primary,
        "formatted_subtext": formatted_secondary,
        "lower_bound_lakhs": lower_bound,
        "upper_bound_lakhs": upper_bound,
        "mae_margin": mae_margin,
        "market_tier": market_tier,
        "price_per_sqft": price_per_sqft,
        "inputs": {
            "total_sqft": total_sqft,
            "bhk": bhk,
            "bath": bath,
            "balcony": balcony,
            "location": location,
            "area_type": area_type
        }
    }
