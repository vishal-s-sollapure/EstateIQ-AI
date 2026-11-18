"""
EstateIQ AI - Phase 6.3: Explainable AI (SHAP) Service
------------------------------------------------------
Provides structured access to SHAP interpretability metadata, global feature
importance rankings, canonical local case studies, and visualization paths.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Optional Streamlit cache decorator
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(f):
        return f

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger("EstateIQ-SHAPService")

FIGURES_DIR = project_root / "reports" / "figures"


@cache_data
def get_shap_figures() -> Dict[str, Path]:
    """Returns absolute paths to the precomputed SHAP diagnostic plots."""
    return {
        "summary": FIGURES_DIR / "06_shap_summary.png",
        "bar_importance": FIGURES_DIR / "07_shap_bar_importance.png",
        "waterfall": FIGURES_DIR / "08_shap_waterfall.png",
        "dependence": FIGURES_DIR / "09_shap_dependence_sqft.png",
    }


@cache_data
def get_global_feature_importance() -> List[Dict[str, Any]]:
    """
    Returns the top 15 features ranked by Mean Absolute SHAP value
    computed across the held-out test partition (N = 2,251).
    """
    return [
        {"rank": 1, "feature": "total_sqft", "category": "Physical Dimension", "mean_abs_shap": 0.3810, "direction": "Strongly Positive for > 1,500 sq ft"},
        {"rank": 2, "feature": "Loc: Giri Nagar", "category": "Locality", "mean_abs_shap": 0.2909, "direction": "Positive micro-market premium"},
        {"rank": 3, "feature": "Loc: Rajaji Nagar", "category": "Locality", "mean_abs_shap": 0.2088, "direction": "High premium heritage enclave"},
        {"rank": 4, "feature": "Loc: Indira Nagar", "category": "Locality", "mean_abs_shap": 0.2018, "direction": "Prime commercial/residential corridor"},
        {"rank": 5, "feature": "Loc: Basavangudi", "category": "Locality", "mean_abs_shap": 0.1655, "direction": "South Bengaluru cultural core"},
        {"rank": 6, "feature": "Loc: 1st Block Jayanagar", "category": "Locality", "mean_abs_shap": 0.1602, "direction": "High-value planned neighborhood"},
        {"rank": 7, "feature": "Loc: HAL 2nd Stage", "category": "Locality", "mean_abs_shap": 0.1534, "direction": "Central east tech corridor premium"},
        {"rank": 8, "feature": "Loc: Ulsoor", "category": "Locality", "mean_abs_shap": 0.1503, "direction": "CBD-adjacent premium"},
        {"rank": 9, "feature": "Loc: Chamrajpet", "category": "Locality", "mean_abs_shap": 0.1390, "direction": "Historic central market location"},
        {"rank": 10, "feature": "Loc: Cooke Town", "category": "Locality", "mean_abs_shap": 0.1355, "direction": "Cantonment residential premium"},
        {"rank": 11, "feature": "Loc: Banashankari Stage II", "category": "Locality", "mean_abs_shap": 0.1251, "direction": "South Bengaluru established hub"},
        {"rank": 12, "feature": "Loc: Koramangala", "category": "Locality", "mean_abs_shap": 0.1241, "direction": "Startup and retail corridor premium"},
        {"rank": 13, "feature": "Loc: Chandapura", "category": "Locality", "mean_abs_shap": 0.1208, "direction": "Suburban affordability discount"},
        {"rank": 14, "feature": "Loc: Domlur", "category": "Locality", "mean_abs_shap": 0.1199, "direction": "Old Airport Road business hub"},
        {"rank": 15, "feature": "Loc: Anekal", "category": "Locality", "mean_abs_shap": 0.1194, "direction": "Periphery industrial discount"},
    ]


@cache_data
def get_canonical_case_study() -> Dict[str, Any]:
    """
    Returns verified ground-truth and SHAP decomposition details
    for the canonical test listing (Hoodi 2 BHK).
    """
    return {
        "listing_id": "Test Index 3456",
        "location": "Hoodi",
        "total_sqft": 1420.0,
        "bhk": 2,
        "bath": 2.0,
        "balcony": 2.0,
        "area_type": "Super built-up  Area",
        "actual_price_lakhs": 75.00,
        "predicted_price_lakhs": 74.97,
        "absolute_error_lakhs": 0.03,
        "relative_error_pct": 0.04,
        "base_value_log": 4.3578,
        "base_value_lakhs": 77.08,
        "model_output_log": 4.3303,
        "contributions": [
            {"feature": "total_sqft (1420.0)", "shap_log": +0.21, "impact": "Positive (Spacious 2 BHK layout)"},
            {"feature": "bath (2.0)", "shap_log": +0.02, "impact": "Positive (Sufficient sanitary ratio)"},
            {"feature": "location (Hoodi)", "shap_log": -0.19, "impact": "Suburban price adjustment vs. Bengaluru CBD mean"},
            {"feature": "area_type (Super built-up Area)", "shap_log": -0.05, "impact": "Baseline apartment undivided share adjustment"},
            {"feature": "balcony (2.0)", "shap_log": -0.02, "impact": "Minor secondary adjustment"},
        ]
    }
