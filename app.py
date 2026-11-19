"""
EstateIQ AI - Phase 6.5: Main Streamlit Application
---------------------------------------------------
AI-Powered Bengaluru Real Estate Valuation & Investment Intelligence Platform
Integrating XGBoost (Log-Target), OpenStreetMap Geospatial Intelligence,
and Explainable AI (SHAP).
"""

import sys
import logging
from pathlib import Path
import streamlit as st

# Configure project path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.valuation_calculator import render_valuation_calculator
from src.ui.geospatial_explorer import render_geospatial_explorer
from src.ui.shap_explainer_view import render_shap_explainer_view
from src.ui.market_insights import render_market_insights
from src.ui.locality_catalog import render_locality_catalog

# Page configuration
st.set_page_config(
    page_title="EstateIQ AI | Bengaluru Real Estate Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .badge-champion {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .sidebar-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 14px;
    }
    .sidebar-card-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .sidebar-card-item {
        font-size: 0.82rem;
        color: #475569;
        margin: 3px 0;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# SIDEBAR: Metadata & Model Architecture
with st.sidebar:
    st.markdown("## 🏢 EstateIQ AI")
    st.markdown("*Bengaluru Property Valuation & Geospatial Intelligence*")
    st.markdown('<span class="badge-champion">Phase 6.5 • Production Ready</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Champion Model Badge Card
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🥇 Champion Valuation Engine</div>
            <div class="sidebar-card-item"><strong>Algorithm:</strong> XGBoost Regressor</div>
            <div class="sidebar-card-item"><strong>Target Scale:</strong> Log(1 + Price)</div>
            <div class="sidebar-card-item"><strong>Test R²:</strong> 0.7667 (&gt; 0.75 goal)</div>
            <div class="sidebar-card-item"><strong>Test MAE:</strong> ₹25.13 Lakhs</div>
            <div class="sidebar-card-item"><strong>Test RMSE:</strong> ₹59.09 Lakhs</div>
            <div class="sidebar-card-item"><strong>CV MAE:</strong> ₹24.34 ± 1.51 Lakhs</div>
            <div class="sidebar-card-item"><strong>Trees:</strong> 300 estimators (depth=6)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Geospatial Intelligence Card
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🗺️ Geospatial Intelligence</div>
            <div class="sidebar-card-item"><strong>Recognized Localities:</strong> 234</div>
            <div class="sidebar-card-item"><strong>Geocoded Centroids:</strong> 211 (90.2%)</div>
            <div class="sidebar-card-item"><strong>Transit / Metro Nodes:</strong> 481 (OSM stations, stops & entrances)</div>
            <div class="sidebar-card-item"><strong>Tech Parks:</strong> 130 hubs</div>
            <div class="sidebar-card-item"><strong>Hospitals:</strong> 1,161 centers</div>
            <div class="sidebar-card-item"><strong>Geodesic Engine:</strong> WGS84 Ellipsoid</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Explainability Card
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🔍 Explainable AI (SHAP)</div>
            <div class="sidebar-card-item"><strong>Method:</strong> TreeSHAP (Cooperative Game Theory)</div>
            <div class="sidebar-card-item"><strong>Base Value:</strong> 4.3578 log units (₹77.08L)</div>
            <div class="sidebar-card-item"><strong>Additivity Discrepancy:</strong> &le; 8.58e-06 log units</div>
            <div class="sidebar-card-item"><strong>Top Predictor:</strong> total_sqft (|SHAP| = 0.381)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Developed as an institutional-grade tabular ML & geospatial intelligence system.")


# MAIN CONTENT AREA
st.markdown('<div class="main-header">EstateIQ AI — Bengaluru Real Estate Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Valuation Engine, OpenStreetMap Geospatial Proximities, and Explainable AI (SHAP)</div>',
    unsafe_allow_html=True
)

# Main Navigation Tabs
tab_val, tab_shap, tab_tourn, tab_cat = st.tabs([
    "🏡 Property Valuation & Spatial Explorer",
    "🔍 Explainable AI (SHAP)",
    "📊 Model Benchmarks & Residual Tiers",
    "🗺️ Locality Proximity Catalog"
])

# TAB 1: VALUATION & SPATIAL EXPLORER
with tab_val:
    inputs, prediction = render_valuation_calculator()
    st.markdown("---")
    render_geospatial_explorer(inputs["location"])

# TAB 2: EXPLAINABLE AI (SHAP)
with tab_shap:
    render_shap_explainer_view()

# TAB 3: MODEL TOURNAMENT & BENCHMARKS
with tab_tourn:
    render_market_insights()

# TAB 4: LOCALITY PROXIMITY CATALOG
with tab_cat:
    render_locality_catalog()

# APPLICATION FOOTER
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
        <strong>EstateIQ AI Platform</strong> • Built with Python, Streamlit, XGBoost, PyDeck & GeoPy.<br/>
        Spatial data © <a href="https://www.openstreetmap.org/copyright" target="_blank" style="color: #475569;">OpenStreetMap contributors</a> (ODbL 1.0).<br/>
        <em>Academic & Portfolio Demonstrator: Valuations represent statistical models of historical market transactions and should not replace certified professional real-estate appraisals.</em>
    </div>
    """,
    unsafe_allow_html=True
)
