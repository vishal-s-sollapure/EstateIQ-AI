"""
EstateIQ AI - Phase 6.3: Property Valuation Calculator UI
--------------------------------------------------------
Interactive property specification controls and real-time appraisal output
powered by the Phase 5 Champion XGBoost pipeline.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple

from src.services.model_service import (
    get_available_localities,
    get_available_area_types,
    predict_property_valuation
)


def render_valuation_calculator() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Renders the property specification input controls and calculated appraisal.

    Returns:
        Tuple of (inputs_dict, prediction_dict)
    """
    st.markdown("### 🏡 Residential Property Valuation Engine")
    st.markdown(
        "Enter property specifications below to generate an institutional-grade fair market "
        "valuation backed by the **Phase 5 Champion XGBoost model** ($R^2 = 0.7667$). "
        "Valuations are trained on 11,252 verified Bengaluru property listings / asking prices."
    )

    # Preset configurations for quick testing
    preset_cols = st.columns([1, 1, 1, 1])
    preset_choice = None
    with preset_cols[0]:
        if st.button("📌 Preset: Hoodi Canonical 2BHK", use_container_width=True):
            st.session_state["calc_loc"] = "Hoodi"
            st.session_state["calc_area"] = "Super built-up  Area"
            st.session_state["calc_sqft"] = 1420
            st.session_state["calc_bhk"] = 2
            st.session_state["calc_bath"] = 2
            st.session_state["calc_balcony"] = 2
    with preset_cols[1]:
        if st.button("📌 Preset: Whitefield 3BHK", use_container_width=True):
            st.session_state["calc_loc"] = "Whitefield"
            st.session_state["calc_area"] = "Super built-up  Area"
            st.session_state["calc_sqft"] = 1850
            st.session_state["calc_bhk"] = 3
            st.session_state["calc_bath"] = 3
            st.session_state["calc_balcony"] = 2
    with preset_cols[2]:
        if st.button("📌 Preset: Indiranagar Luxury 4BHK", use_container_width=True):
            st.session_state["calc_loc"] = "Indira Nagar"
            st.session_state["calc_area"] = "Built-up  Area"
            st.session_state["calc_sqft"] = 3200
            st.session_state["calc_bhk"] = 4
            st.session_state["calc_bath"] = 4
            st.session_state["calc_balcony"] = 3
    with preset_cols[3]:
        if st.button("📌 Preset: Electronic City Budget 2BHK", use_container_width=True):
            st.session_state["calc_loc"] = "Electronic City"
            st.session_state["calc_area"] = "Super built-up  Area"
            st.session_state["calc_sqft"] = 1050
            st.session_state["calc_bhk"] = 2
            st.session_state["calc_bath"] = 2
            st.session_state["calc_balcony"] = 1

    localities = get_available_localities()
    area_types = get_available_area_types()

    # Default session state initialization
    if "calc_loc" not in st.session_state:
        st.session_state["calc_loc"] = "Whitefield"
    if "calc_area" not in st.session_state:
        st.session_state["calc_area"] = "Super built-up  Area"
    if "calc_sqft" not in st.session_state:
        st.session_state["calc_sqft"] = 1420
    if "calc_bhk" not in st.session_state:
        st.session_state["calc_bhk"] = 2
    if "calc_bath" not in st.session_state:
        st.session_state["calc_bath"] = 2
    if "calc_balcony" not in st.session_state:
        st.session_state["calc_balcony"] = 2

    # Input Form Layout
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            loc_idx = localities.index(st.session_state["calc_loc"]) if st.session_state["calc_loc"] in localities else 0
            location = st.selectbox(
                "Locality / Neighborhood",
                options=localities,
                index=loc_idx,
                help="Select from 234 verified Bengaluru localities or 'other' for miscellaneous regions."
            )
            st.session_state["calc_loc"] = location

            area_idx = area_types.index(st.session_state["calc_area"]) if st.session_state["calc_area"] in area_types else 0
            area_type = st.selectbox(
                "Area Classification",
                options=area_types,
                index=area_idx,
                help="Super built-up includes undivided land share + amenities; Plot Area represents freehold land."
            )
            st.session_state["calc_area"] = area_type

            total_sqft = st.number_input(
                "Total Built-up Area (sq. ft.)",
                min_value=300,
                max_value=12000,
                value=int(st.session_state["calc_sqft"]),
                step=25,
                help="Usable residential area in square feet. Typical 2 BHK: 950 - 1,450 sq ft."
            )
            st.session_state["calc_sqft"] = total_sqft

        with col2:
            bhk = st.slider(
                "Bedrooms (BHK)",
                min_value=1,
                max_value=10,
                value=int(st.session_state["calc_bhk"]),
                step=1,
                help="Number of bedrooms."
            )
            st.session_state["calc_bhk"] = bhk

            bath = st.slider(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=int(st.session_state["calc_bath"]),
                step=1,
                help="Number of bathrooms (critical luxury layout indicator in Bengaluru)."
            )
            st.session_state["calc_bath"] = bath

            balcony = st.slider(
                "Balconies",
                min_value=0,
                max_value=5,
                value=int(st.session_state["calc_balcony"]),
                step=1,
                help="Number of attached balconies."
            )
            st.session_state["calc_balcony"] = balcony

    # Execute Model Inference
    prediction = predict_property_valuation(
        total_sqft=float(total_sqft),
        bhk=int(bhk),
        bath=float(bath),
        balcony=float(balcony),
        location=str(location),
        area_type=str(area_type)
    )

    # Render Valuation Results
    st.markdown("#### 🎯 Fair Market Valuation Estimate")

    metric_cols = st.columns([1.5, 1, 1, 1])

    with metric_cols[0]:
        st.metric(
            label="Estimated Market Value",
            value=prediction["formatted_price"],
            delta=prediction["formatted_subtext"],
            delta_color="off"
        )

    with metric_cols[1]:
        st.metric(
            label="Historical Residual-Scale Estimate",
            value=f"₹{prediction['lower_bound_lakhs']:.1f}L – ₹{prediction['upper_bound_lakhs']:.1f}L",
            delta=f"Scale: ±₹{prediction['mae_margin']:.2f}L (Tier MAE)",
            delta_color="off",
            help="Empirical error spread based on held-out test MAE for this price bracket. This represents a historical residual-scale estimate, not a calibrated statistical prediction interval."
        )

    with metric_cols[2]:
        st.metric(
            label="Unit Rate",
            value=f"₹{prediction['price_per_sqft']:,} / sq.ft.",
            help="Estimated price per square foot of total built-up area."
        )

    with metric_cols[3]:
        st.metric(
            label="Market Tier",
            value=prediction["market_tier"].split("(")[0].strip(),
            help=f"Classified into {prediction['market_tier']} based on empirical residual diagnostics."
        )

    # Contextual Guidance Banner
    if prediction["predicted_price_lakhs"] > 200.0:
        st.info(
            "ℹ️ **Luxury Tier Advisory**: For properties above ₹2.00 Crores, empirical residual standard deviation "
            "widens (±₹151 Lakhs) due to bespoke architectural specifications, builder brand equity, and interior luxury tiers "
            "not captured solely by floor area."
        )
    elif location == "other":
        st.warning(
            "⚠️ **Aggregated Category Notice**: You selected 'other'. This category represents an aggregate of "
            "infrequent locations (< 10 historical occurrences) across Bengaluru. Expect wider valuation variance."
        )

    inputs = {
        "location": location,
        "area_type": area_type,
        "total_sqft": total_sqft,
        "bhk": bhk,
        "bath": bath,
        "balcony": balcony
    }

    return inputs, prediction
