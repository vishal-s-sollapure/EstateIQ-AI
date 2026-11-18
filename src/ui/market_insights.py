"""
EstateIQ AI - Phase 6.3: Market Insights & Model Tournament UI
--------------------------------------------------------------
Displays tournament benchmark comparisons, 5-fold cross-validation results,
residual heteroscedasticity diagnostics, and dataset distribution charts.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = project_root / "reports" / "figures"


def render_market_insights():
    """
    Renders comprehensive model tournament benchmarking tables,
    residual diagnostics across price brackets, and dataset EDA figures.
    """
    st.markdown("## 📊 Model Tournament Benchmarks & Residual Diagnostics")
    st.markdown(
        "EstateIQ AI evaluates models using strict academic rigor: **5-Fold Cross-Validation** on the training set "
        "($N = 9,001$) and evaluation on a strictly held-out test partition ($N = 2,251$) untouched during hyperparameter tuning."
    )

    t_tab1, t_tab2, t_tab3 = st.tabs([
        "🏆 Model Tournament Leaderboard",
        "📉 Heteroscedasticity & Residual Tiers",
        "📈 Exploratory Market Distributions"
    ])

    # TAB 1: TOURNAMENT BENCHMARK
    with t_tab1:
        st.markdown("### 🥇 Advanced Gradient Boosting Tournament Results")
        st.markdown(
            "Comparing **XGBoost**, **LightGBM**, and **CatBoost** under both raw continuous pricing and "
            "log-transformed targets ($\\log(1+y)$) against the Phase 4 **Random Forest baseline**:"
        )

        tournament_data = [
            {"Model Configuration": "🥇 XGBoost (Log Target)", "CV MAE (₹L)": "₹24.34 ± 1.51", "CV RMSE (₹L)": "₹59.81", "CV R²": "0.6952", "Train R²": "0.8085", "Overfit Gap": "0.1133", "Test MAE (₹L)": "₹25.13", "Test RMSE (₹L)": "₹59.09", "Test R²": "0.7667", "Train Time": "0.4s"},
            {"Model Configuration": "Random Forest (Raw Price) [Phase 4]", "CV MAE (₹L)": "₹24.70 ± 0.82", "CV RMSE (₹L)": "₹59.48", "CV R²": "0.6884", "Train R²": "0.9580", "Overfit Gap": "0.2697", "Test MAE (₹L)": "₹25.94", "Test RMSE (₹L)": "₹71.22", "Test R²": "0.6611", "Train Time": "3.4s"},
            {"Model Configuration": "LightGBM [Ordinal] (Log Target)", "CV MAE (₹L)": "₹24.71 ± 1.57", "CV RMSE (₹L)": "₹61.17", "CV R²": "0.6817", "Train R²": "0.7896", "Overfit Gap": "0.1079", "Test MAE (₹L)": "₹25.11", "Test RMSE (₹L)": "₹60.23", "Test R²": "0.7576", "Train Time": "0.6s"},
            {"Model Configuration": "Random Forest (Log Target)", "CV MAE (₹L)": "₹24.93 ± 1.20", "CV RMSE (₹L)": "₹61.08", "CV R²": "0.6745", "Train R²": "0.9416", "Overfit Gap": "0.2671", "Test MAE (₹L)": "₹26.44", "Test RMSE (₹L)": "₹73.47", "Test R²": "0.6393", "Train Time": "3.5s"},
            {"Model Configuration": "CatBoost (Log Target)", "CV MAE (₹L)": "₹25.43 ± 1.68", "CV RMSE (₹L)": "₹61.85", "CV R²": "0.6743", "Train R²": "0.7584", "Overfit Gap": "0.0841", "Test MAE (₹L)": "₹26.88", "Test RMSE (₹L)": "₹63.06", "Test R²": "0.7343", "Train Time": "2.7s"},
            {"Model Configuration": "XGBoost (Raw Price)", "CV MAE (₹L)": "₹25.58 ± 1.69", "CV RMSE (₹L)": "₹65.65", "CV R²": "0.6288", "Train R²": "0.9086", "Overfit Gap": "0.2798", "Test MAE (₹L)": "₹26.00", "Test RMSE (₹L)": "₹60.70", "Test R²": "0.7538", "Train Time": "0.5s"},
            {"Model Configuration": "LightGBM (Log Target)", "CV MAE (₹L)": "₹25.72 ± 1.59", "CV RMSE (₹L)": "₹63.79", "CV R²": "0.6539", "Train R²": "0.7591", "Overfit Gap": "0.1052", "Test MAE (₹L)": "₹26.28", "Test RMSE (₹L)": "₹62.17", "Test R²": "0.7417", "Train Time": "0.6s"},
            {"Model Configuration": "CatBoost (Raw Price)", "CV MAE (₹L)": "₹26.56 ± 1.49", "CV RMSE (₹L)": "₹60.68", "CV R²": "0.6846", "Train R²": "0.8120", "Overfit Gap": "0.1275", "Test MAE (₹L)": "₹27.68", "Test RMSE (₹L)": "₹61.94", "Test R²": "0.7436", "Train Time": "2.6s"},
            {"Model Configuration": "LightGBM (Raw Price)", "CV MAE (₹L)": "₹27.45 ± 1.66", "CV RMSE (₹L)": "₹65.59", "CV R²": "0.6325", "Train R²": "0.7834", "Overfit Gap": "0.1510", "Test MAE (₹L)": "₹27.97", "Test RMSE (₹L)": "₹65.79", "Test R²": "0.7107", "Train Time": "0.3s"},
        ]

        df_tour = pd.DataFrame(tournament_data)
        st.dataframe(df_tour, use_container_width=True, hide_index=True)

        fig_tour = FIGURES_DIR / "05_advanced_model_comparison.png"
        if fig_tour.exists():
            st.image(str(fig_tour), caption="Figure 05: 5-Fold Cross-Validation & Test Generalization Across Advanced Architectures", use_container_width=True)

    # TAB 2: RESIDUAL TIERS & HETEROSCEDASTICITY
    with t_tab2:
        st.markdown("### 📉 Residual Heteroscedasticity Analysis by Price Tier")
        st.markdown(
            "Target transformation via $\\log(1+y)$ stabilizes variance across orders of magnitude. "
            "However, residual standard deviation naturally widens in high-ticket luxury real estate:"
        )

        residual_data = [
            {"Price Tier": "Budget (< ₹50 Lakhs)", "Test Listings (N)": 617, "Mean Actual (₹L)": "₹38.89", "Mean Predicted (₹L)": "₹45.86", "Mean Error (₹L)": "-₹6.97", "Residual Std (₹L)": "₹9.13", "MAE (₹L)": "₹8.55"},
            {"Price Tier": "Mid-Market (₹50 - ₹100 Lakhs)", "Test Listings (N)": 990, "Mean Actual (₹L)": "₹71.74", "Mean Predicted (₹L)": "₹71.56", "Mean Error (₹L)": "+₹0.18", "Residual Std (₹L)": "₹18.61", "MAE (₹L)": "₹13.24"},
            {"Price Tier": "Upper-Mid (₹100 - ₹200 Lakhs)", "Test Listings (N)": 420, "Mean Actual (₹L)": "₹137.67", "Mean Predicted (₹L)": "₹128.90", "Mean Error (₹L)": "+₹8.77", "Residual Std (₹L)": "₹42.47", "MAE (₹L)": "₹30.75"},
            {"Price Tier": "Luxury (> ₹200 Lakhs)", "Test Listings (N)": 224, "Mean Actual (₹L)": "₹381.81", "Mean Predicted (₹L)": "₹298.44", "Mean Error (₹L)": "+₹83.37", "Residual Std (₹L)": "₹151.09", "MAE (₹L)": "₹112.79"},
        ]

        df_res = pd.DataFrame(residual_data)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Diagnostic Insights**:
            1. **Budget Precision (MAE ₹8.55L):** The model excels in the high-volume affordable housing sector (< ₹50 Lakhs).
            2. **Mid-Market Neutrality (MAE ₹13.24L):** The core ₹50L - ₹100L segment exhibits negligible mean residual bias (+₹0.18L).
            3. **Luxury Spread (Residual Std ₹151.09L):** High-end villas and penthouses display increased variance because bespoke finishes and ultra-premium brand names cannot be fully quantified through tabular specifications alone.
            """
        )

    # TAB 3: EXPLORATORY MARKET DISTRIBUTIONS
    with t_tab3:
        st.markdown("### 📈 Bengaluru Housing Market Distributions (Phase 3 EDA)")
        st.markdown(
            "Foundational statistical patterns extracted from the 11,252 cleaned residential transactions:"
        )

        eda_cols = st.columns(3)
        fig_price = FIGURES_DIR / "01_price_distribution.png"
        fig_bhk = FIGURES_DIR / "02_bhk_and_area_analysis.png"
        fig_loc = FIGURES_DIR / "03_locality_and_correlation.png"

        with eda_cols[0]:
            if fig_price.exists():
                st.image(str(fig_price), caption="Figure 01: Raw vs Log Price Distribution", use_container_width=True)
        with eda_cols[1]:
            if fig_bhk.exists():
                st.image(str(fig_bhk), caption="Figure 02: BHK & Square Footage Scaling", use_container_width=True)
        with eda_cols[2]:
            if fig_loc.exists():
                st.image(str(fig_loc), caption="Figure 03: Locality Premiums & Feature Correlation", use_container_width=True)
