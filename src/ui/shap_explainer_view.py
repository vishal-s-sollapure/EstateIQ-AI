"""
EstateIQ AI - Phase 6.3: Explainable AI (SHAP) View
--------------------------------------------------
Interactive model transparency and decision audit interface powered by
TreeSHAP (shap.TreeExplainer) and Game Theory.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.services.shap_service import (
    get_shap_figures,
    get_global_feature_importance,
    get_canonical_case_study
)


def render_shap_explainer_view():
    """
    Renders comprehensive Explainable AI (SHAP) diagnostic visualizations,
    canonical local attribution walkthroughs, and governance documentation.
    """
    st.markdown("## 🔍 Explainable AI (SHAP): Model Transparency & Decision Auditing")
    st.markdown(
        "Modern real estate valuation requires more than black-box accuracy. EstateIQ AI integrates "
        "**TreeSHAP** (`shap.TreeExplainer`), grounded in cooperative game theory, to provide "
        "provably fair, additive attribution for every single prediction."
    )

    # Mathematical Formula Banner
    st.info(
        """
        **Mathematical Formulation (Log-Target Space)**:
        
        $$\\log(1 + \\hat{y}) = E[\\log(1+y)] + \\sum_{i=1}^M \\phi_i(x)$$
        
        - **Base Value ($E[\\log(1+y)]$):** **4.3578** (Translates to $\\exp(4.3578) - 1 \\approx$ **₹77.08 Lakhs** baseline market anchor).
        - **$\\phi_i(x)$:** Marginal Shapley additive contribution of feature $i$ in log-price space.
        - **Additivity Verification:** Max numerical discrepancy across all 2,251 test listings is **5.25e-06**, confirming mathematical precision.
        """
    )

    st.warning(
        "⚠️ **Proper Economic Interpretation**: Individual SHAP values are expressed in **log-model margin units**, "
        "not direct linear ₹ additions. Furthermore, SHAP attributions reflect statistical associations within the "
        "historical Bengaluru training corpus and **must not be interpreted as independently proven causal price elasticities**."
    )

    figures = get_shap_figures()

    # Sub-tabs for deep exploration
    shap_tab1, shap_tab2, shap_tab3, shap_tab4 = st.tabs([
        "📊 Local Case Study (Waterfall)",
        "🌐 Global Feature Importance",
        "📈 Non-Linear Feature Interactions",
        "🛡️ Model Governance & Ethics"
    ])

    # TAB 1: LOCAL CASE STUDY
    with shap_tab1:
        st.markdown("### 📌 Local Case Study: Canonical 2 BHK in Hoodi")
        st.markdown(
            "To understand how the 300 gradient boosted trees synthesize property features into a final valuation, "
            "we inspect a representative held-out test property closest to the Bengaluru market median asking price:"
        )

        study = get_canonical_case_study()

        # Listing summary cards
        c_cols = st.columns(4)
        with c_cols[0]:
            st.metric("Property", f"{study['bhk']} BHK, {study['location']}")
        with c_cols[1]:
            st.metric("Built-up Area", f"{study['total_sqft']} sq. ft.")
        with c_cols[2]:
            st.metric("Actual Price", f"₹{study['actual_price_lakhs']:.2f} L")
        with c_cols[3]:
            st.metric(
                "Predicted Valuation",
                f"₹{study['predicted_price_lakhs']:.2f} L",
                delta=f"Diff: ₹{study['absolute_error_lakhs']:.2f} L (0.04%)"
            )

        col_img, col_desc = st.columns([1.4, 1])

        with col_img:
            if figures["waterfall"].exists():
                st.image(str(figures["waterfall"]), caption="Figure 08: SHAP Waterfall Plot for Test Index 3456 (Hoodi)", use_container_width=True)
            else:
                st.error("Waterfall figure not found.")

        with col_desc:
            st.markdown("#### 🔍 Step-by-Step Waterfall Decomposition")
            st.markdown(
                f"""
                1. **Baseline Market Expectation**:
                   - Model begins at expected value $E[f(X)] = 4.3578$ (₹77.08 Lakhs).
                
                2. **Positive Dimension Driver (`total_sqft = 1420.0`)**:
                   - Attributed **+0.21 log units**. The above-average unit size provides strong upward price pressure.
                
                3. **Sanitary Ratio Indicator (`bath = 2.0`)**:
                   - Attributed **+0.02 log units**. Adequate bathroom allocation confirms standard layout viability.
                
                4. **Suburban Locality Adjustment (`location = Hoodi`)**:
                   - Attributed **-0.19 log units**. Suburb discount relative to central high-demand enclaves (Indiranagar, CBD).
                
                5. **Final Output Calculation**:
                   - Output $f(x) = 4.3303$.
                   - Valuation: $\\exp(4.3303) - 1 =$ **₹74.97 Lakhs** (Matches actual asking price ₹75.00 Lakhs within 0.04%!).
                """
            )

    # TAB 2: GLOBAL FEATURE IMPORTANCE
    with shap_tab2:
        st.markdown("### 🌐 Global Feature Importance & Behavioral Dynamics")
        st.markdown(
            "Mean Absolute SHAP values ($\\frac{1}{N} \\sum |\\phi_i|$) reveal which features exerted the highest "
            "overall influence across the entire held-out test distribution ($N = 2,251$):"
        )

        col_g1, col_g2 = st.columns([1.2, 1])

        with col_g1:
            if figures["bar_importance"].exists():
                st.image(str(figures["bar_importance"]), caption="Figure 07: Top 20 Global Features Ranked by Mean Absolute SHAP", use_container_width=True)
            if figures["summary"].exists():
                st.image(str(figures["summary"]), caption="Figure 06: SHAP Summary Beeswarm Plot across Held-out Test Set", use_container_width=True)

        with col_g2:
            st.markdown("#### 🏆 Top 15 Feature Rankings")
            imp_data = get_global_feature_importance()
            df_imp = pd.DataFrame(imp_data)
            st.dataframe(
                df_imp.rename(columns={
                    "rank": "Rank",
                    "feature": "Feature Dimension",
                    "category": "Category",
                    "mean_abs_shap": "Mean |SHAP| (log units)",
                    "direction": "Impact Trend"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                """
                **Key Engineering Takeaways**:
                - **Physical Scale Dominance:** `total_sqft` is by far the single most influential variable (0.3810 log units).
                - **Locality Micro-Markets:** Premium enclaves (`Giri Nagar`, `Rajaji Nagar`, `Indira Nagar`) command huge positive log shifts, while suburban peripheral regions (`Chandapura`, `Anekal`) receive negative attributions.
                - **Layout vs. Partitions:** Bathroom count (`bath`) acts as a stronger luxury signal than bedroom count (`bhk`).
                """
            )

    # TAB 3: NON-LINEAR INTERACTIONS
    with shap_tab3:
        st.markdown("### 📈 Non-Linear Feature Interactions: Square Footage & Bathrooms")
        st.markdown(
            "Traditional linear hedonic models assume constant marginal price per square foot. "
            "SHAP dependence plots demonstrate the actual non-linear learned response curve of the XGBoost ensemble:"
        )

        col_d1, col_d2 = st.columns([1.3, 1])

        with col_d1:
            if figures["dependence"].exists():
                st.image(str(figures["dependence"]), caption="Figure 09: SHAP Dependence of total_sqft colored by bath count", use_container_width=True)

        with col_d2:
            st.markdown(
                """
                #### 💡 Empirical Observations:
                
                1. **Diminishing Marginal Utility**:
                   - For compact to medium units ($500 - 2,500$ sq. ft.), the valuation slope rises steeply.
                   - Beyond $3,500$ sq. ft., the curve flattens noticeably, showing that buyers exhibit diminishing willingness to pay per additional square foot.
                
                2. **Synergistic Luxury Interaction**:
                   - Notice the points at $1,800$ sq. ft.: listings with **3 or 4 bathrooms** (red dots) receive significantly higher positive attributions than those with only **2 bathrooms** (blue dots).
                   - This confirms that spacious square footage requires complementary sanitary ratios to unlock premium valuations.
                """
            )

    # TAB 4: GOVERNANCE & ETHICS
    with shap_tab4:
        st.markdown("### 🛡️ AI Model Governance, Limitations & Ethical Principles")
        st.markdown(
            """
            As an institutional ML system, EstateIQ AI adheres to strict AI governance standards:
            
            1. **Associational vs. Causal Reasoning**:
               SHAP attributions identify statistical associations learned from 11,252 historical listings. They do not constitute causal proof. For example, adding an ornamental balcony to a flat will not automatically increase its resale market value by the SHAP amount without considering structural and aesthetic market demand.
            
            2. **Geographic & Architectural Scope Constraints**:
               The model is strictly calibrated for residential properties in metropolitan Bengaluru within valid physical bounds ($300 - 12,000$ sq ft, $1 - 10$ BHK). Appraisals for commercial complexes, agricultural farmland, or properties outside Bengaluru are out-of-domain and should not be relied upon.
            
            3. **Fairness & Non-Discrimination**:
               The training dataset contains purely structural and geographical variables (`total_sqft`, `bhk`, `bath`, `balcony`, `location`, `area_type`). No demographic, socio-economic, or sensitive human attributes are utilized, preventing algorithmic demographic bias.
            
            4. **Dual Scale Transparency**:
               To eliminate misleading interpretations, predictions are presented in dual units (₹ Lakhs / ₹ Crores and log-margin SHAP units) with empirical confidence margins derived from held-out residual diagnostics.
            """
        )
