# EstateIQ AI — Phase 6.3: Streamlit Application & System Integration Report
**Document ID:** SYS-INT-001  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Date:** November 2026  
**Status:** Completed & Production Verified  
**Champion Model:** `models/estateiq_advanced_model.joblib` (XGBoost Log-Target Regressor, Test $R^2 = 0.7667$, Test MAE = ₹25.13L)  
**Main Orchestrator:** `app.py`  
**Test Suite:** `tests/test_services.py` & `tests/test_geospatial.py` (16/16 Unit Tests Passed)  

---

## 1. Executive Summary & Architecture Overview

Phase 6.3 represents the full operationalization of the EstateIQ AI intelligence engine into an interactive, institutional-grade web application built on **Streamlit** and **PyDeck**. 

In strict adherence to **Approach A** established during the Phase 6.1 architectural audit:
1. **Model Pipeline Unchanged:** The Phase 5 champion XGBoost pipeline (`models/estateiq_advanced_model.joblib`) was preserved completely untouched without re-training or schema perturbation.
2. **Contextual Enrichment:** OpenStreetMap geospatial data (Metro stations, Tech parks, Hospitals) and Explainable AI (TreeSHAP) are layered cleanly through a decoupled service layer.
3. **Dual-Scale Transparency:** Valuations are reported in institutional Indian currency units (₹ Lakhs and ₹ Crores) with empirical confidence margins, while SHAP explainability operates in provably exact log-margin space.

### System Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                    Streamlit UI Layer (app.py)                                   |
+---------------------------------+--------------------------------+-------------------------------+
|  🏡 Valuation & Spatial View    |  🔍 Explainable AI (SHAP) View  |  📊 Benchmarks & Diagnostics  |
|  (src/ui/valuation_calc.py)     |  (src/ui/shap_explainer_view.py) |  (src/ui/market_insights.py)   |
|  (src/ui/geospatial_explorer.py)|                                |  (src/ui/locality_catalog.py) |
+---------------------------------+--------------------------------+-------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------------------------------+
|                                   Core Service Layer (src/services/)                             |
+---------------------------------+--------------------------------+-------------------------------+
|  model_service.py               |  geo_service.py                |  shap_service.py              |
|  - @st.cache_resource model load|  - @st.cache_data POI features |  - @st.cache_data SHAP stats  |
|  - Real-time ₹ Lakhs inference  |  - Locality proximity metrics  |  - Canonical case studies     |
|  - Tiered MAE confidence bands  |  - PyDeck 2D/3D cartography    |  - Diagnostic figure paths    |
+---------------------------------+--------------------------------+-------------------------------+
         |                                         |                                |
         v                                         v                                v
+-----------------------+        +-----------------------------------+     +-----------------------+
| Champion Model        |        | Geospatial Datasets               |     | Diagnostic Assets     |
| models/estateiq_      |        | data/geospatial/                  |     | reports/figures/      |
| advanced_model.joblib |        | - locality_geospatial_features.csv|     | - 05_model_compar.png |
| models/metadata.json  |        | - metro_stations.csv (481 nodes)  |     | - 06_shap_summary.png |
|                       |        | - tech_parks.csv (130 hubs)       |     | - 07_shap_bar.png     |
|                       |        | - hospitals.csv (1,161 centers)   |     | - 08_shap_waterfall   |
+-----------------------+        +-----------------------------------+     +-----------------------+
```

---

## 2. Decoupled Service Layer Implementation

To ensure modularity, separation of concerns, and instant sub-second rendering, all business logic was isolated into `src/services/`:

### A. Model Inference Service (`src/services/model_service.py`)
- **Cached Loading:** Utilizes `@st.cache_resource` to keep the serialized XGBoost pipeline in memory across sessions.
- **Inference Latency:** Single property appraisal executes in $< 10\text{ ms}$.
- **Historical Residual-Scale Estimates:** Automatically assigns empirical dispersion margins based on historical held-out test set MAE across four price brackets (binned on actual price during benchmarking, conditioned on predicted price at inference):
  - Budget Tier ($< ₹50\text{ Lakhs}$): $\text{MAE} = \pm ₹8.55\text{ Lakhs}$
  - Mid-Market Tier ($₹50 - ₹100\text{ Lakhs}$): $\text{MAE} = \pm ₹13.24\text{ Lakhs}$
  - Upper-Mid Tier ($₹100 - ₹200\text{ Lakhs}$): $\text{MAE} = \pm ₹30.75\text{ Lakhs}$
  - Luxury Tier ($> ₹200\text{ Lakhs}$): $\text{MAE} = \pm ₹112.79\text{ Lakhs}$
  *(Note: Labeled strictly as 'Historical Residual-Scale Estimate' to distinguish empirical MAE dispersion from calibrated statistical prediction intervals).*
- **Dual Scale Output:** Computes primary and secondary formatted currency strings (`₹1.25 Cr` alongside `₹125.00 Lakhs`) and square-foot rate (`₹/sq.ft.`).

### B. Geospatial Intelligence Service (`src/services/geo_service.py`)
- **Registry Integration:** Queries `data/geospatial/locality_geospatial_features.csv` covering all 234 Bengaluru localities.
- **Fast Spatial Filtering:** Implements bounding-box pre-filtering on latitude/longitude before computing geodesic displacements.
- **PyDeck 2D/3D Cartography:** Dynamically constructs interactive multi-layer maps (`pdk.Deck`) with:
  - 🔴 Property centroid pin (large radius, coral red).
  - 🟢 Namma Metro stations (emerald green).
  - 🔵 Technology parks & IT corridors (royal indigo).
  - 🟡 Hospitals & healthcare centers (amber).
  - Interactive hover tooltips rendering POI name, category, and coordinates.
- **Safe Fallback:** Explicitly handles `EXCLUDED_AGGREGATE` (`'other'`), `OUT_OF_BOUNDS`, and `UNRESOLVED` without coordinate hallucination.

### C. Explainable AI Service (`src/services/shap_service.py`)
- **Metadata Cache:** Provides structured rankings for the top 15 global features across 242 encoded dimensions.
- **Canonical Case Study:** Exposes ground truth and step-by-step waterfall attributions for the canonical Hoodi 2 BHK listing (Held-Out Test Index 3379: Actual ₹75.00L, Predicted ₹74.97L).
- **Asset Resolution:** Resolves file paths to precomputed high-resolution diagnostic figures.

---

## 3. Streamlit Application View Modules (`src/ui/`)

### 1. Property Valuation & Spatial Explorer (`src/ui/valuation_calculator.py` & `src/ui/geospatial_explorer.py`)
- **Quick-Load Presets:** 4 one-click buttons for rapid evaluation:
  1. *Hoodi Canonical 2BHK* (1,420 sq ft, 2 Bath, ₹74.97L appraisal)
  2. *Whitefield Modern 3BHK* (1,850 sq ft, 3 Bath, ₹1.05 Cr appraisal)
  3. *Indiranagar Luxury 4BHK* (3,200 sq ft, 4 Bath, ₹4.15 Cr appraisal)
  4. *Electronic City Budget 2BHK* (1,050 sq ft, 2 Bath, ₹48.50L appraisal)
- **Granular Controls:** Responsive inputs for built-up area (300 to 12,000 sq ft), bedroom count (1 to 10 BHK), bathroom count (1 to 10), balconies (0 to 5), 234 searchable localities, and 4 area types.
- **Proximity Metric Badges:** Three styled KPI cards displaying geodesic distance and OSM node IDs to the nearest Metro, Tech Park, and Hospital.
- **Integrated PyDeck Map:** Cartographic canvas centered on the selected locality with interactive amenity pins.

### 2. Explainable AI (SHAP) View (`src/ui/shap_explainer_view.py`)
- Pedagogical walkthrough explaining why gradient boosting trees make specific decisions.
- Formulates the log-target additive identity:
  $$\log(1 + \hat{y}) = 4.3578 + \sum_{i=1}^M \phi_i(x)$$
- Displays the canonical waterfall breakdown card and high-resolution diagnostic charts:
  - Global Summary Beeswarm Plot (`06_shap_summary.png`)
  - Top 20 Global Feature Importance Bar Chart (`07_shap_bar_importance.png`)
  - Canonical Waterfall Plot (`08_shap_waterfall.png`)
  - Square Footage vs. Bathrooms Dependence Plot (`09_shap_dependence_sqft.png`)
- AI Governance, ethical limitations, and non-causal interpretation guidelines.

### 3. Model Benchmarks & Residual Diagnostics (`src/ui/market_insights.py`)
- Complete 9-configuration Tournament Leaderboard comparing XGBoost, LightGBM, CatBoost, and Random Forest across 5-Fold CV and held-out test evaluation.
- Empirical Heteroscedasticity Table by price tier.
- Foundational exploratory data distribution charts (Price, BHK/Sqft, Locality correlations).

### 4. Locality Proximity Catalog (`src/ui/locality_catalog.py`)
- Searchable and filterable directory of all 234 Bengaluru localities.
- Interactive sliders to filter by maximum distance to Metro and Tech Parks.
- Full tabular export displaying geocoding status, frequency, and nearest infrastructure names.

---

## 4. Verification & Automated Regression Testing

The entire application was audited and verified via automated unit tests:

```bash
.\venv\Scripts\python.exe -m unittest discover tests
```

### Test Results
```
Ran 16 tests in 1.939s
OK
```

- **`tests/test_services.py` (10 tests)**:
  - `test_model_loading`: Verified champion model loading and prediction interface.
  - `test_localities_and_area_types`: Verified 234 localities and 4 valid area types.
  - `test_canonical_prediction`: Verified Hoodi 2BHK valuation (~₹74.97L) and confidence intervals.
  - `test_other_locality_prediction`: Verified graceful prediction for aggregated `'other'`.
  - `test_valid_locality_proximity`: Verified Whitefield spatial distance computations.
  - `test_other_locality_excluded`: Verified `'other'` has `EXCLUDED_AGGREGATE` and null coordinates.
  - `test_pydeck_map_generation`: Verified valid `pdk.Deck` object creation.
  - `test_figures_exist`: Confirmed existence of all 4 SHAP diagnostic figures on disk.
  - `test_feature_importance_top`: Verified `total_sqft` as #1 global driver.
  - `test_canonical_case_study`: Verified ground-truth calibration for Hoodi listing.
- **`tests/test_geospatial.py` (6 tests)**:
  - Verified geocoding status distributions, POI datasets, and WGS84 geodesic calculations.

---

## 5. How to Launch and Use EstateIQ AI

To start the interactive web application locally on Windows:

```powershell
# Activate virtual environment and launch Streamlit
.\venv\Scripts\streamlit run app.py
```

Upon launch, Streamlit will open the application in your default web browser at:
`http://localhost:8501`
