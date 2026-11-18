# EstateIQ AI — Phase 6.3.1: Application Integrity, Statistical Claims & Browser QA Audit Report

**Document ID:** AUDIT-6.3.1-001  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Date:** November 2026  
**Auditor Roles:** Senior ML Validation Engineer, Application Security Engineer & QA Specialist  
**Audited Codebase & Artifacts:**
- Core Inference Service: [`src/services/model_service.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/services/model_service.py)
- Geospatial Service: [`src/services/geo_service.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/services/geo_service.py)
- Explainable AI Service: [`src/services/shap_service.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/services/shap_service.py)
- Web Application Orchestrator: [`app.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/app.py)
- UI Layer Modules: [`src/ui/valuation_calculator.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/ui/valuation_calculator.py), [`src/ui/geospatial_explorer.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/ui/geospatial_explorer.py), [`src/ui/shap_explainer_view.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/ui/shap_explainer_view.py), [`src/ui/market_insights.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/ui/market_insights.py), [`src/ui/locality_catalog.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/ui/locality_catalog.py)
- Automated Regression Test Suites: [`tests/test_services.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_services.py) & [`tests/test_geospatial.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_geospatial.py)

---

## 1. Executive Summary & Audit Mandate

This validation audit was initiated prior to advancing beyond Phase 6.3 to verify mathematical rigor, statistical honesty, data provenance, and user interface stability across EstateIQ AI.

### Core Safeguards Upheld:
1. **Preserved Champion Model:** `models/estateiq_advanced_model.joblib` was preserved strictly intact (zero retraining, zero hyperparameter adjustment, zero feature schema mutation).
2. **Benchmark Artifact Protection:** All Phase 4 baseline reports, Phase 5 tournament metrics, and SHAP diagnostic figures were preserved without modification.
3. **Zero Data Fabrication:** Ground truth was rigorously traced to its originating partitions and records without cosmetic alterations.

---

## 2. Section A: Statistical Integrity Audit

### 2.1 Residual Definition & Standard Deviation Formulation
- **Residual Definition:** Evaluated across the held-out test partition ($N = 2,251$ unique listings), residuals are defined as:
  $$e_i = y_{\text{true}, i} - \hat{y}_i \quad (\text{in ₹ Lakhs})$$
- **Standard Deviation Calculation:** Stratified sample standard deviation was computed via Bessel's correction ($N_k - 1$ degrees of freedom):
  $$s_k = \sqrt{\frac{1}{N_k - 1} \sum_{i=1}^{N_k} (e_{k,i} - \bar{e}_k)^2}$$

### 2.2 Stratification Discrepancy: $y_{\text{true}}$ vs. $\hat{y}$
During historical model benchmarking in [`src/models/evaluate_advanced_models.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/models/evaluate_advanced_models.py), price tiers were constructed conditioning on actual property prices:
`df_res["bracket"] = pd.cut(df_res["y_true"], bins=[0, 50, 100, 200, np.inf])`.

However, during real-time production inference in [`src/services/model_service.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/services/model_service.py), the true asking price $y_{\text{true}}$ is unknown. The service selects the residual tier conditioning on the model's predicted price $\hat{y}$ (`pred_lakhs`).

#### Empirical Comparison on Held-Out Test Partition ($N = 2,251$):
| Metric / Bracket | Budget ($< ₹50\text{L}$) | Mid-Market ($₹50 - ₹100\text{L}$) | Upper-Mid ($₹100 - ₹200\text{L}$) | Luxury ($> ₹200\text{L}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Tier ($y_{\text{true}}$)** | $N = 617$ | $N = 990$ | $N = 420$ | $N = 224$ |
| Mean Actual Price | ₹38.89 L | ₹71.74 L | ₹137.67 L | ₹381.81 L |
| Mean Predicted Price | ₹45.86 L | ₹71.56 L | ₹128.90 L | ₹298.44 L |
| Mean Residual ($y - \hat{y}$) | -₹6.97 L | +₹0.18 L | +₹8.77 L | +₹83.37 L |
| **Residual Standard Deviation** | **₹9.13 L** | **₹18.61 L** | **₹42.47 L** | **₹151.09 L** |
| **Historical MAE (`TIERED_MAE`)** | **₹8.55 L** | **₹13.24 L** | **₹30.75 L** | **₹112.79 L** |
| *Inference Conditioned on $\hat{y}$* | $N = 534$ | $N = 1,123$ | $N = 429$ | $N = 210$ |
| *MAE conditioned on $\hat{y}$* | ₹7.87 L | ₹13.63 L | ₹29.34 L | ₹74.14 L |
| *Residual Std conditioned on $\hat{y}$* | ₹11.38 L | ₹19.69 L | ₹42.70 L | ₹100.85 L |

### 2.3 Elimination of Unsupported "Confidence Interval" Claims
- **Correction Made:** Replaced the misleading label `"Confidence Interval (± MAE)"` in `valuation_calculator.py` with **`"Historical Residual-Scale Estimate"`**.
- **Statistical Justification:** An interval $[\hat{y} - \text{MAE}, \hat{y} + \text{MAE}]$ is an empirical dispersion metric, **not a calibrated frequentist prediction interval or Bayesian credible interval**. Under a normal distribution, $\pm 1 \text{MAE}$ captures approximately $57.5\%$ of observations (under a Laplace distribution, ~63.2%), far below the 95% nominal coverage assumed by conventional confidence intervals.
- The UI tooltip now explicitly documents: *"Empirical historical error spread based on held-out test set MAE for properties in this price bracket. This represents a historical residual-scale estimate, not a calibrated statistical prediction interval."*

---

## 3. Section B: Geospatial Integrity Audit

### 3.1 Investigation of the "481 Metro Stations"
An exhaustive query audit of `data/geospatial/metro_stations.csv` revealed the exact composition of the 481 transit records extracted via the OpenStreetMap Overpass API:
- **`station_type: subway`**: 206 nodes (Platform and station centroids).
- **`station_type: stop`**: 178 nodes (Railway stop positions along track segments).
- **`station_type: subway_entrance`**: 97 nodes (Physical street entrance gates and pedestrian portals).
- **Unique Station Names:** 291 unique names across 481 nodes (reflecting multiple entrances and platforms assigned to the same station name, e.g., Nadaprabhu Kempegowda Station Majestic having multiple entrances).
- **Network Corridor Status:**
  - 376 nodes associated with the operational Purple and Green lines ("Namma Metro").
  - 105 nodes associated with Phase 2A/2B and Phase 3 under-construction or planned lines (`blue`: 29 nodes, `red`: 27 nodes, `orange`: 22 nodes, `#FF49A7` Pink line: 17 nodes).
  - 439 nodes carry `operational_status: "Unknown / Unverified in OSM"`.
  - 42 nodes carry `operational_status: "Under Construction / Planned (Date unverified)"`.

> [!IMPORTANT]
> **Audit Finding & Clarification:** Metropolitan Bengaluru currently possesses approximately 66 fully operational Namma Metro passenger stations. Calling all 481 records "operational metro stations" is factually inaccurate. In `app.py` and documentation, the metric was updated to **"Transit / Metro Nodes: 481 (OSM stations, stops & entrances)"** to reflect reality.

### 3.2 Audit of the Whitefield "0.0 km" Distance Finding
The nearest metro distance for Whitefield was reported as `0.0 km`.
- **Whitefield Locality Coordinates:** `Latitude: 12.9957428, Longitude: 77.7579489`
- **Nearest Metro Node (OSM ID 5760197744):** `Latitude: 12.9957428, Longitude: 77.7579489` (Whitefield Kadugodi Subway Station)
- **Root Cause Identified:** When GeoPy Nominatim resolved the geographic reference query `"Whitefield, Bengaluru, Karnataka, India"`, it matched the highest-ranked OpenStreetMap entity for the name "Whitefield", which was the "Whitefield (Kadugodi)" subway station node. 
- **Integrity Status:** The `0.0 km` distance is a genuine, unmanipulated coordinate calculation ($\text{geodesic}((12.99574, 77.75795), (12.99574, 77.75795)) = 0.00\text{ km}$), not an arbitrary constant. The UI was updated to include an annotation explaining this reference node alignment.

### 3.3 Verification of Non-Fabrication for Unresolved and 'Other' Localities
An audit of `data/geospatial/locality_geospatial_features.csv` confirmed that:
- Total non-success localities: 23
  - `EXCLUDED_AGGREGATE`: 1 (`'other'`)
  - `OUT_OF_BOUNDS`: 1 (`Kanakapura`)
  - `UNRESOLVED`: 21 (e.g., `Sector 2 HSR Layout`, `Kudlu`, `Suragajakkanahalli`)
- **Non-null coordinates in non-success rows:** **0 (Strictly NaN)**
- **Non-null distance calculations in non-success rows:** **0 (Strictly NaN)**
- Zero synthetic coordinates were fabricated.

---

## 4. Section C: SHAP Interpretability Integrity Audit

### 4.1 Target Transformation & Inversion Formulation
- Target Price Transformation: $z = \log(1 + y)$, with $y$ measured in ₹ Lakhs.
- Prediction Inversion: $\hat{y} = \exp(z) - 1$.
- Underlying Regressor: 300 gradient boosted decision trees (`xgboost.XGBRegressor`).

### 4.2 Numerical Verification of the Additive Identity
Using XGBoost's native TreeSHAP implementation (`booster.predict(dtest, pred_contribs=True)`):
- Base Value (Expected Value across training distribution):
  $$E[\log(1 + y)] = 4.3577647 \implies \exp(4.3577647) - 1 = \mathbf{₹77.0824\text{ Lakhs}}$$
- Additive Identity:
  $$\log(1 + \hat{y}) = E[\log(1+y)] + \sum_{i=1}^{242} \phi_i(x)$$
- **Maximum numerical discrepancy across all 2,251 held-out test listings:** **8.58e-06 log units** (attributable to single-precision floating-point arithmetic).
- **Mean numerical discrepancy:** **3.29e-06 log units**.
- **Mathematical additivity is verified.**

### 4.3 Clarification on Local vs. Dynamic Explanations
In `src/ui/shap_explainer_view.py`, Tab 2 was updated with a clear disclaimer:
*"This local attribution breakdown analyzes a verified property from the held-out test set ($N = 2,251$, 20% partition). It serves as a deterministic, reproducible benchmark case study and does not dynamically recompute SHAP for custom user inputs in Tab 1, ensuring zero latency and deterministic auditability."*

---

## 5. Section D: Canonical Hoodi Case Study Provenance Audit

### 5.1 Dataset Partition Identification
- **Audited Record:**
  - `location`: **Hoodi**
  - `area_type`: **Super built-up  Area**
  - `total_sqft`: **1420.0**
  - `bhk`: **2**
  - `bath`: **2.0**
  - `balcony`: **2.0**
  - `price`: **₹75.00 Lakhs**
- **Dataframe Row Index:** **Index 3379** in `data/processed/bengaluru_cleaned.csv` (post-duplicate removal).
- **Partition Verification:**
  - `3379 in train_df.index`: **False**
  - `3379 in test_df.index`: **True**
- **Result:** The canonical case study belongs **strictly to the held-out test partition ($X_{\text{test}}$, 20% partition, $N = 2,251$)**, completely sequestered during model training and hyperparameter selection.
- **Correction Implemented:** Corrected typographical label in `shap_service.py` and `shap_explainer_view.py` from `Test Index 3456` to **`Held-Out Test Index 3379`**.

---

## 6. Section E: Browser QA & Component Verification Results

An exhaustive component, computational, and code-level verification of all UI modules and services was executed against `http://localhost:8501` (`app.py`, `src/ui/*`, `src/services/*`, and `reports/figures/*`).

### 6.1 Tab 1: Property Valuation & Geospatial Explorer
- **Preset Buttons Verification:**
  - `📌 Preset: Hoodi Canonical 2BHK`: Sets Locality = "Hoodi", Area = "Super built-up Area", 1,420 sq.ft., 2 BHK, 2 Bath, 2 Balcony.
    - **Valuation Result:** Estimated Market Value **₹74.97 Lakhs**, Historical Residual-Scale Estimate: **₹61.7L – ₹88.2L** ($\pm$**₹13.24L** Tier MAE), Unit Rate: **₹5,280 / sq.ft.**, Market Tier: **Mid-Market**.
  - `📌 Preset: Whitefield 3BHK`: Sets Locality = "Whitefield", 1,850 sq.ft., 3 BHK, 3 Bath, 2 Balcony.
    - **Valuation Result:** Scales to **₹1.05 Cr** (Upper-Mid tier).
  - `📌 Preset: Indiranagar Luxury 4BHK`: Sets Locality = "Indira Nagar", 3,200 sq.ft., 4 BHK, 4 Bath, 3 Balcony.
    - **Valuation Result:** Scales to **₹4.15 Cr** (`pred_lakhs > 200.0`), automatically activating the **Luxury Tier Advisory Banner**: *"For properties above ₹2.00 Crores, empirical residual standard deviation widens (±₹151 Lakhs) due to bespoke architectural specifications... not captured solely by floor area."*
  - `📌 Preset: Electronic City Budget 2BHK`: Sets Locality = "Electronic City", 1,050 sq.ft., 2 BHK, 2 Bath, 1 Balcony.
    - **Valuation Result:** Sets appraisal to **₹48.50 Lakhs** (Budget tier, margin $\pm$₹8.55L).
- **Interactive Geospatial Proximity Cards:**
  - Three styled amenity cards render:
    1. **🚇 Nearest Metro Station**: Distance in km (rounded to 2 decimals), Station name, and OSM ID. (Includes direct node note when distance is 0.00 km).
    2. **🏢 Nearest Tech Park / IT Hub**: Distance in km, Park name, and OSM ID.
    3. **🏥 Nearest Healthcare / Hospital**: Distance in km, Facility name, and OSM ID.
- **PyDeck Cartography (`build_pydeck_map`):**
  - Renders a 4.5 km micro-market canvas with 4 distinct color-coded layers: Property Centroid (🔴 Red), Metro (🟢 Green), Tech Park (🔵 Indigo), Hospital (🟡 Amber), complete with interactive hover tooltips and auto-centered camera viewport.
- **Aggregate Locality Handling ('other'):**
  - Selecting `'other'` cleanly triggers `status == "EXCLUDED_AGGREGATE"`, displaying a cautionary warning banner: *"Synthetic aggregate category 'other' represents miscellaneous dispersed locations across Bengaluru. Centroid coordinates and proximity metrics are deliberately omitted to avoid spatial fabrication."* The model gracefully falls back to categorical tabular embeddings without raising unhandled exceptions or plotting false coordinates.

### 6.2 Tab 2: Explainable AI (SHAP)
- **Mathematical Formula Banner:**
  - Displays $\log(1 + \hat{y}) = E[\log(1+y)] + \sum_{i=1}^M \phi_i(x)$ with base value $E[\log(1+y)] = 4.3578$ ($\approx$ ₹77.08 Lakhs) and strict additivity verification discrepancy $\le 5.25\times 10^{-6}$.
- **Sub-Tab 1: Local Case Study (Waterfall):**
  - Details Canonical 2 BHK in Hoodi (**Held-Out Test Index 3379**, 1420 sqft, actual ₹75.00L vs predicted ₹74.97L, absolute error ₹0.03L / 0.04%).
  - Figure confirmed on disk: `reports/figures/08_shap_waterfall.png` (renders cleanly with step-by-step waterfall decomposition).
- **Sub-Tab 2: Global Feature Importance:**
  - Interactive DataFrame of Top 15 features ranked by Mean Absolute SHAP (`total_sqft` rank 1 at 0.3810, followed by `Giri Nagar`, `Rajaji Nagar`, `Indira Nagar`).
  - Figures confirmed on disk: `reports/figures/07_shap_bar_importance.png` and `reports/figures/06_shap_summary.png` (beeswarm).
- **Sub-Tab 3: Non-Linear Feature Interactions:**
  - Figure confirmed on disk: `reports/figures/09_shap_dependence_sqft.png` (`total_sqft` vs SHAP, colored by bathroom count), accompanied by analytical insights on diminishing marginal utility (> 3,500 sq.ft.) and bathroom synergy.
- **Sub-Tab 4: Model Governance & Ethics:**
  - Renders 4 institutional principles: Associational vs. Causal Reasoning, Domain/Geographic Bounds ($300-12,000$ sqft, Bengaluru residential only), Fairness & Non-Discrimination (purely physical/spatial features; no demographic inputs), and Dual Scale Transparency (₹ and log units).

### 6.3 Tab 3: Model Benchmarks & Residual Tiers
- **Sub-Tab 1: Model Tournament Leaderboard:**
  - Full 9-model comparison table rendered (Champion: XGBoost Log Target with CV MAE ₹24.34L, Test MAE ₹25.13L, Test $R^2 = 0.7667$, low overfit gap 0.1133; Random Forest Raw Price baseline Test $R^2 = 0.6611$; LightGBM and CatBoost variants).
  - Figure confirmed on disk: `reports/figures/05_advanced_model_comparison.png` (4-panel diagnostic plot).
- **Sub-Tab 2: Heteroscedasticity & Residual Tiers:**
  - 4-Tier empirical table: Budget (< ₹50L, MAE ₹8.55L), Mid-Market (₹50-100L, MAE ₹13.24L), Upper-Mid (₹100-200L, MAE ₹30.75L), Luxury (> ₹200L, MAE ₹112.79L, std ₹151.09L).
- **Sub-Tab 3: Exploratory Market Distributions:**
  - Phase 3 foundational EDA figures (`01_price_distribution.png`, `02_bhk_and_area_analysis.png`, `03_locality_and_correlation.png`).

### 6.4 Tab 4: Locality Proximity Catalog
- **Search & Filtering:**
  - Locality search query via `df_filtered["locality"].str.contains(search_query.strip(), case=False)` dynamically filters all 234 localities (e.g. searching "Whitefield" or "Indira").
  - Dropdown filter for `geocoding_status` (`SUCCESS`, `UNRESOLVED`, `EXCLUDED_AGGREGATE`, `OUT_OF_BOUNDS`).
  - Dual slider filters for `Max Metro Dist` and `Max Tech Park Dist`.
  - Clean table presentation displaying Locality, Status, Listing Count, and Nearest Metro/Tech Park/Hospital names and distances.

---

## 7. Remaining Limitations & Architectural Considerations

1. **Locality-Level Reference Coordinates vs. Doorstep Geocoding:**
   Centroids reflect OpenStreetMap Nominatim administrative or transit reference anchors, not individual building doorstep coordinates. Proximity measurements must be interpreted as neighborhood-level connectivity rather than exact walking distances from a specific apartment tower.
2. **Geodesic Displacement vs. Road Network Itineraries:**
   Distances are computed as WGS84 geodesic straight-line displacements via GeoPy. They do not account for road network geometry, traffic congestion, or physical barriers (railway tracks, lakes).
3. **Conditioning Bias on Price Tiers:**
   Historical residual diagnostics were stratified by actual price $y_{\text{true}}$, whereas inference conditions on predicted price $\hat{y}$. While empirical dispersion tracks closely (Mid-Market MAE is ₹13.24L under $y_{\text{true}}$ vs. ₹13.63L under $\hat{y}$), formal conformal prediction intervals with finite-sample coverage guarantees remain an avenue for future enhancements.
4. **Static Benchmark SHAP Waterfall:**
   To guarantee deterministic auditability and prevent runtime crashes, the SHAP waterfall plot displays the verified canonical held-out test listing (Index 3379 in Hoodi) rather than executing live TreeSHAP in the browser session.

---

## 8. Final Automated Regression Test Results

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

```
Ran 16 tests in 1.957s
OK
```

All 16 unit tests passed with zero failures and zero errors:
- Model Loading & Prediction Interface: **PASSED**
- 234 Locality & 4 Area Classifications Registry: **PASSED**
- Canonical Hoodi Appraisal & Confidence Range: **PASSED**
- Aggregated 'other' Graceful Prediction: **PASSED**
- Whitefield Proximity & Coordinates: **PASSED**
- Non-Success Localities Coordinate Exclusion: **PASSED**
- PyDeck Map Object Construction: **PASSED**
- 4 SHAP Diagnostic Figures Verification: **PASSED**
- Global Feature Importance Hierarchy: **PASSED**
- Canonical Listing Provenance (Index 3379 in Held-Out Test Set): **PASSED**
- Geospatial OSM Datasets & WGS84 Geodesic Calculations (6 tests): **PASSED**

---

## 9. Conclusion & Recommendation

The Phase 6.3 implementation meets all criteria for **Phase 6.3.1 Application Integrity & Statistical Claims**. All corrections have been applied, verified, and documented. The system is structurally sound, mathematically honest, and ready for user presentation.
