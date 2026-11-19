# EstateIQ AI 🏢
### AI-Powered Bengaluru Real Estate Valuation & Investment Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyDeck](https://img.shields.io/badge/PyDeck-0.9+-00D2B4?logo=uber&logoColor=white)](https://deckgl.readthedocs.io/)
[![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-Overpass_API-7EBC6F?logo=openstreetmap&logoColor=white)](https://www.openstreetmap.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Phase 6.4 Production Ready](https://img.shields.io/badge/Status-Phase_6.4:_Production_Ready-success)]()

---

## 1. Executive Summary

**EstateIQ AI** is an institutional-grade machine learning and geospatial intelligence platform engineered to predict residential property valuations and deliver transparent investment analytics across metropolitan **Bengaluru, India** ("Silicon Valley of India").

Rather than relying on black-box heuristics or broker markups, EstateIQ AI synthesizes:
1. **Advanced Gradient Boosting**: Tournament-winning **XGBoost (Log-Target)** pipeline achieving **Test $R^2 = 0.7667$** and **Test MAE = ₹25.13 Lakhs** across 11,252 historical property listings / asking prices.
2. **OpenStreetMap Geospatial Intelligence**: Live geodesic proximity modeling to 481 transit/metro nodes, 130 technology parks, and 1,161 healthcare centers across 234 Bengaluru localities.
3. **Explainable AI (TreeSHAP)**: Provably exact Shapley additive attributions in model log-margin space ($\log(1+y)$), explaining the exact economic drivers behind every valuation.
4. **Interactive Cartographic Dashboard**: A high-performance **Streamlit & PyDeck** web application supporting sub-25ms inference, interactive 2D/3D micro-market maps, preset appraisals, and historical residual-scale estimates.

---

## 2. Champion Model Performance & Tournament Leaderboard

During Phase 5, an exhaustive tournament benchmarked 9 model configurations using **5-Fold Cross-Validation** on the training set ($N = 9,001$) and evaluation on a held-out test partition ($N = 2,251$):

| Rank | Model Configuration | CV MAE (₹L) | CV RMSE (₹L) | CV R² | Test MAE (₹L) | Test RMSE (₹L) | Test R² | Overfit Gap |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost (Log Target)** | **₹24.34 ± 1.51** | ₹59.81 | **0.6952** | **₹25.13** | **₹59.09** | **0.7667** | **0.1133** |
| 🥈 | Random Forest (Raw Price) [Phase 4] | ₹24.70 ± 0.82 | ₹59.48 | 0.6884 | ₹25.94 | ₹71.22 | 0.6611 | 0.2697 |
| 🥉 | LightGBM [Ordinal] (Log Target) | ₹24.71 ± 1.57 | ₹61.17 | 0.6817 | ₹25.11 | ₹60.23 | 0.7576 | 0.1079 |
| 4 | Random Forest (Log Target) | ₹24.93 ± 1.20 | ₹61.08 | 0.6745 | ₹26.44 | ₹73.47 | 0.6393 | 0.2671 |
| 5 | CatBoost (Log Target) | ₹25.43 ± 1.68 | ₹61.85 | 0.6743 | ₹26.88 | ₹63.06 | 0.7343 | 0.0841 |
| 6 | XGBoost (Raw Price) | ₹25.58 ± 1.69 | ₹65.65 | 0.6288 | ₹26.00 | ₹60.70 | 0.7538 | 0.2798 |
| 7 | LightGBM (Log Target) | ₹25.72 ± 1.59 | ₹63.79 | 0.6539 | ₹26.28 | ₹62.17 | 0.7417 | 0.1052 |
| 8 | CatBoost (Raw Price) | ₹26.56 ± 1.49 | ₹60.68 | 0.6846 | ₹27.68 | ₹61.94 | 0.7436 | 0.1275 |
| 9 | LightGBM (Raw Price) | ₹27.45 ± 1.66 | ₹65.59 | 0.6325 | ₹27.97 | ₹65.79 | 0.7107 | 0.1510 |

### Empirical Heteroscedasticity Analysis across Price Brackets (Test Set):
- **Budget (< ₹50 Lakhs):** $N = 617$, Mean Actual ₹38.89L, Residual Std ₹9.13L, **MAE = ₹8.55L**
- **Mid-Market (₹50 - ₹100 Lakhs):** $N = 990$, Mean Actual ₹71.74L, Residual Std ₹18.61L, **MAE = ₹13.24L**
- **Upper-Mid (₹100 - ₹200 Lakhs):** $N = 420$, Mean Actual ₹137.67L, Residual Std ₹42.47L, **MAE = ₹30.75L**
- **Luxury (> ₹200 Lakhs):** $N = 224$, Mean Actual ₹381.81L, Residual Std ₹151.09L, **MAE = ₹112.79L**

---

## 3. Geospatial Intelligence & Amenity Proximity

EstateIQ AI integrates OpenStreetMap (OSM) spatial features to enrich tabular real estate data without data leakage:
- **Localities Resolved:** 211 of 234 recognized localities geocoded via GeoPy Nominatim (90.2% resolution rate).
- **Coordinate Integrity:** Synthetic category `'other'` and 21 unresolved suburbs strictly retain `NaN` coordinates to avoid spatial hallucination.
- **Infrastructure Registry (WGS84 Ellipsoid):**
  - **Transit / Metro Nodes:** 481 nodes (206 subway stations, 178 track stops, 97 entrances; covering operational and Phase 2A/2B/3 corridors).
  - **Technology Parks & IT Corridors:** 130 hubs (ITPL, Manyata, Electronic City, Bellandur ORR, Bagmane).
  - **Hospitals & Healthcare Centers:** 1,161 facilities.
- **PyDeck Cartography:** 2D/3D multi-layer scatterplot rendering property reference centroids (🔴 Red), Metro nodes (🟢 Green), Tech hubs (🔵 Indigo), and Healthcare centers (🟡 Amber).

---

## 4. Explainable AI (SHAP) & Game Theory

To explain tree decisions, EstateIQ AI incorporates **TreeSHAP** (`shap.TreeExplainer`):
- **Additive Formulation:** $\log(1 + \hat{y}) = E[\log(1+y)] + \sum_{i=1}^M \phi_i(x)$
- **Base Value:** $E[\log(1+y)] = 4.3578 \implies \exp(4.3578) - 1 \approx \mathbf{₹77.08\text{ Lakhs}}$
- **Numerical Discrepancy:** Maximum error between raw margin and SHAP sum across all 2,251 test properties is **$\le 8.58 \times 10^{-6}$ log units**, confirming exact additivity.
- **Top Feature Drivers:** `total_sqft` (rank 1, Mean $|\text{SHAP}| = 0.3810$), followed by micro-market premiums (`Giri Nagar`, `Rajaji Nagar`, `Indira Nagar`) and layout signals (`bath` outranking `bhk`).

---

## 5. System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                  Streamlit Web Interface (app.py)                                |
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
|  - Tiered MAE residual scales   |  - PyDeck 2D/3D cartography    |  - Diagnostic figure paths    |
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

## 6. Project Directory Structure

```text
EstateIQ-AI/
├── app.py                            # Main Streamlit application entry point
├── Procfile                          # Deployment process configuration (Render/Heroku)
├── requirements.txt                  # Production package dependencies
├── .gitignore                        # Git exclusion rules
├── .streamlit/
│   └── config.toml                   # Streamlit server and theme configuration
├── src/
│   ├── services/                     # Decoupled business logic & caching layer
│   │   ├── model_service.py          # Cached model inference & residual bands
│   │   ├── geo_service.py            # Geospatial lookups & PyDeck map constructor
│   │   └── shap_service.py           # SHAP metadata & canonical case study data
│   ├── ui/                           # Modular UI views
│   │   ├── valuation_calculator.py   # Valuation form & appraisal KPI cards
│   │   ├── geospatial_explorer.py    # Proximity badges & interactive map
│   │   ├── shap_explainer_view.py    # Game-theoretic XAI view & waterfall
│   │   ├── market_insights.py        # Tournament leaderboard & residual tiers
│   │   └── locality_catalog.py       # Filterable catalog of all 234 localities
│   ├── models/                       # Model training and evaluation routines
│   ├── geospatial/                   # Overpass extraction & geocoding scripts
│   └── data/                         # Cleaning and validation scripts
├── models/
│   ├── estateiq_advanced_model.joblib # Serialized Champion XGBoost Regressor (677 KB)
│   └── model_metadata.json           # Model configuration and performance metadata
├── data/
│   ├── raw/                          # Raw listing data
│   ├── processed/                    # Cleaned housing dataset (bengaluru_cleaned.csv)
│   └── geospatial/                   # Geocoded localities and OSM POI tables
├── reports/
│   ├── figures/                      # High-resolution diagnostic figures (01 - 09)
│   ├── phase_6_4_validation_report.md # Production readiness & DevSecOps audit
│   └── streamlit_application_report.md# UI architecture & system integration report
└── tests/
    ├── test_e2e_validation.py        # End-to-end functional and profiling tests
    ├── test_services.py              # Service layer unit tests
    └── test_geospatial.py            # Geospatial queries and distance tests
```

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10 – 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/EstateIQ-AI.git
cd EstateIQ-AI
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production dependencies (for application & inference)
python -m pip install -r requirements.txt

# Optional development & notebook dependencies (Jupyter, ipykernel)
python -m pip install -r requirements-dev.txt
```

### 4. Run Automated Regression Test Suite
```bash
python -m unittest discover tests
```
*(All 27 unit and end-to-end tests should pass with `OK`).*

### 5. Launch the Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 8. Deployment Guide

### Recommended Deployment Target: Streamlit Community Cloud
EstateIQ AI is natively structured for zero-friction deployment on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**.

The pre-configured `.streamlit/config.toml` automatically activates headless server settings, light theme, and telemetry suppression.

### Alternative Target: Render / Railway / Heroku
A [`Procfile`](Procfile) is provided in the repository root:
```text
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## 9. Academic & Portfolio Context

- **Developer**: Second-Year CSE (AI & Machine Learning) Student.
- **Domain Focus**: Production Tabular ML, Geospatial Data Engineering, Explainable AI (XAI), DevSecOps.
- **License**: MIT License. Real estate appraisals reflect statistical models of historical property listings / asking prices and should not replace certified professional appraisals.
