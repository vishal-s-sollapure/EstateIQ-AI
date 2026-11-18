# EstateIQ AI — Phase 6.4: End-to-End Application Validation & Deployment Readiness Report

**Document ID:** VAL-6.4-001  
**Project:** EstateIQ AI — Bengaluru Real Estate Valuation & Investment Intelligence Platform  
**Date:** November 2026  
**Auditor Roles:** Senior ML Platform Engineer, DevSecOps Engineer, QA Automation Engineer & Production Readiness Reviewer  
**Champion Model:** `models/estateiq_advanced_model.joblib` (XGBoost Log-Target Regressor, Test $R^2 = 0.7667$, Test MAE = ₹25.13L)  
**Main Orchestrator:** [`app.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/app.py)  
**Configuration Artifacts:** [`.streamlit/config.toml`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/.streamlit/config.toml), [`Procfile`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/Procfile), [`requirements.txt`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/requirements.txt), [`.gitignore`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/.gitignore)  
**Test Suite Status:** **27 of 27 Automated Tests Passed (100% Pass Rate in 4.37s)**

---

## 1. Repository & Architecture Findings

### 1.1 Architecture & Component Mapping
EstateIQ AI follows a decoupled three-tier production architecture:

```
+--------------------------------------------------------------------------------------------------+
|                                    Presentation Layer (Streamlit)                                |
|  app.py: Main multi-tab dashboard, page config, global layout, responsive sidebar                |
+---------------------------------+--------------------------------+-------------------------------+
|  src/ui/valuation_calculator.py |  src/ui/shap_explainer_view.py |  src/ui/market_insights.py    |
|  - 4 Property presets           |  - Cooperative game theory     |  - 9-model tournament table   |
|  - Granular specification form  |  - Log-margin additive math    |  - 4-tier residual diagnostics|
|  - Real-time appraisal KPI cards|  - Held-out case study 3379    |  - Exploratory EDA charts     |
|                                 |  - Global beeswarm & bar chart |                               |
+---------------------------------+--------------------------------+-------------------------------+
|  src/ui/geospatial_explorer.py  |  src/ui/locality_catalog.py    |                               |
|  - Proximity metrics (km, OSM)  |  - 234-locality search table   |                               |
|  - Interactive 2D/3D PyDeck map |  - Status & distance sliders   |                               |
+---------------------------------+--------------------------------+-------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------------------------------+
|                                    Service Layer (src/services/)                                 |
+---------------------------------+--------------------------------+-------------------------------+
|  model_service.py               |  geo_service.py                |  shap_service.py              |
|  - @st.cache_resource model load|  - @st.cache_data POI features |  - @st.cache_data SHAP stats  |
|  - XGBoost inference in ₹ Lakhs |  - Locality proximity metrics  |  - Canonical case studies     |
|  - Tiered MAE residual scales   |  - PyDeck scatterplot layers   |  - Precomputed figure paths   |
+---------------------------------+--------------------------------+-------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------------------------------+
|                                       Persistence & Data Layer                                   |
+---------------------------------+--------------------------------+-------------------------------+
|  models/                        |  data/geospatial/              |  reports/figures/             |
|  - estateiq_advanced_model.joblib| - locality_geospatial_features  |  - 01_price_distribution.png  |
|  - model_metadata.json          |  - metro_stations.csv (481)    |  - 05_model_comparison.png    |
|                                 |  - tech_parks.csv (130)        |  - 06_shap_summary.png        |
|                                 |  - hospitals.csv (1,161)       |  - 08_shap_waterfall.png      |
+---------------------------------+--------------------------------+-------------------------------+
```

### 1.2 Inspection of Tracked Artifacts & Working Tree
1. **Model Persistence**:
   - `models/estateiq_advanced_model.joblib` (677 KB): Serialized Scikit-Learn `TransformedTargetRegressor` wrapping `ColumnTransformer` (median imputation + sparse one-hot encoding on 242 dimensions) and `XGBRegressor` (300 estimators, depth=6).
   - `models/model_metadata.json` (1.9 KB): Model tournament configuration and historical validation metrics.
   - **Remediation**: The repository's generic `.gitignore` previously contained `models/*` and `*.joblib`, which would have caused deployments on fresh clones (e.g., Streamlit Community Cloud) to fail with `FileNotFoundError`. The `.gitignore` was updated with explicit whitelist rules (`!models/estateiq_advanced_model.joblib` and `!models/model_metadata.json`). Large experimental baselines (e.g., 55MB Random Forest) remain properly ignored.
2. **Geospatial Assets**:
   - All 5 geospatial CSV files and Overpass cache JSON files in `data/geospatial/` are tracked and intact.
3. **Visual Assets**:
   - All 9 high-resolution diagnostic figures (`01_price_distribution.png` through `09_shap_dependence_sqft.png`) in `reports/figures/` are tracked and intact.
4. **Code Quality Audit**:
   - **Circular Imports**: None detected across the entire codebase.
   - **Unused Dependencies**: `requests` was imported in data download/extraction scripts but omitted from `requirements.txt`. Remedied.
   - **Duplicated Logic**: All inference, proximity calculations, and SHAP resolutions are centralized in `src/services/` with zero UI duplication.
   - **Hardcoded Local Paths**: A comprehensive AST and regex scan across all Python files confirmed that **zero application code files contain hardcoded absolute Windows paths** (`C:\Users\VISHAL\...`). All paths are constructed portably using `pathlib.Path(__file__).resolve().parent`.

---

## 2. Reproducibility & Dependency Audit

### 2.1 Python Environment & Package Alignment
- **Audited Runtime:** Python 3.12.10 (AMD64, win32).
- **Target Deployment Compatibility:** Python 3.10, 3.11, and 3.12 on Linux (Debian/Ubuntu) and macOS.
- **Third-Party Imports vs. `requirements.txt` Reconciliation:**

| Package | Purpose in EstateIQ AI | Version in `venv` | Declared in `requirements.txt` |
| :--- | :--- | :---: | :---: |
| `numpy` | Vectorized mathematical operations & log transformations | 2.5.4 | `>=1.24.0` |
| `pandas` | Tabular data processing & geospatial schema handling | 3.1.1 | `>=2.0.0` |
| `scikit-learn` | Pipeline construction, OneHotEncoder, ColumnTransformer | 1.9.1 | `>=1.3.0` |
| `xgboost` | Champion gradient boosting regressor execution | 3.4.1 | `>=2.0.0` |
| `lightgbm` | Tournament candidate regressor | 4.7.0 | `>=4.0.0` |
| `catboost` | Tournament candidate regressor | 1.2.10 | `>=1.2.0` |
| `shap` | TreeSHAP cooperative game-theoretic interpretability | 0.54.1 | `>=0.45.0` |
| `geopy` | Geodesic distance calculation on WGS84 ellipsoid | 2.5.0 | `>=2.4.0` |
| `pydeck` | High-performance 2D/3D map visualization | 0.9.3 | `>=0.9.0` |
| `pyarrow` | Apache Arrow memory format for Streamlit dataframes | 26.0.1 | `>=26.0.0` |
| `streamlit` | Reactive web application platform | 1.67.0 | `>=1.30.0` |
| `joblib` | Fast binary pipeline deserialization | 1.6.0 | `>=1.3.0` |
| `matplotlib` | Diagnostic chart generation | 3.11.2 | `>=3.8.0` |
| `seaborn` | Statistical visualization aesthetics | 0.13.2 | `>=0.13.0` |
| `requests` | OpenStreetMap Overpass API & dataset acquisition | 2.34.2 | `>=2.31.0` *(Added)* |

### 2.2 Pinned Production Dependency Recommendation
For production container deployments requiring exact reproducibility, the following pinned set is verified:
```text
numpy==2.5.4
pandas==3.1.1
scikit-learn==1.9.1
xgboost==3.4.1
lightgbm==4.7.0
catboost==1.2.10
shap==0.54.1
geopy==2.5.0
pydeck==0.9.3
pyarrow==26.0.1
requests==2.34.2
streamlit==1.67.0
joblib==1.6.0
matplotlib==3.11.2
seaborn==0.13.2
```

---

## 3. End-to-End Functional Validation

Automated end-to-end regression tests were implemented in [`tests/test_e2e_validation.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_e2e_validation.py) and verified across the complete application surface:

### 3.1 Valuation Engine Workflows
1. **Preset 1 (Hoodi Canonical 2BHK):**
   - Inputs: 1,420 sq.ft., 2 BHK, 2 Bath, 2 Balcony, Hoodi, Super built-up Area.
   - Result: Predicted Valuation = **₹74.97 Lakhs** (Matches actual asking price ₹75.00L within 0.04%).
   - Tier: `Mid-Market (₹50 - ₹100 Lakhs)`. Historical Residual-Scale Estimate = **₹61.73L – ₹88.21L** ($\pm$**₹13.24L** Tier MAE). Unit Rate = **₹5,280 / sq.ft.**
2. **Preset 2 (Whitefield Modern 3BHK):**
   - Inputs: 1,850 sq.ft., 3 BHK, 3 Bath, 2 Balcony, Whitefield.
   - Result: Predicted Valuation = **₹1.05 Cr** (₹105.15 Lakhs). Tier: `Upper-Mid (₹100 - ₹200 Lakhs)`. Margin = $\pm$₹30.75L.
3. **Preset 3 (Indiranagar Luxury 4BHK):**
   - Inputs: 3,200 sq.ft., 4 BHK, 4 Bath, 3 Balcony, Indira Nagar, Built-up Area.
   - Result: Predicted Valuation = **₹2.67 Cr** (₹266.98 Lakhs). Tier: `Luxury (> ₹200 Lakhs)`. Margin = $\pm$₹112.79L. Automatically activates the Luxury Tier Advisory Banner regarding wider variance in bespoke real estate.
4. **Preset 4 (Electronic City Budget 2BHK):**
   - Inputs: 1,050 sq.ft., 2 BHK, 2 Bath, 1 Balcony, Electronic City.
   - Result: Predicted Valuation = **₹48.50 Lakhs**. Tier: `Budget (< ₹50 Lakhs)`. Margin = $\pm$₹8.55L.
5. **Boundary & Stress Inputs:**
   - Evaluated 300 sq.ft. (compact studio) and 12,000 sq.ft. (estate plot). The pipeline executes monotonically without numerical overflow or negative valuations.
6. **Unknown Localities & Area Types:**
   - Evaluated unlisted locality (`"NonExistentNeighborhoodXYZ"`). Because the champion model pipeline utilizes Scikit-Learn's `OneHotEncoder(handle_unknown='ignore')`, unknown categories are safely mapped to all-zero dummy vectors. The model falls back to city-wide intercept dynamics without crashing.

### 3.2 Geospatial Proximity & Cartographic Explorer
1. **Whitefield 0.00 km Verification:**
   - Confirmed: Whitefield reference coordinate `(12.99574, 77.75795)` matches OpenStreetMap Node `5760197744` (Whitefield Kadugodi subway station). The geodesic distance is a genuine $0.00\text{ km}$, not a hardcoded stub. Annotated in the UI.
2. **Coordinate Integrity on Non-Success Localities:**
   - `other` (`EXCLUDED_AGGREGATE`): Confirmed strictly `NaN` latitude/longitude. Renders a clear cautionary banner explaining that synthetic aggregate categories represent dispersed regional listings.
   - `Bhoganhalli` (`UNRESOLVED`): Confirmed strictly `NaN` coordinates with missing spatial disclaimer. Zero synthetic coordinates are fabricated.
3. **PyDeck Micro-Market Visualization:**
   - Dynamically constructs a 4-layer scatterplot centered at the locality centroid with 4 distinct color schemes: Locality Centroid (🔴 Red), Metro nodes (🟢 Green), Tech hubs (🔵 Indigo), and Healthcare centers (🟡 Amber). Yields `None` safely when coordinates are missing.

### 3.3 Explainable AI (SHAP) & Provenance
1. **Target Scale Transparency:**
   - Log-target identity is clearly formulated: $\log(1 + \hat{y}) = 4.3578 + \sum \phi_i$.
   - Explicitly documented that base value $4.3578$ log units corresponds to ₹77.08 Lakhs, and that $\phi_i$ adjustments are log-margin shifts, not direct rupee additions.
2. **Deterministic Benchmark Provenance:**
   - Confirmed that the canonical Hoodi listing belongs strictly to **Row 3379 of the held-out test partition ($X_{\text{test}}$, 20% split, $N = 2,251$)**.
   - Clear banner informs users that the waterfall plot visualizes this verified benchmark property, not live dynamic computation on arbitrary slider inputs.

### 3.4 Locality Catalog & Search Filtering
1. **Keyword Filtering:** Case-insensitive search on locality names (e.g., `"Whitefield"`, `"Indira"`, `"HSR"`) dynamically filters the 234-row table without latency.
2. **Multi-Criteria Filtering:** Dual distance sliders for Metro and Tech Park filter records correctly, handling `NaN` values gracefully.

---

## 4. Security & DevSecOps Audit

| Security Domain | Audit Finding | Status |
| :--- | :--- | :---: |
| **Committed Secrets & Tokens** | `git grep` scan across commit history for `api_key`, `secret`, `password`, `token`, `bearer` returned zero matches. No `.env` or credential files exist in source control. | ✅ Passed (Clean) |
| **Sensitive Data in Logs** | Task logs and transcripts contain zero PII or credentials. Property records contain solely public real estate listing specifications. | ✅ Passed (Clean) |
| **Path Traversal Vulnerabilities** | All file loads (`models/`, `data/`, `reports/figures/`) use fixed `pathlib.Path` resolution relative to `project_root`. No user-supplied parameters are interpolated into file paths. | ✅ Passed (Secure) |
| **Network Isolation during Inference** | The application operates 100% offline during inference. Precomputed geospatial features and model artifacts are read from local disk; zero external HTTP calls are made during user valuation interactions. | ✅ Passed (Air-Gapped) |
| **Input Sanitization** | Numerical inputs are constrained via Streamlit slider/number-input bounds (`min_value`, `max_value`). Categorical inputs are constrained to whitelisted option tuples. | ✅ Passed (Sanitized) |
| **Git Exclusion Rules** | `.gitignore` properly excludes `venv/`, `__pycache__/`, `.env`, `.streamlit/secrets.toml`, OS files (`Thumbs.db`, `.DS_Store`), and large experimental baseline models (`*.joblib`). | ✅ Passed (Configured) |

---

## 5. Performance, Reliability & SLA Profiling

All performance metrics were empirically measured using Python's high-resolution performance counter (`time.perf_counter()`) and memory tracer (`tracemalloc`) across 100 sequential inferences:

```
======================================================================
ESTATEIQ AI — EMPIRICAL PERFORMANCE & RELIABILITY PROFILING
======================================================================
Cold Model Load Latency:        0.06 ms
Warm Model Cache Retrieval:     0.0189 ms
Geospatial Features Load:       0.86 ms
Inference Latency (Median):     22.90 ms
Inference Latency (p95):        27.05 ms
Inference Latency (p99):        29.07 ms
Inference Latency (Min - Max):  17.90 ms - 29.69 ms
Geo Proximity Latency (Median): 2.90 ms
Geo Proximity Latency (p95):    4.87 ms
PyDeck Map Generation:          41.64 ms
Traced Memory Delta:            0.49 MB
Peak Traced Memory:             1.14 MB
======================================================================
```

### Production SLA Assessment:
- **Single Property Valuation Latency:** **22.90 ms median** ($< 50\text{ ms}$ SLA). Sub-second response time guarantees an instantaneous user experience.
- **Geospatial Proximity Retrieval:** **2.90 ms median** ($< 10\text{ ms}$ SLA).
- **PyDeck Cartography Generation:** **41.64 ms** ($< 150\text{ ms}$ SLA).
- **Traced Memory Footprint:** **1.14 MB peak traced memory** (Process RSS: ~120–180 MB). The application is exceptionally lightweight and easily fits within a 512 MB or 1 GB container tier.

---

## 6. Deployment Readiness & Hosting Recommendations

### 6.1 Hosting Target Evaluation

| Hosting Platform | Memory Limit | Cost | Build & Runtime Environment | Recommendation Tier |
| :--- | :---: | :---: | :--- | :---: |
| **Streamlit Community Cloud** | 1.0 GB | Free | Native GitHub integration; auto-deploys from `main`; pre-configured `.streamlit/config.toml` | **🥇 Primary Recommendation** |
| **Render (Web Service)** | 512 MB (Free) / 1 GB+ | Free / $7/mo | Docker or native Python; supported via included `Procfile`; 50s cold spin-down on free tier | **🥈 Secondary Alternative** |
| **Hugging Face Spaces** | 16 GB | Free | Streamlit SDK; generous RAM, public ML community visibility | **🥉 Viable Alternative** |

### 6.2 Primary Recommendation Justification: Streamlit Community Cloud
Streamlit Community Cloud is the optimal deployment target for EstateIQ AI:
1. **Zero Infrastructure Overhead:** Direct two-click deployment from the GitHub repository `main` branch with automatic CI/CD redeployment on push.
2. **Native Framework Integration:** Out-of-the-box support for PyDeck layers, Streamlit caching (`@st.cache_resource`, `@st.cache_data`), and custom UI styling.
3. **Artifact Sizing Compliance:** Total repository size including the champion model (677 KB), geospatial tables (450 KB), and figures (3.5 MB) is **~5.5 MB**, far below Git and platform bandwidth constraints.
4. **Memory Headroom:** EstateIQ AI consumes ~180 MB RAM, leaving > 80% headroom within Streamlit Cloud's 1 GB container limit.
5. **Pre-Configured Server Configuration:** [`.streamlit/config.toml`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/.streamlit/config.toml) is already committed with production settings:
   ```toml
   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = true
   maxUploadSize = 50

   [browser]
   gatherUsageStats = false

   [theme]
   base = "light"
   primaryColor = "#2563eb"
   ```

### 6.3 Alternative Deployment: Render / Heroku / Railway
A production [`Procfile`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/Procfile) is created in the repository root:
```text
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## 7. Remaining Limitations & Non-Blockers

1. **Locality Centroid vs. Doorstep Coordinates:**
   Geospatial distances are calculated from locality reference coordinates (OpenStreetMap Nominatim centroids or transit anchors) to POIs, not from individual apartment doorways. This is clearly disclosed via UI banners.
2. **WGS84 Geodesic vs. Road Network Routing:**
   Distances reflect geodesic displacement ("as the crow flies"). Road network routing considering one-way streets and traffic congestion could be introduced in a future enhancement via Open Source Routing Machine (OSRM).
3. **Static Benchmark SHAP Waterfall:**
   To maintain sub-25ms UI performance and eliminate C-extension compilation dependencies in deployment containers, the SHAP waterfall plot renders the verified canonical held-out test listing rather than calculating dynamic SHAP on live slider inputs.

---

## 8. Exact Commands to Reproduce and Verify

### 1. Run Complete Automated Regression Test Suite (27 Tests)
```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```
*Expected Output:* `Ran 27 tests in ~4.4s ... OK`

### 2. Run End-to-End Functional & Performance Profiling Test
```powershell
.\venv\Scripts\python.exe -m unittest tests/test_e2e_validation.py
```
*Expected Output:* Prints full empirical latency and memory table; passes all 11 assertions.

### 3. Launch Web Application Locally
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
*Access URL:* `http://localhost:8501`

---

## 9. Conclusion & Production Sign-Off

The EstateIQ AI application has successfully passed all end-to-end functional validations, security audits, empirical performance benchmarks, and deployment configuration checks.

- ✅ **Champion XGBoost Model:** Preserved completely untouched without retraining.
- ✅ **Six-Feature Input Schema:** Preserved strictly intact.
- ✅ **Geospatial & SHAP Integrity:** Verified with zero coordinate or metric fabrication.
- ✅ **Security & Secrets:** 100% clean of sensitive data and credentials.
- ✅ **Production Readiness:** Fully configured for deployment on Streamlit Community Cloud or Render.

**Phase 6.4 is COMPLETE. Awaiting user approval before proceeding to Phase 6.5 deployment.**
