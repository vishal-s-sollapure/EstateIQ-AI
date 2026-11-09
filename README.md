# EstateIQ AI 🏢
### AI-Powered Real Estate Valuation & Investment Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-7EBC6F?logo=openstreetmap&logoColor=white)](https://www.openstreetmap.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_1:_EDA_&_Pipeline_Setup-orange)]()

---

## 1. Executive Summary

**EstateIQ AI** is an open-source machine learning and geospatial intelligence platform engineered to predict residential property valuations and deliver transparent investment analytics across metropolitan **Bengaluru, India**. 

Rather than treating property valuation as a black-box spreadsheet calculation, EstateIQ AI combines:
1. **Supervised Regression Modeling** on cleaned physical property characteristics.
2. **Geospatial Proximity Analytics** via OpenStreetMap (OSM) to measure real-world access to transit, tech corridors, and civic amenities.
3. **Explainable AI (SHAP)** to explain the exact economic drivers behind every valuation.
4. **Investment Scenario Analysis** to calculate rental yields and financial returns for homebuyers and investors.

---

## 2. The Real-World Problem

Real estate is one of India's largest wealth-generating asset classes, yet property valuation in urban hubs like Bengaluru suffers from severe market friction:

- **Information Asymmetry**: Prospective buyers and retail investors lack transparent benchmarks, relying on broker quotes that often carry unverified markup buffers.
- **Rapid Spatial Transformation**: In Bengaluru ("Silicon Valley of India"), the rapid expansion of the **Namma Metro**, new tech parks (Manyata, ITPL, Bellandur ORR), and arterial roads dramatically shifts land values within months.
- **Black-Box Estimates**: Traditional automated valuation models (AVMs) output flat price tags without explaining *why* a property was appraised at a specific value.
- **Data Quality Inconsistencies**: Property listings are plagued by mixed measurement units, hyphenated area ranges, non-standard room descriptions, and high categorical cardinality.

---

## 3. Project Objectives

- **Develop a Robust Regression Pipeline**: Ingest, sanitize, and model Bengaluru housing data with zero synthetic data fabrication.
- **Incorporate Geospatial Intelligence**: Quantify the impact of urban infrastructure (metro access, tech hubs, healthcare density) using OpenStreetMap without relying on hardcoded coordinates.
- **Ensure Model Explainability**: Integrate **SHAP (SHapley Additive exPlanations)** to provide transparent, feature-level attribution for every prediction.
- **Deliver an Interactive Product**: Build an intuitive **Streamlit** web application featuring interactive Folium maps, financial scenario sliders, and localized valuation reports.
- **Maintain Production Engineering Rigor**: Follow modular design patterns, strict version control, data contracts, and complete offline reproducibility.

---

## 4. Planned Feature Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EstateIQ AI Capabilities                           │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│    Valuation Engine      │  Geospatial Intelligence │   Investment Analytics    │
├──────────────────────────┼──────────────────────────┼───────────────────────────┤
│ • Multi-algorithm models │ • Locality Geocoding     │ • Rental yield estimation │
│ • Outlier-aware pipeline │ • Namma Metro proximity  │ • EMI & mortgage modeling │
│ • Quantile intervals     │ • Tech park distance     │ • Holding-period returns  │
│ • SHAP feature breakdown │ • Amenity buffer counts  │ • Infrastructure risk tier│
└──────────────────────────┴──────────────────────────┴───────────────────────────┘
```

- **Price Prediction**: Predicts expected residential price (in Lakhs INR) from user inputs (location, total sq. ft., BHK, bathrooms).
- **Explainability Waterfall**: Displays local SHAP waterfall plots detailing which factors increased or reduced the appraisal.
- **Neighborhood Proximity Radar**: Computes geodesic distances to transit and tech hubs.
- **Financial Scenario Calculator**: Calculates down payment, monthly EMI, and gross rental yield based on user-defined financial assumptions.

---

## 5. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core development runtime. |
| **Data Manipulation** | `pandas`, `numpy` | Tabular data wrangling, cleaning pipelines, and vectorized mathematics. |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm` | Preprocessing, feature encoding, model training, cross-validation, and metrics. |
| **Model Interpretability**| `shap` | Cooperative game-theoretic feature attribution (Explainable AI). |
| **Geospatial Analytics** | `geopy`, `osmnx`, `geopandas`, `shapely` | Nominatim forward geocoding, Overpass API queries, spatial indexing ($k$-d trees). |
| **Interactive Visualization**| `folium`, `plotly`, `matplotlib`, `seaborn` | Interactive Leaflet maps, distribution charts, and metric diagnostics. |
| **Web Application** | `streamlit` | Reactive frontend dashboard for model inference and investment simulation. |
| **Model Persistence** | `joblib` | Fast binary serialization of trained pipelines into `models/`. |

---

## 6. System Architecture

```
                 RAW HOUSING DATA
             (Bengaluru_House_Data.csv)
                         │
                         ▼
             ┌───────────────────────┐
             │  src/data/clean_data  │ ◀─── Strips ranges, cleans BHK & location,
             └───────────────────────┘      removes domain outliers (sqft/bhk < 300)
                         │
                 CLEANED DATASET
                         │
                         ▼
             ┌───────────────────────┐      OPENSTREETMAP (OSM)
             │  src/features/osm     │ ◀─── Overpass API + Nominatim Geocoding
             └───────────────────────┘      (Metro distance, tech hubs, amenities)
                         │
              ENRICHED FEATURE MATRIX
                         │
                         ▼
             ┌───────────────────────┐
             │  src/models/train.py  │ ◀─── Cross-validation, hyperparameter tuning
             └───────────────────────┘      (Ridge, Random Forest, XGBoost)
                         │
              SERIALIZED MODEL (.joblib)
                         │
                         ▼
       ┌───────────────────────────────────┐
       │         STREAMLIT DASHBOARD       │
       │              (app.py)             │
       ├─────────────────┬─────────────────┤
       │ Valuation Form  │  Folium Map     │
       │ SHAP Waterfall  │  Investment ROI │
       └─────────────────┴─────────────────┘
```

---

## 7. Project Directory Structure

```text
EstateIQ-AI/
├── data/
│   ├── raw/                # Immutable original datasets (e.g. Bengaluru_House_Data.csv)
│   └── processed/          # Cleaned, standardized, and geospatially enriched datasets
├── notebooks/              # Jupyter notebooks for EDA and experimental validation
├── src/                    # Modular production source package
│   ├── __init__.py
│   ├── check_dataset.py    # Dataset health, integrity, and schema inspector
│   ├── data/               # Ingestion, parsing, and cleaning modules
│   ├── features/           # OSM geospatial extractors and feature transformers
│   ├── models/             # Training routines, tuning, and SHAP explainers
│   └── utils/              # Helper utilities, logging, and configuration
├── models/                 # Saved model binaries (.joblib) and encoders
├── reports/                # Evaluation reports, diagnostic charts, and metric logs
├── screenshots/            # UI captures and map visuals for documentation
├── docs/                   # Architectural blueprints and technical specifications
│   ├── data_schema.md      # Multi-layer data schema specification
│   └── geospatial_plan.md  # OpenStreetMap integration and caching blueprint
├── app.py                  # Streamlit interactive application entry point
├── train.py                # Standalone training pipeline entry point
├── requirements.txt        # Pinned Python package dependencies
├── .gitignore              # Git ignore rules for environments, caches, and datasets
└── README.md               # Project documentation
```

---

## 8. Dataset Provenance & Known Limitations

- **Source**: Bengaluru residential real estate listings compiled by Amitabha Joy via [Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data).
- **Records**: 13,320 historical listings across 9 attributes.
- **Price Metric**: Quoted in **Lakhs INR** ($1\text{ Lakh} = 100,000\text{ INR}$).

### Critical Analytical Distinctions:
1. **Historical Asking Price vs. Transaction Price**: The dataset contains **seller listing prices** from approximately 2016–2018, not official registered stamp-duty transaction prices from the Karnataka Kaveri portal. Asking prices typically contain a 5%–15% negotiation buffer.
2. **Locality Centroids vs. Doorstep Coordinates**: The raw dataset contains neighborhood names (`location`) but **no latitude/longitude coordinates**. In Phase 2, coordinates are resolved to neighborhood centroids via OpenStreetMap Nominatim. They represent approximate suburb centers, not exact property doorsteps.
3. **Zero Synthetic Fabrication**: Missing fields (e.g., `society`) are preserved as `NULL` or audited via boolean flags. We never generate artificial properties or synthetic coordinates.

---

## 9. Installation & Quickstart (Windows 11 / PowerShell)

### Prerequisites
- Python 3.10 – 3.12 installed and added to `PATH`.
- Git installed.

### 1. Clone the Repository & Navigate
```powershell
git clone https://github.com/<your-username>/EstateIQ-AI.git
cd EstateIQ-AI
```

### 2. Activate the Virtual Environment
```powershell
# Create venv if not already present: python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If prompted with an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your current PowerShell session).*

### 3. Install Phase 1 Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Dataset Health Inspector
Ensure `Bengaluru_House_Data.csv` is placed inside `data/raw/`, then run:
```powershell
python src/check_dataset.py
```

### 5. Launch the Web Application (Preview)
```powershell
streamlit run app.py
```

---

## 10. Current Development Status

| Milestone | Phase | Status | Key Deliverables |
| :--- | :---: | :---: | :--- |
| **Project Setup & Environment** | Phase 0 | **Completed** ✅ | Clean folder structure, `.gitignore`, `requirements.txt`, virtual environment. |
| **Architecture Documentation** | Phase 0 | **Completed** ✅ | [`docs/data_schema.md`](docs/data_schema.md), [`docs/geospatial_plan.md`](docs/geospatial_plan.md). |
| **Dataset Health Check** | Phase 1 | **Completed** ✅ | [`src/check_dataset.py`](src/check_dataset.py) with full audit output. |
| **Exploratory Data Analysis (EDA)** | Phase 1 | **In Progress** 🔄 | Analyzing distributions, outlier boundaries, and non-standard area formats. |
| **Data Cleaning Pipeline** | Phase 1 | **In Progress** 🔄 | Standardizing `total_sqft`, extracting `bhk`, frequency-grouping rare locations. |
| **OpenStreetMap Integration** | Phase 2 | **Planned** ⏳ | Nominatim forward geocoding, persistent JSON caching, POI distance extraction. |
| **Model Benchmarking** | Phase 3 | **Planned** ⏳ | Baseline vs. Ridge vs. Random Forest vs. XGBoost / LightGBM. |
| **Explainable AI (SHAP)** | Phase 4 | **Planned** ⏳ | TreeExplainer, local waterfall plots, global feature importance. |
| **Investment Analytics Engine** | Phase 5 | **Planned** ⏳ | Rental yield computation, mortgage EMI engine, scenario modeling. |
| **Interactive Dashboard & Deployment** | Phase 6 | **Planned** ⏳ | Full Streamlit UI with Folium choropleths, deployment on Streamlit Community Cloud. |

> *Note: Model accuracy metrics ($R^2$, RMSE, MAE) will be published here upon completion of the Phase 3 benchmarking runs. No metrics are fabricated prior to empirical evaluation.*

---

## 11. Author & Academic Context

- **Developer**: Second-Year CSE (AI & Machine Learning) Student.
- **Academic Focus**: Applied Machine Learning, Geospatial Data Science, Explainable AI (XAI).
- **Contributions**: Contributions, issue reports, and feedback are welcome!
