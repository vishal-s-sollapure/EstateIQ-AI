# EstateIQ AI — Unified Property Data Schema Specification
**Document Version:** 1.0.0  
**Scope:** Architectural blueprint for property-level tabular, geospatial, multimodal, and inference data.  
**Path:** `docs/data_schema.md`

---

## 1. Executive Summary & Design Principles

EstateIQ AI is designed as a multimodal real estate valuation and investment intelligence engine. To support evolution from simple tabular regression into geospatial intelligence (OpenStreetMap), explainable AI (SHAP), multimodal appraisal (text & vision embeddings), and investment analytics, the platform enforces a **strictly partitioned, multi-layer data schema**.

### Core Schema Principles:
1. **Explicit Lineage & Provenance**: Every attribute belongs to a distinct operational layer (raw, cleaned, engineered, user-assumed, or model-inferred).
2. **Zero-Fabrication Policy**: Missing values (e.g., coordinates, societies, descriptions) remain explicitly `NULL` (`None`) or flagged with audit metadata. No synthetic values, coordinates, or dates are hallucinated.
3. **Decoupled Architecture**: Tabular, geospatial, and embedding vectors are linked via an immutable primary key (`property_id`), allowing independent versioning.

---

## 2. Layered Property-Level Schema Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: System Identity & Data Governance Metadata                   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Raw Ingested Attributes (Immutable Historical Record)        │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Cleaned & Standardized Physical Attributes                   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Engineered Geospatial & Proximity Attributes (OSM)           │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Multimodal Embeddings (Text & Imagery - Planned)             │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 6: Macroeconomic & Temporal Context (External Indices)           │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 7: Runtime User Assumptions (Investment Scenario Modeling)      │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 8: Model-Generated Valuations, XAI & Investment Inferences      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Data Dictionary

### Layer 1: System Identity & Data Governance Metadata

| Field Name | Data Type | Nullable? | Origin | Description & Business Rules |
| :--- | :---: | :---: | :---: | :--- |
| `property_id` | `UUIDv4` / `string` | **No** | System | Universally unique, deterministic hash or UUID assigned upon ingestion (e.g., `prp_blr_a1b2c3d4`). Primary key. |
| `data_source` | `string` | **No** | Metadata | Origin channel of record (e.g., `kaggle_historical_2018`, `user_submission`, `scraped_portal`). |
| `ingestion_timestamp`| `datetime (UTC)` | **No** | System | ISO 8601 timestamp of record creation (`YYYY-MM-DDTHH:MM:SSZ`). |
| `data_quality_flag` | `string` | **No** | System | Quality classification (`VERIFIED`, `OUTLIER_REMOVED`, `MISSING_COORDINATES`, `DIRTY_AREA`). |
| `is_outlier` | `boolean` | **No** | Pipeline | Flag marking statistical or domain anomaly (e.g., extreme $\text{sqft}/\text{bhk}$ ratio). |

---

### Layer 2: Raw Ingested Attributes (Immutable Historical Snapshot)

*These fields reflect the exact raw data as ingested from `Bengaluru_House_Data.csv` before transformations.*

| Field Name | Raw Type | Nullable? | Raw Dataset Column | Description |
| :--- | :---: | :---: | :---: | :--- |
| `raw_area_type` | `string` | Yes | `area_type` | Super built-up Area, Built-up Area, Plot Area, Carpet Area. |
| `raw_availability` | `string` | Yes | `availability` | Historical string (`Ready To Move`, `19-Dec`, etc.). |
| `raw_location` | `string` | Yes | `location` | Unsanitized locality text entered by seller. |
| `raw_size` | `string` | Yes | `size` | Unsanitized bedroom string (`2 BHK`, `4 Bedroom`). |
| `raw_society` | `string` | Yes (~41%)| `society` | Housing society name/code. |
| `raw_total_sqft` | `string` | Yes | `total_sqft` | Raw string with ranges (`2100 - 2850`) or units (`34.46Sq. Meter`). |
| `raw_bath` | `float` | Yes (<1%) | `bath` | Recorded bathroom count. |
| `raw_balcony` | `float` | Yes (~5%) | `balcony` | Recorded balcony count. |
| `raw_price_lakhs` | `float` | No | `price` | Historical asking price in Lakhs INR ($1\text{ Lakh} = 100,000\text{ INR}$). |

---

### Layer 3: Cleaned & Standardized Physical Attributes

*Produced by `src/data/clean_data.py`. Standardized tabular baseline ready for preprocessing.*

| Field Name | Target Type | Nullable? | Transformation Logic |
| :--- | :---: | :---: | :--- |
| `clean_location` | `string` | **No** | Trimmed, lowercase, grouped rare entries ($\le 10$ listings) into `"other"`. |
| `bhk` | `integer` | **No** | Integer extracted from `raw_size` ($1 \le \text{bhk} \le 10$). |
| `total_sqft` | `float` | **No** | Parsed numeric square footage (mean of hyphenated ranges; dropped unconvertible). |
| `bath` | `integer` | **No** | Median-imputed or valid integer bathroom count. |
| `balcony` | `integer` | Yes | Cleaned balcony count ($0, 1, 2, 3$). |
| `actual_price_lakhs`| `float` | **No** | Benchmark ground truth price in Lakhs INR (Target $y$). |
| `price_per_sqft` | `float` | **No** | $(\text{actual\_price\_lakhs} \times 100,000) / \text{total\_sqft}$. **Restricted to outlier filtering; never used as a model training input.** |

---

### Layer 4: Engineered Geospatial & Proximity Attributes (OpenStreetMap)

*Sourced via Geocoding + OpenStreetMap (OSM) Overpass API. Lat/Lon are stored explicitly.*

| Field Name | Type | Nullable? | Source | Description |
| :--- | :---: | :---: | :---: | :--- |
| `latitude` | `float (decimal)`| **Yes** | OSM Nominatim | WGS84 Latitude of neighborhood centroid (e.g., `12.9698`). |
| `longitude` | `float (decimal)`| **Yes** | OSM Nominatim | WGS84 Longitude of neighborhood centroid (e.g., `77.7500`). |
| `geo_resolution` | `enum` | **Yes** | System | Accuracy level (`EXACT_ADDRESS`, `NEIGHBORHOOD_CENTROID`, `CITY_DEFAULT`, `UNRESOLVED`). |
| `dist_nearest_metro_km` | `float` | **Yes** | OSM Overpass | Haversine distance in km to nearest Namma Metro station. |
| `dist_nearest_tech_park_km`| `float`| **Yes** | OSM Overpass | Distance to nearest major IT corridor (ITPL, Manyata, Electronic City). |
| `dist_cbd_mg_road_km` | `float` | **Yes** | Calculated | Distance to central business district (MG Road / Brigade Road centroid). |
| `dist_kempegowda_airport_km`| `float`| **Yes** | Calculated | Distance to Kempegowda International Airport (BLR). |
| `schools_within_2km_count` | `integer`| **Yes** | OSM Overpass | Density of primary/secondary schools within a $2\text{ km}$ geodesic radius. |
| `hospitals_within_3km_count`| `integer`| **Yes** | OSM Overpass | Density of clinics and hospitals within a $3\text{ km}$ geodesic radius. |

---

### Layer 5: Multimodal Embeddings (Planned Architecture)

*Reserved for future multimodal valuation phases. Not populated in Phase 1.*

| Field Name | Type | Nullable? | Expected Source | Description |
| :--- | :---: | :---: | :---: | :--- |
| `property_description` | `text` | **Yes** | Listing Portals | Unstructured listing text highlighting interior finishes, views, orientation. |
| `text_embedding_vector` | `float[384]` | **Yes** | Sentence-BERT / MiniLM | Dense semantic embedding vector of property description text. |
| `image_url` | `string` | **Yes** | Portfolio / Media | Secure URI or file path to property elevation or interior photo. |
| `image_embedding_vector`| `float[512]` | **Yes** | CLIP / ResNet | Visual feature vector capturing property aesthetic quality and condition. |

---

### Layer 6: Macroeconomic & Temporal Context (External Indicators)

*Used for temporal inflation alignment between historical listing price (2016-2018) and real-time valuation.*

| Field Name | Type | Nullable? | External Source | Description |
| :--- | :---: | :---: | :---: | :--- |
| `listing_year` | `integer` | **Yes** | Metadata | Year when property was advertised (estimated 2016–2018 for Kaggle base). |
| `rbi_repo_rate_pct` | `float` | **Yes** | RBI Bulletin | Benchmark Reserve Bank of India lending repo rate at transaction epoch. |
| `cpi_housing_index` | `float` | **Yes** | Ministry of Statistics | Consumer Price Index specific to housing inflation in urban Karnataka. |

---

### Layer 7: Runtime User Assumptions (Investment Scenario Modeling)

*Input parameters supplied by the end-user via the Streamlit dashboard (`app.py`) for investment evaluation.*

| Field Name | Type | Nullable? | User Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `assumed_down_payment_pct` | `float` | No | `20.0%` | Share of purchase price paid in cash upfront. |
| `assumed_loan_interest_pct`| `float` | No | `8.5%` | Bank home loan annual interest rate. |
| `assumed_loan_tenure_years`| `integer` | No | `20` | Mortgage tenure in years. |
| `assumed_monthly_rent_inr` | `float` | Yes | `Auto-estimated` | Expected gross monthly rental income. |
| `assumed_holding_years` | `integer` | No | `5` | Projected investment holding horizon before resale. |

---

### Layer 8: Model-Generated Valuations, XAI & Inferences

*Generated dynamically by the machine learning pipeline and rendered in the user interface.*

| Field Name | Type | Nullable? | Producing Component | Description |
| :--- | :---: | :---: | :---: | :--- |
| `predicted_price_lakhs` | `float` | No | Model Pipeline | Point estimate valuation in Lakhs INR. |
| `pred_confidence_interval_low`| `float`| No | Quantile Regression | Lower bound of valuation at 90% confidence. |
| `pred_confidence_interval_high`|`float`| No | Quantile Regression | Upper bound of valuation at 90% confidence. |
| `top_shap_positive_feature`| `string` | No | SHAP Explainer | Feature contributing most positively to valuation markup. |
| `top_shap_negative_feature`| `string` | No | SHAP Explainer | Feature discounting the property's valuation. |
| `estimated_rental_yield_pct`| `float` | No | Investment Analytics | $(\text{Annual Rent} / \text{Predicted Price}) \times 100$. |
| `investment_risk_tier` | `enum` | No | Rules Engine | Risk grade: `LOW`, `BALANCED`, `SPECULATIVE` based on infrastructure density. |

---

## 4. Provenance & Feasibility Matrix

| Schema Layer | Can be populated from Kaggle CSV? | Additional Required Source | Implementation Phase |
| :--- | :---: | :--- | :---: |
| **Layer 1: Identity & Metadata** | **Yes** (Generated) | None (UUID & hash pipelines) | Phase 1 |
| **Layer 2: Raw Ingested Data** | **Yes** | Kaggle `Bengaluru_House_Data.csv` | Phase 1 |
| **Layer 3: Cleaned Physical Data**| **Yes** | Ingestion & cleaning script (`clean_data.py`) | Phase 1 |
| **Layer 4: Geospatial & POI Data**| **No** | OpenStreetMap Nominatim & Overpass API | Phase 2 |
| **Layer 5: Multimodal Embeddings**| **No** | Property descriptions & photos + Vision/NLP models | Phase 4 (Advanced) |
| **Layer 6: Macroeconomic Indices**| **No** | RBI Historical Rates & Ministry of Statistics CPI | Optional Enrichment |
| **Layer 7: User Assumptions** | **No** | Streamlit UI interactive sliders in `app.py` | Phase 3 |
| **Layer 8: Model Predictions & SHAP**| **No** | Scikit-Learn / XGBoost models + SHAP library | Phase 3 |

---

## 5. Data Quality & Imputation Integrity Rules

1. **No Synthetic Geographic Coordinates**: If OpenStreetMap Nominatim cannot resolve a location name with high confidence, `latitude` and `longitude` are set to `NULL` and `geo_resolution` is marked `UNRESOLVED`. The system will not generate fake coordinates.
2. **Missing Balconies**: Missing balconies are imputed with the median ($1$) and tagged with an imputation flag, or modeled with a dedicated `has_balcony` binary indicator.
3. **Target Integrity**: Under no circumstances should `price_per_sqft` or any Layer 8 field enter the feature set $X$ during model training.
