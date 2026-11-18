# EstateIQ AI — Geospatial Data Extraction & Proximity Intelligence Report
**Document ID:** GEO-EXT-001  
**Project:** EstateIQ AI — Phase 6.2: Bengaluru Locality Geocoding & OpenStreetMap POI Extraction  
**Date:** November 18, 2026  
**Author:** Senior Geospatial Data Engineer  
**Status:** Complete & Verified via Automated Tests  
**Data Sources:** OpenStreetMap Nominatim, OpenStreetMap Overpass API (ODbL 1.0)  

---

## 1. Executive Summary

In Phase 6.2 of EstateIQ AI, a production-grade geospatial intelligence pipeline was constructed to enrich metropolitan Bengaluru residential listings with transit, employment, and civic infrastructure proximity metrics without altering the Phase 5 champion model (**Approach A**).

### Key Empirical Accomplishments:
- **Locality Geocoding Coverage**: Successfully geocoded **211 out of 234 unique localities (90.2%)** from [`data/processed/bengaluru_cleaned.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/processed/bengaluru_cleaned.csv) using OpenStreetMap's Nominatim engine with a rate-limited protocol ($1.0\text{s}$ delay) and on-disk persistent caching.
- **Zero Coordinate Fabrication**: Exactly 21 unresolved localities, 1 out-of-bounds locality, and 1 aggregated cluster (`'other'`) were recorded with `NaN` coordinates and explicit audit statuses. Zero synthetic coordinates were invented.
- **OpenStreetMap POI Extraction**: Extracted and validated **481 Metro Station POIs**, **130 Technology Parks & Major IT Corridors**, and **1,161 Healthcare Centers & Hospitals** across the metropolitan Bengaluru bounding box ($12.75^\circ - 13.25^\circ\text{ N}, 77.35^\circ - 77.85^\circ\text{ E}$).
- **Geodesic Distance Computation**: Calculated true ellipsoid geodesic distances (in kilometres) from each locality reference point to the nearest metro station, IT hub, and hospital using `geopy.distance.geodesic`.
- **Automated Validation**: All 6 unit and integration tests in [`tests/test_geospatial.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_geospatial.py) passed with zero errors.

---

## 2. Locality Geocoding Audit (`locality_coordinates.csv`)

### A. Geocoding Status Breakdown ($N = 234$ Unique Localities)

| Status Category | Count | Percentage | Description & Governance Treatment |
| :--- | :---: | :---: | :--- |
| **`SUCCESS`** | **211** | **90.2%** | Geocoded with high confidence; coordinates verified within Bengaluru bounding box. |
| **`UNRESOLVED`** | **21** | **9.0%** | Locality string not resolved by Nominatim (spelling variants or hyper-local names). Preserved with `NaN` coordinates. |
| **`EXCLUDED_AGGREGATE`** | **1** | **0.4%** | The frequency-grouped category `'other'` (2,496 raw listings). Explicitly excluded from geocoding with zero synthetic coordinates. |
| **`OUT_OF_BOUNDS`** | **1** | **0.4%** | Locality `'Kanakapura'` resolved to Kanakapura taluk ($12.54^\circ\text{ N}$), outside metropolitan city bounds. Safely flagged with `NaN` coordinates. |
| **Total** | **234** | **100.0%** | Complete locality universe from cleaned dataset. |

### B. Geocoding Limitations & Spatial Precision Disclaimer
> [!IMPORTANT]
> **Spatial Precision Boundary**:
> 1. The housing dataset specifies only the **suburb or locality name** (e.g. `"Whitefield"`, `"Indiranagar"`), **not the building number, street address, or cadastral survey plot**.
> 2. Geocoded coordinates represent the **approximate geometric reference point** of that locality in OpenStreetMap (typically the neighborhood market center or administrative centroid), with an inherent spatial uncertainty of $\approx 1\text{ km} - 3\text{ km}$.
> 3. Distances derived from these reference points represent **neighborhood-level accessibility**, **NOT property-doorstep walking distances**.

---

## 3. OpenStreetMap Points of Interest (POIs) Summary

Overpass API queries were executed across the metropolitan extent of Bengaluru ($[12.75, 77.35, 13.25, 77.85]$) with automated multi-endpoint failover and raw JSON response caching in `data/geospatial/cache/`:

| Amenity Category | Output File Path | Extracted & Validated POIs | Extraction Criteria & Query Tags |
| :--- | :--- | :---: | :--- |
| **Metro Stations** | [`data/geospatial/metro_stations.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/metro_stations.csv) | **481** | `railway=station` AND `station=subway` / `operator=BMRCL` / `network=Namma Metro`. Records station name, coordinates, lines, and operational status. |
| **Technology Parks** | [`data/geospatial/tech_parks.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/tech_parks.csv) | **130** | Named commercial/office campuses matching `Tech Park`, `IT Park`, `ITPL`, `SEZ`, and verified major IT corridors (Manyata, Bagmane, Electronic City, RMZ Ecospace, etc.). |
| **Healthcare Infrastructure**| [`data/geospatial/hospitals.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/hospitals.csv) | **1,161** | `amenity=hospital` AND `healthcare=hospital`. Retains facility name, OSM ID, coordinates, and emergency classification where available. |

### Operational Status Transparency for Metro Stations:
- In strict adherence to engineering requirements, stations are **not assumed to be operational solely because they appear in OSM**.
- BMRCL Purple and Green line stations are marked as `Operational (Namma Metro Purple/Green Line)`.
- Phase 2 extensions or stations with construction tags are recorded as `Under Construction / Planned`.
- Unspecified entries are explicitly recorded as `Unknown / Unverified in OSM`.

---

## 4. Nearest-Amenity Distance Distribution Analysis

Computed via `geopy.distance.geodesic` on the WGS84 ellipsoid across the **211 successfully resolved localities**:

| Metric | Nearest Metro Station (`nearest_metro_km`) | Nearest Technology Park (`nearest_tech_park_km`) | Nearest Hospital (`nearest_hospital_km`) |
| :--- | :---: | :---: | :---: |
| **Mean Distance** | **1.65 km** | **2.59 km** | **0.70 km** |
| **Median (50th %)** | **1.10 km** | **2.26 km** | **0.38 km** |
| **Interquartile Range (25% - 75%)** | 0.52 km – 2.09 km | 1.28 km – 3.44 km | 0.20 km – 0.72 km |
| **Minimum Distance** | 0.00 km | 0.00 km | 0.03 km |
| **Maximum Distance** | 15.96 km | 18.56 km | 13.23 km |
| **Missing Distance Count** | 23 localities (100% accounted by unresolved/aggregate) | 23 localities | 23 localities |

### Economic & Urban Planning Insights:
1. **High Civic Healthcare Density**: Over 75% of resolved Bengaluru localities have a hospital within **$720\text{ meters}$**, reflecting the high spatial density of medical clinics and nursing homes across BBMP wards.
2. **Rapid Transit Accessibility**: The median locality distance to Namma Metro is **$1.10\text{ km}$**, demonstrating substantial transit coverage in central, western, and eastern radial corridors. Localities beyond $5\text{ km}$ represent outer peripheral suburbs (e.g., Anekal, Devanahalli outskirts).
3. **Employment Hub Proximity**: Over half of all localities lie within **$2.26\text{ km}$** of a major technology campus or commercial IT park, illustrating Bengaluru's polycentric employment architecture.

---

## 5. Technical Validation & Quality Assurance

Executed via [`tests/test_geospatial.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_geospatial.py):
1. **Bounding Box Enforcement**: Confirmed that all 211 resolved coordinates strictly satisfy $12.70 \le \text{Lat} \le 13.35$ and $77.30 \le \text{Lon} \le 77.95$.
2. **Duplicate Check**: Zero duplicate OSM IDs exist within `metro_stations.csv`, `tech_parks.csv`, or `hospitals.csv`.
3. **Non-Negative & Finite Values**: All calculated geodesic distances are strictly $\ge 0.0\text{ km}$ and finite.
4. **Offline Nearest-POI Unit Test**: Verified that `find_nearest_poi` deterministically selects the true minimum geodesic distance using controlled mock points.
5. **No Road-Circuity Multipliers**: In accordance with requirements, straight-line geodesic displacements were **not** artificially multiplied by assumed circuity factors.

---

## 6. Deliverables Inventory

| File Path | File Description | Status |
| :--- | :--- | :---: |
| [`src/geospatial/geocode_localities.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/geospatial/geocode_localities.py) | Rate-limited Nominatim geocoding script with caching. | Complete ✅ |
| [`src/geospatial/extract_osm_amenities.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/geospatial/extract_osm_amenities.py) | Overpass POI extractor with multi-endpoint failover. | Complete ✅ |
| [`src/geospatial/calculate_locality_distances.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/geospatial/calculate_locality_distances.py) | Geodesic proximity calculation engine. | Complete ✅ |
| [`tests/test_geospatial.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/tests/test_geospatial.py) | Unit and integration test suite (6/6 passing). | Complete ✅ |
| [`data/geospatial/locality_coordinates.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/locality_coordinates.csv) | Geocoded coordinates and metadata for 234 localities. | Generated ✅ |
| [`data/geospatial/metro_stations.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/metro_stations.csv) | Validated metro stations across Bengaluru. | Generated ✅ |
| [`data/geospatial/tech_parks.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/tech_parks.csv) | Validated technology corridors and campuses. | Generated ✅ |
| [`data/geospatial/hospitals.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/hospitals.csv) | Validated hospitals and healthcare facilities. | Generated ✅ |
| [`data/geospatial/locality_geospatial_features.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/geospatial/locality_geospatial_features.csv) | Final locality features with nearest-amenity metrics. | Generated ✅ |
| [`requirements.txt`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/requirements.txt) | Updated with `geopy>=2.4.0`. | Updated ✅ |

---

## 7. Model Safeguards Verification

- **Champion Model**: [`models/estateiq_advanced_model.joblib`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/models/estateiq_advanced_model.joblib) was **not modified or retrained**.
- **Phase 4 & 5 Reports**: [`reports/advanced_model_comparison.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/advanced_model_comparison.md) and [`reports/shap_model_explainability.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/shap_model_explainability.md) remain untouched and intact.
- **Application Entry Point**: `app.py` has **not been created yet**, pending Phase 6.3 approval.
