# EstateIQ AI — OpenStreetMap (OSM) Geospatial Integration Architecture
**Document Version:** 1.0.0  
**Phase:** Phase 6 Implementation Blueprint  
**Path:** `docs/geospatial_plan.md`

---

## 1. Executive Overview & Objectives

In real estate economics, **location is the single greatest determinant of residential valuation**. While tabular features (BHK, total square footage, bathrooms) define the physical unit, spatial proximity to transit, employment hubs, and civic amenities drives land appreciation and price premiums.

This document plans the **Phase 6 Geospatial Intelligence Pipeline** for EstateIQ AI, integrating OpenStreetMap (OSM) to enrich Bengaluru residential properties with high-resolution location intelligence without fabricating coordinates or violating API policies.

---

## 2. Technology Stack & Roles

| Library / Tool | Primary Technical Function |
| :--- | :--- |
| **OpenStreetMap (OSM)** | Open, crowd-sourced geographic database of worldwide infrastructure, transport, and amenities. |
| **GeoPy (Nominatim)** | Forward geocoding service translating textual locality strings (e.g., `"Whitefield, Bengaluru"`) into WGS84 decimal coordinates. |
| **OSMnx** | Python package to retrieve, model, analyze, and visualize OpenStreetMap street networks and geospatial POIs (Points of Interest). |
| **GeoPandas** | Spatial extension of `pandas` enabling geometric data types (`Point`, `Polygon`), spatial joins, and coordinate reference system (CRS) projections. |
| **Shapely** | Core computational geometry library for point-in-polygon tests, geodesic buffering ($1\text{ km}$, $2\text{ km}$ circles), and intersections. |
| **Folium** | Leaflet.js-powered interactive mapping library integrated directly into Streamlit (`streamlit-folium`) to render interactive choropleth and marker maps. |

---

## 3. Locality Geocoding & The Spatial Resolution Boundary

### A. Forward Geocoding Strategy
The raw Kaggle dataset provides neighborhood text strings in the `location` column (e.g., `"Indiranagar"`, `"Electronic City Phase 1"`, `"Yelahanka"`), but **zero latitude and longitude values**.

To map these strings to geographic points:
1. **String Normalization**: Strip whitespaces, lower-case, remove special characters, and append standard administrative hierarchy:  
   `"{clean_location}, Bengaluru, Karnataka, India"`
2. **GeoPy Nominatim Querying**:
   ```python
   from geopy.geocoders import Nominatim
   geolocator = Nominatim(user_agent="estateiq_ai_academic_research")
   location = geolocator.geocode("Whitefield, Bengaluru, Karnataka, India")
   # Returns: (latitude, longitude)
   ```

### B. Crucial Distinction: Exact Coordinates vs. Approximate Locality Centroids

> [!IMPORTANT]
> **Spatial Precision Disclaimer**:
> - **Locality Centroid ($\approx 1\text{ km} - 3\text{ km}$ precision)**: Our dataset only specifies the *neighborhood*, not the specific street address, building number, or survey plot. A geocoded point represents the **geometric centroid of that entire suburb**, not the specific doorstep.
> - **Doorstep Accuracy**: Real estate portals often charge for or withhold exact street coordinates to protect seller privacy.
> - **Scientific Integrity**:
>   1. We will **never** claim that distances calculated from locality centroids are exact property doorstep distances.
>   2. Distances will be clearly reported in reports and UI as **"Estimated Neighborhood Centroid Distance"**.
>   3. If Nominatim cannot resolve a locality name with confidence, the record will be flagged as `geo_resolution = UNRESOLVED` with `NULL` coordinates. **We will never fabricate random or synthetic coordinates.**

---

## 4. OpenStreetMap Data Extraction & Tagging

OSM structures real-world entities into nodes (points), ways (lines/boundaries), and relations, tagged with key-value pairs. We will query the **Overpass API** (via `osmnx` or direct Overpass queries) for the metropolitan area of Bengaluru (`bbox` or administrative boundary):

```
                                  OSM Overpass API
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  Transit Layers                 Employment Corridors              Civic Amenities
┌──────────────────────┐       ┌──────────────────────┐         ┌──────────────────────┐
│ railway = subway     │       │ landuse = commercial │         │ amenity = hospital   │
│ railway = station    │       │ office = IT          │         │ amenity = school     │
│ highway = bus_stop   │       │ industrial = IT      │         │ leisure = park       │
└──────────────────────┘       └──────────────────────┘         └──────────────────────┘
```

### Specific OSM Tags for Bengaluru:

| Feature Category | OSM Key-Value Query | Real-World Target |
| :--- | :--- | :--- |
| **Namma Metro** | `railway=station` and `subway=yes` | All operational and under-construction Namma Metro stations. |
| **Primary IT Corridors** | Coordinates of major tech parks | Electronic City, Manyata Tech Park, ITPL (Whitefield), Embassy TechVillage, Bagmane Tech Park. |
| **Healthcare** | `amenity=hospital` or `amenity=clinic` | Multi-specialty hospitals and clinics. |
| **Education** | `amenity=school` or `amenity=college` | Primary, secondary, and international schools. |
| **Green Spaces** | `leisure=park` or `landuse=recreation_ground` | Public parks (e.g., Cubbon Park, Lalbagh, local BDA parks). |
| **Arterial Highways**| `highway=motorway` or `highway=trunk` | Outer Ring Road (ORR), Hosur Road, Bellary Road, NICE Ring Road. |

---

## 5. Distance and Density Computation Methodology

### A. Distance Calculation (Point-to-Point)
For calculating distance from a property centroid to the nearest metro station or major IT hub:
- **Geodesic / Haversine Distance**: Measures the great-circle distance between two points on the Earth's surface (WGS84 ellipsoid).
- **Metric Projection (`EPSG:32643`)**: For fast spatial queries across thousands of records, coordinates will be projected from WGS84 (`EPSG:4326` in degrees) to **UTM Zone 43N (`EPSG:32643` in meters)**, which is the official projected coordinate system for Karnataka/Bengaluru.

### B. High-Performance Proximity with $k$-d Trees
Instead of iterating with nested $O(N \times M)$ loops:
- We will index all OSM points of interest into a **Spatial $k$-d Tree** (`scipy.spatial.cKDTree` or `sklearn.neighbors.BallTree`).
- Query time drops to $O(\log M)$, allowing instant retrieval of the nearest metro station and exact Euclidean distance in meters.

### C. Radius Density Metrics (Amenity Counts)
To calculate civic infrastructure density:
1. Construct geodesic buffer circles around each locality centroid:
   - $r_1 = 1.0\text{ km}$ (immediate neighborhood walkability)
   - $r_2 = 3.0\text{ km}$ (short vehicular commute)
2. Perform a **Spatial Join (`gpd.sjoin`)** to count the number of hospitals, schools, and parks intersecting each buffer.

---

## 6. API Governance, Rate Limiting, and Persistent Caching

OpenStreetMap Nominatim is a shared community resource governed by strict usage guidelines:
1. **Nominatim Usage Policy**:
   - Maximum rate: **1 request per second**.
   - A valid and descriptive HTTP `User-Agent` identifying the application is mandatory.
   - Bulk geocoding without caching is strictly prohibited.

### Caching Architecture (The Geocoding Database):
Because our cleaned dataset has $\approx 240$ unique locations after frequency grouping, **we never need to query Nominatim thousands of times**.

```
[Cleaned Unique Locations (~240)]
             │
             ▼
   Does location exist in
 `data/processed/osm_geocache.json`?
      ├── YES ──▶ Load (lat, lon, amenities) from Cache (0 ms)
      └── NO  ──▶ Query Nominatim (1 request/sec delay)
                  └──▶ Write to `osm_geocache.json` permanently
```

By persisting this cache to `data/processed/osm_geocache.json`, geocoding is executed **exactly once**, eliminating API dependency and ensuring 100% offline reproducibility.

### Legal Attribution:
Every map rendered in Streamlit and report generated by EstateIQ AI must include the mandatory OSM copyright notice:
> *"Map data © OpenStreetMap contributors, ODbL 1.0."*

---

## 7. End-to-End Geospatial Pipeline Architecture

When implemented in Phase 6, the modular source code will reside in [`src/features/`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/features/):

```
src/features/
├── __init__.py
├── geocoder.py          # GeoPy Nominatim wrapper with rate-limiting & disk cache
├── osm_collector.py     # Overpass / OSMnx queries to fetch Metro, IT parks, amenities
└── spatial_features.py  # BallTree distance calculations & buffer density aggregations
```

### Execution Flow:
1. **Input**: Read `data/processed/bengaluru_cleaned.csv`.
2. **Geocode Extraction**: Look up `clean_location` against `osm_geocache.json` (populate cache if new).
3. **POI Ingestion**: Download and cache Bengaluru metro station and IT park coordinates.
4. **Spatial Join**:
   - Compute `dist_nearest_metro_km`
   - Compute `dist_nearest_tech_park_km`
   - Compute `schools_within_2km`
   - Compute `hospitals_within_3km`
5. **Output**: Write enriched table to `data/processed/bengaluru_geospatial_enriched.csv`.
6. **Visualization**: Provide interactive Folium choropleth layers inside `app.py`.

