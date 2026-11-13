# EstateIQ AI — Production Data Cleaning & Preprocessing Report
**Document ID:** CLN-RPT-001  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Input Dataset:** `data/raw/Bengaluru_House_Data.csv` (13,320 rows, 9 columns)  
**Output Dataset:** `data/processed/bengaluru_cleaned.csv` (11,479 rows, 8 columns)  
**Pipeline Script:** `src/data/clean_data.py`  
**Execution Status:** Verified & Passed (Exit Code: 0)  

---

## 1. Executive Summary & Objectives

The goal of Phase 3 was to engineer an automated, robust, and reproducible data cleaning pipeline that converts raw, noisy, human-entered Bengaluru real estate listings into a sanitized, machine-learning-ready tabular dataset.

The pipeline applies domain-aware filtering and statistical boundaries without blindly stripping authentic high-value luxury real estate or causing data leakage.

```
┌────────────────────────────────────────────────────────┐
│  Raw Dataset: 13,320 rows × 9 columns                  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  1. Deduplication: -529 exact duplicate listings       │
├────────────────────────────────────────────────────────┤
│  2. Feature Pruning: Dropped 'society' & 'availability'│
├────────────────────────────────────────────────────────┤
│  3. Missing Value Handling: -17 rows (size/location)   │
├────────────────────────────────────────────────────────┤
│  4. Area Standardizing: Converted ranges & units to sqft│
├────────────────────────────────────────────────────────┤
│  5. Amenity Imputation: Contextual median imputation   │
├────────────────────────────────────────────────────────┤
│  6. Location Grouping: 1,305 ──▶ 234 dense categories  │
├────────────────────────────────────────────────────────┤
│  7. Domain Outlier Removal:                            │
│     • Area Density (< 300 sqft/BHK): -738 rows         │
│     • Bathroom Mismatch (bath > bhk+2): -9 rows        │
│     • Agricultural Cutoff (< ₹1,000/sqft): -28 rows    │
│     • Locality 2-Sigma Filtering: -520 rows            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Final Cleaned Dataset: 11,479 rows × 8 columns        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Data Cleaning Audit Trail

### A. Deduplication (Removed 529 Records)
- **Problem**: 529 exact duplicate listings were identified in the raw dataset. In property portals, multiple brokers frequently cross-post identical flats with identical dimensions and asking prices.
- **Action**: Applied `df.drop_duplicates()`.
- **Impact**: Dataset reduced from 13,320 to **12,791 rows**. Prevents data leakage between future cross-validation folds.

---

### B. Feature Selection & Column Pruning
- **`society` (Dropped)**: 5,502 rows missing (41.31%). The remaining non-null entries were spread across 2,688 unique complex names (highest count was only 80 listings). Imputation would inject massive noise; one-hot encoding would cause severe sparsity.
- **`availability` (Dropped)**: 79.4% dominated by `"Ready To Move"`. The remaining values consisted of arbitrary historical possession dates (`18-May`, `19-Dec`) that provide negligible generalizable predictive signal for long-term valuation.
- **`area_type` (Retained)**: Kept because independent plotted developments (`Plot Area`) have fundamentally different valuation dynamics and land ownership rights compared to multi-story apartments (`Super built-up Area`).

---

### C. Handling Missing Values
- **Critical Identifiers (`size` & `location`)**:
  - Missing in 16 and 1 records respectively.
  - Dropped these 17 records. Property appraisal cannot proceed without bedroom configuration or neighborhood location.
- **`bath` (Imputed)**:
  - Contextually imputed using the **median bathroom count for that specific BHK tier** (e.g., 2 BHK $\rightarrow 2$ baths, 3 BHK $\rightarrow 3$ baths), rather than a naive flat city-wide average.
- **`balcony` (Imputed)**:
  - Imputed using the city-wide median ($2.0$).

---

### D. Square Footage Standardization (`total_sqft`)
- **Range Processing**: Values formatted as ranges (e.g., `'2100 - 2850'`) were parsed and converted to their arithmetic midpoint:
  $$\text{Midpoint} = \frac{2100 + 2850}{2} = 2475.0\text{ sq ft}$$
  *Rationale:* Builders and developers advertise minimum and maximum unit configurations for standard layouts; the arithmetic midpoint provides an unbiased expected area.
- **Metric & Traditional Unit Conversions**:
  - `Sq. Meter` $\rightarrow$ Multiplied by $10.7639$
  - `Sq. Yards` $\rightarrow$ Multiplied by $9.0$
  - `Acres` $\rightarrow$ Multiplied by $43,560.0$
  - `Guntha` $\rightarrow$ Multiplied by $1,089.0$
  - `Perch` $\rightarrow$ Multiplied by $272.25$
  - `Cents` $\rightarrow$ Multiplied by $435.56$
  - `Grounds` $\rightarrow$ Multiplied by $2,400.0$

---

### E. Location Cardinality Compression
- **Problem**: 1,305 raw unique locations created severe high cardinality (the curse of dimensionality).
- **Rule**: Standardized string casing and whitespace. Grouped all locations with **$\le 10$ listings** into a consolidated `"other"` category.
- **Result**: Compressed the location feature space from **1,305 down to 234 high-density categories**, preserving strong statistical support for each location while preventing model overfitting.

---

### F. Domain-Aware Outlier Removal

| Outlier Filter | Rule / Condition | Rows Filtered | Technical & Economic Justification |
| :--- | :--- | :---: | :--- |
| **1. Space Density** | $\frac{\text{total\_sqft}}{\text{bhk}} < 300$ | **738** | Under Indian municipal building codes, a standard residential bedroom with proportionate living space requires at least $300\text{ sq ft}$. Discards impossible records (e.g., 8 bedrooms on 600 sq ft). |
| **2. Bathroom Mismatch** | $\text{bath} > \text{bhk} + 2$ | **9** | Identifies severe anomalies (e.g., 2-BHK apartments listed with 5 to 40 bathrooms). |
| **3. Agricultural Land Floor** | $\text{price\_per\_sqft} < 1,000$ | **28** | Urban residential construction and land in Bengaluru costs well over ₹2,000/sqft. Properties listed at ₹2.25/sqft or ₹40/sqft represent vast rural agricultural farmland parcels (10 to 30 acres) mistakenly classified as residential homes. |
| **4. Locality $2\sigma$ Filter** | Outside $[\mu - 2\sigma, \mu + 2\sigma]$ per locality | **520** | Filters neighborhood-specific data entry typos while preserving 95.4% of the distribution. |

---

## 3. Luxury Property Preservation Audit

> [!IMPORTANT]
> **Zero Naive Pruning of High-Value Real Estate**:
> In naive statistical outlier removal (such as global IQR or global 3-sigma), legitimate multi-crore luxury properties are mistakenly deleted.
> 
> Because EstateIQ AI uses a **locality-aware 2-sigma filter**, prime neighborhoods with naturally high property values retain their authentic luxury inventory:
> - **Number of Luxury Properties ($\ge ₹300\text{ Lakhs}$)**: **530 listings preserved**
> - **Number of Ultra-Luxury Properties ($\ge ₹500\text{ Lakhs}$)**: **135 listings preserved**
> - **Highest Preserved Listing**: **₹2,200 Lakhs (₹22 Crores)** for an authentic palatial residence in an elite locality.
> - **Prime Locality Rate Retained**: Indira Nagar (₹30,241/sqft) and HAL 2nd Stage (₹27,083/sqft) are fully preserved.

---

## 4. Final Cleaned Dataset Schema & Validation Assertions

File location: [`data/processed/bengaluru_cleaned.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/processed/bengaluru_cleaned.csv)

| Column Name | Inferred Type | Non-Null Count | Allowed Range | Semantic Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`area_type`** | `object` | 11,479 | 4 categories | Super built-up Area, Built-up Area, Plot Area, Carpet Area |
| **`location`** | `object` | 11,479 | 234 categories | 233 primary Bengaluru localities + `"other"` |
| **`total_sqft`** | `float64` | 11,479 | $300.0 - 15,000.0$ | Physical area in square feet |
| **`bhk`** | `int64` | 11,479 | $1 - 16$ | Number of bedrooms |
| **`bath`** | `float64` | 11,479 | $1.0 - 16.0$ | Number of bathrooms |
| **`balcony`** | `float64` | 11,479 | $0.0 - 3.0$ | Number of balconies |
| **`price`** | `float64` | 11,479 | $₹9.0\text{ L} - ₹2,200.0\text{ L}$ | Target asking price in Lakhs INR |
| **`price_per_sqft`**| `float64` | 11,479 | $₹1,000 - ₹30,242$ | Diagnostic metric (Target leakage guarded) |

### Validation Assertions Passed:
- [x] Zero `NaN` or `null` values across all 8 features.
- [x] All physical area and price values are strictly positive ($>0$).
- [x] All bedroom and bathroom values are strictly valid ($>0$).
- [x] Exactly 11,479 high-quality rows generated.

