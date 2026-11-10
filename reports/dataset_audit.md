# EstateIQ AI — Phase 2: Exploratory Data Analysis & Dataset Audit Report
**Document ID:** EDA-RPT-001  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Target Dataset:** `data/raw/Bengaluru_House_Data.csv`  
**Auditor:** Senior Data Scientist & ML Engineer  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

A comprehensive, empirical data audit of the **Bengaluru House Price dataset** was conducted without modifying the underlying raw data. The dataset contains **13,320 property records** across **9 features**.

The audit uncovered critical structural anomalies, extreme outliers, non-standard unit representations, and severe missingness in specific columns. This report establishes the empirical foundation for our upcoming data cleaning, feature engineering, and geospatial intelligence pipelines.

---

## 2. Dataset Dimensions & High-Level Metadata

| Metric | Value | Data Science Interpretation |
| :--- | :---: | :--- |
| **Total Rows (Samples)** | **13,320** | Sufficient sample size for supervised tabular regression. |
| **Total Columns (Features)**| **9** | Compact initial feature space. |
| **Raw Disk Footprint** | **0.88 MB** | Fits comfortably in local memory for vectorized Pandas operations. |
| **Memory Usage in RAM** | **~1.6 MB** | High processing efficiency. |
| **Target Variable ($y$)** | **`price`** | Continuous numerical target quoted in **₹ Lakhs INR** ($1\text{ Lakh} = 100,000\text{ INR}$). |

---

## 3. Structural Data Type Audit

| Column Name | Inferred Dtype | Intended Semantic Type | Structural Status / Red Flags |
| :--- | :---: | :---: | :--- |
| **`area_type`** | `str` / `object` | Nominal Categorical | Clean 4 categories (`Super built-up Area`, `Built-up Area`, `Plot Area`, `Carpet Area`). |
| **`availability`** | `str` / `object` | Mixed Categorical / Temporal | Dominated by `"Ready To Move"`; sparse arbitrary future completion dates. |
| **`location`** | `str` / `object` | High-Cardinality Categorical | **1,305 unique values**; contains trailing spaces, casing inconsistencies, and rare entries. |
| **`size`** | `str` / `object` | Discrete Numerical (`bhk`) | Text strings combining numbers and room descriptions (`"2 BHK"`, `"4 Bedroom"`). |
| **`society`** | `str` / `object` | Nominal Categorical | **Severe missingness (>41%)**; high sparsity (2,688 unique complex names). |
| **`total_sqft`** | `str` / `object` | Continuous Numerical (`float64`) | **Critical Anomaly**: Contains string ranges (`"2100 - 2850"`) and metric units (`"Sq. Meter"`). |
| **`bath`** | `float64` | Discrete Numerical | Floated due to NaNs; contains extreme physical outliers ($40\text{ bathrooms}$). |
| **`balcony`** | `float64` | Discrete Numerical | Floated due to NaNs; valid range from $0$ to $3$. |
| **`price`** | `float64` | Continuous Numerical | Target variable; zero missingness; heavily right-skewed. |

---

## 4. Missing Value Quantification

```
Missing Values Breakdown:
society       ████████████████████████████████ 41.31% (5,502 missing)
balcony       ███ 4.57% (609 missing)
bath          ▏ 0.55% (73 missing)
size          ▏ 0.12% (16 missing)
location      ▏ 0.01% (1 missing)
area_type     0.00%
availability  0.00%
total_sqft    0.00%
price         0.00%
```

### Detailed Breakdown:
| Feature | Non-Null Count | Null Count | Null % | Data Science Strategy & Treatment |
| :--- | :---: | :---: | :---: | :--- |
| **`society`** | 7,818 | **5,502** | **41.31%** | **Drop feature**. Over 41% missing; remaining entries have 2,688 sparse unique societies. Attempting imputation will introduce severe synthetic noise. |
| **`balcony`** | 12,711 | **609** | **4.57%** | **Impute with median ($2$)** or treat as optional amenity flag (`has_balcony`). |
| **`bath`** | 13,247 | **73** | **0.55%** | **Impute with median ($2$)** or drop the 73 rows (minimal data loss of <0.6%). |
| **`size`** | 13,304 | **16** | **0.12%** | **Drop the 16 rows**. Bedroom count is fundamental; 16 rows represents negligible loss. |
| **`location`**| 13,319 | **1** | **0.01%** | **Drop the single unlabelled row**. Real estate cannot be valued without location. |
| **`total_sqft`**| 13,320 | **0** | **0.00%** | Zero nulls, but contains format inconsistencies requiring parsing. |
| **`price`** | 13,320 | **0** | **0.00%** | Ground truth target is 100% complete. |

---

## 5. Duplicate Records Analysis

- **Exact Duplicate Count**: **529 records** (**3.97%** of the dataset).
- **Domain Root Cause**: In Indian real estate aggregators, identical apartments or plots are frequently cross-posted by multiple brokerage agencies with identical square footage, BHK, and asking prices.
- **Engineering Recommendation**: Drop exact duplicate rows during data cleaning. Retaining identical listings inflates weights and risks data leakage between training and validation folds.

---

## 6. Numerical & Statistical Profiling

```
                     bath        balcony          price (₹ Lakhs)
Count           13,247.00      12,711.00                13,320.00
Mean                 2.69           1.58                   112.57
Std Dev              1.34           0.82                   148.97
Min                  1.00           0.00                     8.00
25% (Q1)             2.00           1.00                    50.00
50% (Median)         2.00           2.00                    72.00
75% (Q3)             3.00           2.00                   120.00
Max                 40.00           3.00                 3,600.00
IQR                  1.00           1.00                    70.00
```

### Statistical Observations:
1. **Target Skewness (`price`)**:
   - **Median = ₹72 Lakhs**, while **Mean = ₹112.57 Lakhs**.
   - The substantial divergence between mean and median indicates **extreme right-skewness** driven by multi-crore luxury listings (reaching up to ₹36 Crores).
   - *Modeling Impact:* Linear models will suffer from heteroscedasticity; a log-transform ($\log(1 + y)$) or tree-based ensembles (LightGBM/XGBoost) are recommended.
2. **Extreme Bathroom Anomaly (`bath = 40.0`)**:
   - The maximum bathroom count is **40**, which is physically implausible for a residential unit. This represents either a commercial hostel/PG misclassified as residential, or a data-entry typo.
3. **Balcony Bounds**:
   - Discrete, orderly integer distribution ranging from 0 to 3.

---

## 7. Categorical Cardinality & Distributions

| Column | Unique Categories | Top Category | Top Frequency | Frequency % | Data Science Verdict |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`area_type`** | 4 | `Super built-up Area` | 8,790 | 66.0% | Clean, low cardinality. One-Hot Encode. |
| **`availability`**| 81 | `Ready To Move` | 10,581 | 79.4% | Highly skewed. Low predictive power; recommend dropping. |
| **`location`** | **1,305** | `Whitefield` | 540 | 4.1% | **High Cardinality**. Must group locations with $\le 10$ records into `"other"`. |
| **`size`** | 31 | `2 BHK` | 5,199 | 39.0% | Must parse into numerical `bhk` integer. |
| **`society`** | 2,688 | `GrrvaGr` | 80 | 0.6% | Severe sparsity + 41% missing. Drop. |
| **`total_sqft`** | 2,117 | `1200` | 843 | 6.3% | Parse numeric ranges and convert to float. |

---

## 8. Deep-Dive Anomaly Catalogue

### Anomaly 1: Non-Numeric Formatting in `total_sqft`
- **Impacted Rows**: $\approx 247$ records.
- **Pattern A (Ranges)**: Strings like `'2100 - 2850'`, `'1015 - 1540'`.
  - *Fix:* Parse tokens and compute the arithmetic midpoint: $\frac{2100 + 2850}{2} = 2475.0\text{ sq ft}$.
- **Pattern B (Non-Standard Units)**: Strings like `'34.46Sq. Meter'`, `'4125Perch'`, `'1Acres'`, `'24Guntha'`.
  - *Fix:* Safely drop unconvertible non-standard unit records to preserve numerical integrity.

### Anomaly 2: Extreme Bedroom Counts (`bhk > 10`)
- Records exist reporting **27 BHK** and **43 Bedroom** configurations:
  - *Example:* A 43-bedroom property listed on a 2,400 sq. ft. plot. This translates to less than $55\text{ sq ft}$ per bedroom, which violates physical residential building codes.
  - *Fix:* Filter out unrealistic residential densities.

### Anomaly 3: Bathroom vs. Bedroom Discrepancy
- The dataset contains listings where the bathroom count exceeds the bedroom count by more than 2:
  - *Example:* A 2-BHK apartment with 5 bathrooms.
  - *Fix:* Enforce domain rule: $\text{bath} \le \text{bhk} + 2$.

### Anomaly 4: Extreme Space Density ($\text{total\_sqft} / \text{bhk} < 300$)
- In Indian residential architectural norms, a standard bedroom (including proportionate living/kitchen area) requires **at least 300 sq. ft.**.
- Listings showing a 600 sq. ft. flat with 6 bedrooms represent clear data corruption.
- *Fix:* Enforce threshold $\frac{\text{total\_sqft}}{\text{bhk}} \ge 300$.

---

## 9. Feature Selection Matrix for Downstream Modeling

| Feature | Action | Category in Model | Technical Rationale |
| :--- | :---: | :---: | :--- |
| **`location`** | **Keep** | Nominal Categorical | Prime spatial determinant of real estate value. Cleaned via `"other"` consolidation. |
| **`total_sqft`** | **Keep** | Continuous Numerical | Primary physical scale metric. Converted to float. |
| **`bhk`** (from `size`) | **Keep** | Discrete Numerical | Capacity and demographic tier indicator. |
| **`bath`** | **Keep** | Discrete Numerical | Luxury and layout tier indicator. |
| **`balcony`** | **Keep** | Discrete Numerical | Secondary convenience amenity indicator. |
| **`area_type`** | **Keep** | Categorical One-Hot | Distinguishes plotted developments from high-rise societies. |
| **`price`** | **TARGET ($y$)** | Continuous Target | Ground truth valuation (in Lakhs INR). |
| **`society`** | **Drop** | Discarded | $>41\%$ missingness and high sparsity (2,688 unique values). |
| **`availability`** | **Drop** | Discarded | $>79\%$ `"Ready To Move"`; sparse arbitrary completion dates add noise. |

---

## 10. Senior Data Scientist Action Plan for Data Cleaning (Next Step)

```
[Raw Bengaluru Data (13,320 rows)]
               │
               ▼
       1. Drop 'society' & 'availability'
               │
               ▼
       2. Drop exact duplicate rows (529 rows removed)
               │
               ▼
       3. Extract clean integer 'bhk' from 'size'
               │
               ▼
       4. Parse 'total_sqft' (midpoints for ranges, drop invalid text)
               │
               ▼
       5. Clean 'location' strings & group infrequent (<= 10) into 'other'
               │
               ▼
       6. Apply Domain Outlier Filters:
          • total_sqft / bhk >= 300
          • bath <= bhk + 2
          • Remove statistical price_per_sqft outliers per location (mean ± 1 std dev)
               │
               ▼
[Cleaned Processed Dataset (data/processed/bengaluru_cleaned.csv)]
```

*Report compiled and verified for EstateIQ AI.*

