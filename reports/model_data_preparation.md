# EstateIQ AI — Model Data Preparation & Feature Selection Report
**Document ID:** ML-PREP-001  
**Project:** EstateIQ AI — Baseline Regression Model Development (Phase 4)  
**Input Source:** `data/processed/bengaluru_cleaned.csv`  
**Total Records:** 11,479 listings  
**Total Features in Processed File:** 8 columns  

---

## 1. Problem Formulation & Prediction Objective

- **Task Type**: Supervised Continuous Regression ($f: X \rightarrow y$)
- **Target Variable ($y$)**: `price` (Continuous asking price quoted in **₹ Lakhs INR**)
- **Prediction Objective**: Estimate the fair market asking price of a residential property in metropolitan Bengaluru given physical and location attributes provided by a user at appraisal time.

---

## 2. Feature Space Architecture & Selection Rationale

| Column Name | Inferred Dtype | Status | Feature Role | Technical Rationale & Inclusion Justification |
| :--- | :---: | :---: | :---: | :--- |
| **`price`** | `float64` | **TARGET** | Ground Truth ($y$) | The continuous variable the regression models are trained to predict. |
| **`location`** | `object (str)`| **INCLUDED** | Categorical Feature ($X$) | Primary spatial driver of land value. Encoded as 234 high-density categories (233 localities + `"other"`). |
| **`total_sqft`**| `float64` | **INCLUDED** | Numerical Feature ($X$) | Primary physical scale metric. Has the highest linear correlation with price ($r = 0.58$). |
| **`bhk`** | `int64` | **INCLUDED** | Numerical Feature ($X$) | Room capacity metric representing property tier and buyer segment. |
| **`bath`** | `float64` | **INCLUDED** | Numerical Feature ($X$) | Indicator of layout standard and luxury accommodation. |
| **`balcony`** | `float64` | **INCLUDED** | Numerical Feature ($X$) | Secondary amenity convenience indicator ($0, 1, 2, 3$). |
| **`area_type`** | `object (str)`| **INCLUDED** | Categorical Feature ($X$) | Distinguishes plotted land/villas (`Plot Area`) from multi-story apartment flats (`Super built-up Area`). |
| **`price_per_sqft`**| `float64` | **EXCLUDED** | **TARGET LEAKAGE** | **Quarantined**. Calculated directly from `price` and `total_sqft`. Including this feature would create artificial $R^2 = 1.0$ and catastrophic failure at inference time. |

---

## 3. Strict Target Leakage Safeguard Protocol

> [!CAUTION]
> **Why `price_per_sqft` is Strict Target Leakage**:
> - Formula: $\text{price\_per\_sqft} = \frac{\text{price} \times 100,000}{\text{total\_sqft}}$
> - If `price_per_sqft` is placed into the training matrix $X$, the machine learning model does not learn real estate economics; it simply learns the trivial algebraic equation:
>   $$\text{price} = \frac{\text{price\_per\_sqft} \times \text{total\_sqft}}{100,000}$$
> - In production, when a user asks EstateIQ AI to appraise an unlisted property, **the price is unknown**, meaning `price_per_sqft` is also unknown.
> - **Enforced Decision:** `price_per_sqft` is strictly isolated as a diagnostic metric and is **never** passed into `scikit-learn` estimators.

---

## 4. Inference-Time Feature Availability Check

To ensure that EstateIQ AI operates cleanly in the Streamlit web application (`app.py`), every feature in $X$ must be effortlessly provided by an end user:

| Input Feature | How User Inputs It in Web UI | Input Type |
| :--- | :--- | :--- |
| `location` | Dropdown selection from 234 validated localities | Selectbox |
| `total_sqft` | Total built-up area in square feet | Number Input ($300 - 15,000$) |
| `bhk` | Number of bedrooms | Slider / Stepper ($1 - 10$) |
| `bath` | Number of bathrooms | Slider / Stepper ($1 - 10$) |
| `balcony` | Number of balconies | Slider / Radio ($0 - 3$) |
| `area_type` | Structural layout type | Selectbox (`Super built-up`, `Plot Area`, etc.) |

*Result:* **100% of the selected input features are naturally available at inference time.**

---

## 5. Duplicate Records & Data Cleaning Audit Before Split

- **Exact Full-Row Duplicates Remaining**: `227` records.
- **Action for Next Step**: Drop these 227 duplicate records immediately prior to the Train/Test split to ensure zero test-fold contamination.
- **Total Modeling Dataset**: 11,252 clean property listings.

---

## 6. Preprocessing Blueprint for Next Step

```
                 CLEANED DATASET (11,252 rows)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Train Set (80%)                       Test Set (20%)
     (Stratified / Shuffled)               (Strictly Isolated)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              ColumnTransformer Pipeline                │
├──────────────────────────┬─────────────────────────────┤
│   Numerical Features     │    Categorical Features     │
│ [total_sqft, bhk, bath]  │    [location, area_type]    │
│           │              │              │              │
│     StandardScaler       │      OneHotEncoder          │
│                          │  (handle_unknown='ignore')  │
└──────────────────────────┴─────────────────────────────┘
```
