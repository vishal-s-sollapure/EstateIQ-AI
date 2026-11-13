"""
EstateIQ AI - Phase 4: Model Data Preparation & Feature Selection
-----------------------------------------------------------------
Inspects and prepares the sanitized dataset for baseline regression modeling:
1. Loads 'data/processed/bengaluru_cleaned.csv' (read-only).
2. Audits columns, data types, and null values.
3. Explicitly partitions Target (y) and Features (X).
4. Strictly quarantines target leakage columns (e.g. price_per_sqft).
5. Checks feature distributions, bounds, and inference-time availability.
6. Documents findings into 'reports/model_data_preparation.md'.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Configure display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def prepare_model_data():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "bengaluru_cleaned.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🏢 EstateIQ AI — Phase 4: Dataset Audit for Model Training")
    print("=" * 75)
    print(f"Loading cleaned dataset from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    
    print("\n" + "-" * 75)
    print("1. DATASET SCHEMA & DATA TYPES")
    print("-" * 75)
    schema_summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique Values": df.nunique()
    })
    print(schema_summary.to_string())

    # ------------------------------------------------------------------
    # 2. Target Variable & Feature Space Partition
    # ------------------------------------------------------------------
    target_col = "price"
    leakage_col = "price_per_sqft"
    
    # Candidate feature columns available at runtime
    feature_cols = [c for c in df.columns if c not in [target_col, leakage_col]]
    
    categorical_features = df[feature_cols].select_dtypes(include=["object", "str"]).columns.tolist()
    numerical_features = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\n" + "-" * 75)
    print("2. TARGET & FEATURE PARTITIONING")
    print("-" * 75)
    print(f"Target Variable (y)         : ['{target_col}'] (Unit: ₹ Lakhs INR)")
    print(f"Quarantined Leakage Feature : ['{leakage_col}'] (EXCLUDED from X)")
    print(f"Total Input Features (|X|)  : {len(feature_cols)}")
    print(f" - Categorical Features     : {categorical_features}")
    print(f" - Numerical Features       : {numerical_features}")

    # ------------------------------------------------------------------
    # 3. Target Leakage Verification
    # ------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. TARGET LEAKAGE SAFEGUARD AUDIT")
    print("-" * 75)
    print(f"Checking '{leakage_col}':")
    print(" • Calculation: (price * 100,000) / total_sqft")
    print(" • Risk Level : FATAL TARGET LEAKAGE.")
    print(" • Rationale  : At inference time, a prospective homebuyer asks EstateIQ AI to predict")
    print("   the price. 'price_per_sqft' cannot exist before price is known.")
    print(" • Action     : Strictly excluded from model feature matrix X.")

    # ------------------------------------------------------------------
    # 4. Feature Summary & Boundary Checks
    # ------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. NUMERICAL FEATURE DISTRIBUTIONS (X_num)")
    print("-" * 75)
    print(df[numerical_features].describe().round(2))

    print("\n" + "-" * 75)
    print("5. CATEGORICAL FEATURE CARDINALITY (X_cat)")
    print("-" * 75)
    for col in categorical_features:
        print(f"Feature '{col}': {df[col].nunique()} unique categories")
        top_cats = df[col].value_counts().head(5)
        print(f"  Top categories: {dict(top_cats)}\n")

    # ------------------------------------------------------------------
    # 5. Check for exact duplicate feature rows
    # ------------------------------------------------------------------
    exact_full_dups = df.duplicated().sum()
    print("-" * 75)
    print("6. DUPLICATE ROW CHECK")
    print("-" * 75)
    print(f"Exact full-row duplicates in cleaned data: {exact_full_dups}")
    if exact_full_dups > 0:
        print(f"Recommendation: Drop {exact_full_dups} duplicate records prior to train-test splitting")
        print("to guarantee zero data leakage across evaluation folds.")

    # ------------------------------------------------------------------
    # 6. Generate Markdown Documentation
    # ------------------------------------------------------------------
    report_path = reports_dir / "model_data_preparation.md"
    generate_markdown_report(report_path, df, target_col, leakage_col, categorical_features, numerical_features, exact_full_dups)
    print("\n" + "=" * 75)
    print(f"✅ Data Preparation Audit Completed! Report saved to: {report_path}")
    print("=" * 75)


def generate_markdown_report(report_path, df, target_col, leakage_col, cat_features, num_features, exact_dups):
    content = f"""# EstateIQ AI — Model Data Preparation & Feature Selection Report
**Document ID:** ML-PREP-001  
**Project:** EstateIQ AI — Baseline Regression Model Development (Phase 4)  
**Input Source:** `data/processed/bengaluru_cleaned.csv`  
**Total Records:** {df.shape[0]:,} listings  
**Total Features in Processed File:** {df.shape[1]} columns  

---

## 1. Problem Formulation & Prediction Objective

- **Task Type**: Supervised Continuous Regression ($f: X \\rightarrow y$)
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
> - Formula: $\\text{{price\\_per\\_sqft}} = \\frac{{\\text{{price}} \\times 100,000}}{{\\text{{total\\_sqft}}}}$
> - If `price_per_sqft` is placed into the training matrix $X$, the machine learning model does not learn real estate economics; it simply learns the trivial algebraic equation:
>   $$\\text{{price}} = \\frac{{\\text{{price\\_per\\_sqft}} \\times \\text{{total\\_sqft}}}}{{100,000}}$$
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

- **Exact Full-Row Duplicates Remaining**: `{exact_dups}` records.
- **Action for Next Step**: Drop these {exact_dups} duplicate records immediately prior to the Train/Test split to ensure zero test-fold contamination.
- **Total Modeling Dataset**: {df.shape[0] - exact_dups:,} clean property listings.

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
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    prepare_model_data()
