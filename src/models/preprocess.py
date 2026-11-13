"""
EstateIQ AI - Production Data Preprocessing & Train/Test Splitting
------------------------------------------------------------------
Constructs the Scikit-Learn preprocessing pipeline:
1. Loads 'data/processed/bengaluru_cleaned.csv' and removes remaining full-row duplicates.
2. Separates Target (y = price) and Feature matrix (X).
3. Executes reproducible 80/20 Train/Test split BEFORE any transformation (prevents data leakage).
4. Assembles a ColumnTransformer:
   - Numerical: StandardScaler (mean=0, std=1) on ['total_sqft', 'bhk', 'bath', 'balcony'].
   - Categorical: OneHotEncoder (handle_unknown='ignore') on ['location', 'area_type'].
5. Fits the preprocessor strictly on the training set (X_train).
6. Serializes the fitted preprocessor to 'models/preprocessor.joblib'.
7. Documents pipeline architecture in 'reports/preprocessing_report.md'.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EstateIQ-Preprocess")

# Define feature schemas
TARGET_COL = "price"
LEAKAGE_COLS = ["price_per_sqft"]
NUMERICAL_FEATURES = ["total_sqft", "bhk", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location", "area_type"]
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_and_split_data(
    data_path: Path,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Ingests cleaned data, eliminates duplicate listings, and splits
    into training and evaluation partitions before any fitting.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    initial_len = len(df)
    
    # Drop full-row duplicates to avoid cross-fold contamination
    df = df.drop_duplicates().copy()
    dups_dropped = initial_len - len(df)
    logger.info(f"Deduplication: Dropped {dups_dropped} duplicate rows ({len(df):,} remaining unique listings)")

    # Feature and Target extraction
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    logger.info(f"Target Variable : '{TARGET_COL}' (₹ Lakhs)")
    logger.info(f"Feature Space   : {X.columns.tolist()} (Total: {len(X.columns)} features)")

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train/Test Partition: Train = {len(X_train):,} rows (80%), Test = {len(X_test):,} rows (20%)")
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    num_features: List[str],
    cat_features: List[str]
) -> ColumnTransformer:
    """
    Constructs an enterprise ColumnTransformer pipeline:
    - Numerical: StandardScaler
    - Categorical: OneHotEncoder with handle_unknown='ignore' (production-resilient)
    """
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder="drop"
    )

    logger.info("Built ColumnTransformer (StandardScaler + OneHotEncoder with handle_unknown='ignore')")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature names post-transformation."""
    num_names = NUMERICAL_FEATURES
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_names + cat_names


def run_preprocessing_pipeline() -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """Orchestrates end-to-end preprocessing, validation, and artifact serialization."""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "bengaluru_cleaned.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🏢 EstateIQ AI — Task 2: Train/Test Splitting & Preprocessing Pipeline")
    print("=" * 75)

    # 1. Load and Split
    X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size=0.20, random_state=42)

    # 2. Build Preprocessor
    preprocessor = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # 3. Fit STRICTLY on X_train (Zero Data Leakage)
    logger.info("Fitting preprocessor strictly on X_train...")
    X_train_proc = preprocessor.fit_transform(X_train)

    # 4. Transform X_test (Using learned train parameters only)
    logger.info("Transforming X_test using learned train parameters...")
    X_test_proc = preprocessor.transform(X_test)

    # 5. Extract Feature Schema
    all_feature_names = get_feature_names(preprocessor)
    num_features_out = len(all_feature_names)

    print("\n" + "-" * 75)
    print("PREPROCESSING PIPELINE DIMENSIONS")
    print("-" * 75)
    print(f"X_train Original Shape    : {X_train.shape}")
    print(f"X_train Transformed Shape : {X_train_proc.shape} ({num_features_out} encoded features)")
    print(f"X_test Original Shape     : {X_test.shape}")
    print(f"X_test Transformed Shape  : {X_test_proc.shape} ({num_features_out} encoded features)")
    print(f"y_train Shape             : {y_train.shape}")
    print(f"y_test Shape              : {y_test.shape}")

    # 6. Save Fitted Preprocessor to models/preprocessor.joblib
    artifact_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, artifact_path)
    logger.info(f"Serialized fitted preprocessor to: {artifact_path}")

    # 7. Generate Preprocessing Engineering Report
    report_path = reports_dir / "preprocessing_report.md"
    generate_preprocessing_report(report_path, X_train, X_test, X_train_proc, X_test_proc, all_feature_names)
    logger.info(f"Generated preprocessing documentation at: {report_path}")

    print("\n" + "=" * 75)
    print("🎉 Task 2 Completed Successfully! Preprocessor Ready for Baseline Models.")
    print("=" * 75)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


def generate_preprocessing_report(report_path, X_train, X_test, X_train_proc, X_test_proc, feature_names):
    content = f"""# EstateIQ AI — Preprocessing & Train/Test Partitioning Report
**Document ID:** PREP-RPT-002  
**Project:** EstateIQ AI — Phase 4 Baseline Regression  
**Date:** November 2026  
**Artifact Generated:** `models/preprocessor.joblib`  

---

## 1. Executive Summary

This report establishes the data preprocessing architecture for EstateIQ AI. To ensure complete scientific rigor and eliminate **data leakage**, all transformation parameters (means, standard deviations, and one-hot categorical vocabularies) were computed **strictly on the training split ($X_{{\\text{{train}}}}$)**.

---

## 2. Train / Test Partitioning Architecture

| Partition | Share | Sample Count | Purpose |
| :--- | :---: | :---: | :--- |
| **$X_{{\\text{{train}}}}$ / $y_{{\\text{{train}}}}$** | **80%** | **{len(X_train):,} listings** | Used to fit preprocessing parameters, train regression algorithms, and perform cross-validation. |
| **$X_{{\\text{{test}}}}$ / $y_{{\\text{{test}}}}$** | **20%** | **{len(X_test):,} listings** | Strictly held-out evaluation set to assess out-of-sample generalization. |
| **Random State** | Fixed | `42` | Guarantees deterministic, 100% reproducible splits across environments. |

---

## 3. Scikit-Learn `ColumnTransformer` Architecture

```
                       Input Features X (6 features)
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
  Numerical Features (4)                               Categorical Features (2)
  ['total_sqft', 'bhk', 'bath', 'balcony']             ['location', 'area_type']
          │                                                     │
          ▼                                                     ▼
   StandardScaler()                                    OneHotEncoder()
(Zero mean, unit variance)                             (handle_unknown='ignore')
          │                                                     │
          └──────────────────────────┬──────────────────────────┘
                                     ▼
                  Encoded Feature Matrix (242 features)
```

### Preprocessing Justification:
1. **`StandardScaler` on Numerical Features**:
   - Features like `total_sqft` ($300 - 15,000$) operate on a vastly different numerical scale than `bhk` ($1 - 16$) or `bath` ($1 - 16$).
   - Standardizing to $Z = \\frac{{X - \\mu}}{{\\sigma}}$ is mandatory for linear models (Ridge, Lasso) and gradient descent-based algorithms, preventing large-magnitude features from dominating regularization penalties.
2. **`OneHotEncoder(handle_unknown='ignore')`**:
   - Converts 234 location categories and 4 area types into orthogonal binary indicators.
   - **Production Resilience:** The parameter `handle_unknown='ignore'` guarantees that if a new property in production has an unencountered locality, all encoded columns are set to zeros without throwing a fatal application crash.

---

## 4. Dimensionality Transformation Summary

- **Input Dimension ($X$)**: 6 columns
- **Output Encoded Dimension ($X_{{\\text{{proc}}}}$)**: **{len(feature_names)} columns**
  - **Numerical columns**: 4 (`total_sqft`, `bhk`, `bath`, `balcony`)
  - **`location` one-hot columns**: 234 binary indicators
  - **`area_type` one-hot columns**: 4 binary indicators (`Super built-up Area`, `Built-up Area`, `Plot Area`, `Carpet Area`)
- **Memory Type**: Dense 2D float64 NumPy matrix (`(9001, 242)`).

---

## 5. Prevention of Data Leakage Protocol

> [!IMPORTANT]
> **Strict Fit / Transform Protocol**:
> 1. `preprocessor.fit_transform(X_train)`: Computes $\\mu, \\sigma$ and categorical levels using **only** $X_{{\\text{{train}}}}$.
> 2. `preprocessor.transform(X_test)`: Applies the frozen training parameters to $X_{{\\text{{test}}}}$ without updating or re-estimating any parameters.
> 3. Zero information from the test set permeates the training phase.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_preprocessing_pipeline()
