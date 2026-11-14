# EstateIQ AI — Phase 5 Model Readiness Audit Report
**Document ID:** AUDIT-PHASE5-001  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Auditor:** Senior Machine Learning Engineer  
**Audit Date:** November 14, 2026  
**Status:** Complete — Empirical Audit (Read-Only)  

---

## 1. Executive Summary & Readiness Verdict

### 🚦 Readiness Status: ⚠️ CONDITIONALLY READY (BLOCKED ON DEPENDENCY INSTALLATION)

An empirical audit of the Phase 4 baseline modeling artifacts, data contract, preprocessing pipeline, and local runtime environment was performed. 

**Summary of Findings:**
- The data contract, train/test split, and baseline evaluation are **scientifically sound** with **zero data leakage** and **zero test set contamination**.
- The baseline benchmark is established: **Random Forest achieved Test $R^2 = 0.6634$** (MAE = ₹25.95 Lakhs).
- **Blocker for Phase 5:** Advanced libraries (`lightgbm`, `xgboost`, `catboost`, `shap`) are **not currently installed** in the virtual environment.
- **Architectural Enhancements Identified:** The preprocessing pipeline needs `SimpleImputer` for inference-time robustness, and models must incorporate target transformation ($\log(\text{price})$) to break the $R^2 > 0.75$ barrier.

---

## 2. Phase 4 Modeling Artifacts Inventory

| File Path | Artifact Type | Status | Role in Project |
| :--- | :---: | :---: | :--- |
| [`data/processed/bengaluru_cleaned.csv`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/data/processed/bengaluru_cleaned.csv) | Processed Data | Verified | Cleaned tabular dataset (11,479 rows, 8 columns). |
| [`src/models/prepare_data.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/models/prepare_data.py) | Source Code | Verified | Feature selection, schema inspection, and target leakage quarantine. |
| [`src/models/preprocess.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/models/preprocess.py) | Source Code | Verified | 80/20 train/test splitting and Scikit-Learn `ColumnTransformer` builder. |
| [`src/models/train_baseline.py`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/src/models/train_baseline.py) | Source Code | Verified | 5-Fold CV and test evaluation of OLS, Ridge, Lasso, and Random Forest. |
| [`models/preprocessor.joblib`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/models/preprocessor.joblib) | Binary Model | Verified | Fitted `ColumnTransformer` artifact (6,406 bytes). |
| [`models/estateiq_baseline_model.joblib`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/models/estateiq_baseline_model.joblib) | Binary Model | Verified | Fitted Random Forest Regressor pipeline (~69.6 MB). |
| [`reports/model_data_preparation.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/model_data_preparation.md) | Documentation | Verified | Feature selection report and leakage quarantine protocol. |
| [`reports/preprocessing_report.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/preprocessing_report.md) | Documentation | Verified | Preprocessor architecture and dimensional transformation report. |
| [`reports/baseline_models_benchmark.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/baseline_models_benchmark.md) | Documentation | Verified | Formal benchmark metrics across all baseline algorithms. |
| [`reports/figures/04_baseline_model_performance.png`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/figures/04_baseline_model_performance.png) | Visual Figure | Verified | Actual vs. Predicted scatter and residual error distribution. |

---

## 3. Ingestion & Schema Audit (`bengaluru_cleaned.csv`)

- **Total Rows**: Exactly **11,479 listings**
- **Total Columns**: **8 columns**
- **Missing Values**: **0 nulls** across all columns.
- **Duplicate Records**: **227 exact full-row duplicates** present in the CSV file (arising from dropping `society` and `availability` in Phase 3).
  - *Mitigation in Phase 4*: Handled in `preprocess.py` by dropping these 227 rows, yielding **11,252 unique listings**.
- **Target Variable**: **`price`**
  - **Unit**: Indian Rupees in Lakhs ($\text{₹ Lakhs}$), where $1\text{ Lakh} = 100,000\text{ INR}$.
  - **Range**: ₹9.00 Lakhs to ₹2,200.00 Lakhs.
  - **Distribution**: Mean = ₹100.84 Lakhs, Median = ₹69.00 Lakhs, Standard Deviation = ₹110.69 Lakhs (heavily right-skewed).

---

## 4. Feature Space & Target Leakage Audit

### A. Feature Space Partitioning
- **Input Features ($X$)**: 6 candidate features:
  - **Numerical Features (4)**: `total_sqft`, `bhk`, `bath`, `balcony`
  - **Categorical Features (2)**: `location` (234 categories), `area_type` (4 categories)
- **Target Feature ($y$)**: `price`

### B. Target Leakage Verification
- **Feature Checked**: `price_per_sqft`
- **Audit Result**: **PASSED ✅**
- **Evidence**: In `src/models/preprocess.py`, `FEATURE_COLS` explicitly excludes `price_per_sqft`. Only the 6 legitimate physical and location attributes enter the pipeline.
- **Inference Availability**: All 6 features are easily provided by a prospective homebuyer via UI dropdowns and sliders.

---

## 5. Preprocessing Pipeline Capabilities & Gap Analysis

| Preprocessing Function | Current Status | Current Implementation | Gap / Production Risk for Phase 5 |
| :--- | :---: | :--- | :--- |
| **Numerical Scaling** | **Implemented** ✅ | `StandardScaler()` | None for linear models; trees do not strictly need scaling but are unaffected. |
| **Categorical Encoding** | **Implemented** ✅ | `OneHotEncoder(sparse_output=False)` | Expands to 242 sparse columns. Sub-optimal for gradient boosters (LightGBM/CatBoost prefer native categorical binning). |
| **Unseen Locations** | **Implemented** ✅ | `handle_unknown='ignore'` | Safely sets one-hot columns to zeros if an unlisted locality is supplied. |
| **Missing Numerical Values**| **Missing** ⚠️ | None (Assumes 0 nulls) | **Risk:** If an inference request omits `bath` or `balcony`, the pipeline crashes with a `NaN` error. Needs `SimpleImputer(strategy='median')`. |
| **Missing Categorical Values**| **Missing** ⚠️ | None (Assumes 0 nulls) | **Risk:** Needs `SimpleImputer(strategy='constant', fill_value='other')` for production inference resilience. |
| **Target Transformation** | **Missing** ⚠️ | Raw linear price target ($y$) | **Risk:** High heteroscedasticity limits model accuracy. Needs $\log(1 + y)$ target scaling to reach $R^2 > 0.75$. |

---

## 6. Baseline Model Benchmark Results (Phase 4 Evidence)

Extracted empirically from [`reports/baseline_models_benchmark.md`](file:///c:/Users/VISHAL/OneDrive/Desktop/EstateIQ-AI/reports/baseline_models_benchmark.md):

| Model Architecture | Train $R^2$ | 5-Fold CV $R^2$ (Train) | Test $R^2$ (Out-of-Sample) | Test RMSE (₹ Lakhs) | Test MAE (₹ Lakhs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | $0.7226$ | $0.6879 \pm 0.065$ | $0.5802$ | ₹$79.26\text{ L}$ | ₹$29.90\text{ L}$ |
| **Ridge Regression ($\alpha=1.0$)** | $0.7221$ | $0.6874 \pm 0.065$ | $0.5796$ | ₹$79.32\text{ L}$ | ₹$29.84\text{ L}$ |
| **Lasso Regression ($\alpha=0.1$)** | $0.7011$ | $0.6683 \pm 0.067$ | $0.5599$ | ₹$81.16\text{ L}$ | ₹$31.47\text{ L}$ |
| **🥇 Random Forest ($n=100$)** | **$0.9595$** | **$0.6870 \pm 0.074$** | **$0.6634$** | **₹$70.98\text{ L}$** | **₹$25.95\text{ L}$** |

### Test Set Integrity Check:
- **Partitioning**: 80% Train ($9,001\text{ rows}$) / 20% Test ($2,251\text{ rows}$).
- **Leakage Check**: **PASSED ✅**. In `src/models/train_baseline.py`, 5-fold cross-validation was strictly executed on `X_train`. The test set was touched **only once** for final metric reporting. It was not used for tuning $\alpha$, tree depth, or feature selection.

---

## 7. Advanced Libraries Installation Audit

Audited via `pip list` in active virtual environment (`venv`):

| Target Library | Installed? | Current Version | Required for Phase 5 |
| :--- | :---: | :---: | :--- |
| **`lightgbm`** | **NO** ❌ | None | High-speed gradient boosting with leaf-wise tree growth. |
| **`xgboost`** | **NO** ❌ | None | Scalable, regularized gradient boosted decision trees. |
| **`catboost`** | **NO** ❌ | None | Gradient boosting optimized for categorical features. |
| **`shap`** | **NO** ❌ | None | Game-theoretic Explainable AI (XAI) feature attribution. |

---

## 8. Required Corrections Before Phase 5 Training

1. **Install Advanced Dependencies**:
   Install `lightgbm`, `xgboost`, `catboost`, and `shap`.
2. **Defensive Imputation**:
   Wrap numerical features in `SimpleImputer(strategy='median')` and categorical features in `SimpleImputer(strategy='constant', fill_value='other')` before scalers/encoders to guarantee zero runtime failures during user inference.
3. **Log-Target Transformation**:
   Wrap estimators in Scikit-Learn's `TransformedTargetRegressor(regressor, func=np.log1p, inverse_func=np.expm1)`. This models $\log(\text{price})$ to stabilize error variance and directly tackles heteroscedasticity, which is essential to drive $R^2$ above $0.75$.
4. **Target Encoding / Native Categorical Support**:
   For LightGBM and CatBoost, explore native categorical handling or Target Encoding to avoid exploding tree depth on 234 one-hot columns.

---

## 9. Recommended Next Implementation Step

- **Step 02**: Install the Phase 5 packages (`lightgbm`, `xgboost`, `catboost`, `shap`) in `venv`, run unblock checks on Windows 11, and verify successful imports.

