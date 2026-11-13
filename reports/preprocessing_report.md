# EstateIQ AI — Preprocessing & Train/Test Partitioning Report
**Document ID:** PREP-RPT-002  
**Project:** EstateIQ AI — Phase 4 Baseline Regression  
**Date:** November 2026  
**Artifact Generated:** `models/preprocessor.joblib`  

---

## 1. Executive Summary

This report establishes the data preprocessing architecture for EstateIQ AI. To ensure complete scientific rigor and eliminate **data leakage**, all transformation parameters (means, standard deviations, and one-hot categorical vocabularies) were computed **strictly on the training split ($X_{\text{train}}$)**.

---

## 2. Train / Test Partitioning Architecture

| Partition | Share | Sample Count | Purpose |
| :--- | :---: | :---: | :--- |
| **$X_{\text{train}}$ / $y_{\text{train}}$** | **80%** | **9,001 listings** | Used to fit preprocessing parameters, train regression algorithms, and perform cross-validation. |
| **$X_{\text{test}}$ / $y_{\text{test}}$** | **20%** | **2,251 listings** | Strictly held-out evaluation set to assess out-of-sample generalization. |
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
   - Standardizing to $Z = \frac{X - \mu}{\sigma}$ is mandatory for linear models (Ridge, Lasso) and gradient descent-based algorithms, preventing large-magnitude features from dominating regularization penalties.
2. **`OneHotEncoder(handle_unknown='ignore')`**:
   - Converts 234 location categories and 4 area types into orthogonal binary indicators.
   - **Production Resilience:** The parameter `handle_unknown='ignore'` guarantees that if a new property in production has an unencountered locality, all encoded columns are set to zeros without throwing a fatal application crash.

---

## 4. Dimensionality Transformation Summary

- **Input Dimension ($X$)**: 6 columns
- **Output Encoded Dimension ($X_{\text{proc}}$)**: **242 columns**
  - **Numerical columns**: 4 (`total_sqft`, `bhk`, `bath`, `balcony`)
  - **`location` one-hot columns**: 234 binary indicators
  - **`area_type` one-hot columns**: 4 binary indicators (`Super built-up Area`, `Built-up Area`, `Plot Area`, `Carpet Area`)
- **Memory Type**: Dense 2D float64 NumPy matrix (`(9001, 242)`).

---

## 5. Prevention of Data Leakage Protocol

> [!IMPORTANT]
> **Strict Fit / Transform Protocol**:
> 1. `preprocessor.fit_transform(X_train)`: Computes $\mu, \sigma$ and categorical levels using **only** $X_{\text{train}}$.
> 2. `preprocessor.transform(X_test)`: Applies the frozen training parameters to $X_{\text{test}}$ without updating or re-estimating any parameters.
> 3. Zero information from the test set permeates the training phase.
