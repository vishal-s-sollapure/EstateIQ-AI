# EstateIQ AI — Visual Exploratory Data Analysis (EDA) Findings
**Document ID:** EDA-RPT-002  
**Project:** EstateIQ AI — AI-Powered Real Estate Valuation & Investment Intelligence Platform  
**Target Dataset:** `data/raw/Bengaluru_House_Data.csv`  
**Figures Generated:**
- [`reports/figures/01_price_distribution.png`](figures/01_price_distribution.png)
- [`reports/figures/02_bhk_and_area_analysis.png`](figures/02_bhk_and_area_analysis.png)
- [`reports/figures/03_locality_and_correlation.png`](figures/03_locality_and_correlation.png)

---

## 1. Executive Summary

This report documents the empirical findings from Visual Exploratory Data Analysis (EDA) conducted across the 13,320 historical Bengaluru property listings. Visualizations were constructed without permanently altering the original raw dataset.

---

## 2. Key Diagnostic Findings

### A. Target Variable Distribution (`price`)
- **Raw Price Skewness: 8.06** (Extremely right-skewed).
  - **Median = ₹72.0 Lakhs**
  - **Mean = ₹112.6 Lakhs**
  - **Variance Impact**: The top 5% of luxury listings pull the arithmetic mean ₹40.6 Lakhs above the median.
- **Log-Transformed Price Skewness: 0.83** (Near-Gaussian normality).
  - Applying $\log(y)$ compresses extreme variance and stabilizes residual errors.
  - *Recommendation:* Model targets using log-transformed values or use non-parametric tree ensembles (LightGBM, XGBoost, CatBoost).

### B. Unit Economics by Room Tier (Price per Sq. Ft. vs. BHK)
- **1 BHK & 2 BHK**: Highly stabilized pricing cluster (Median $\approx ₹4,800 - ₹5,100/\text{sq ft}$). Tight interquartile range (IQR).
- **3 BHK**: Steady stepping stone (Median $\approx ₹5,700/\text{sq ft}$).
- **4 BHK, 5 BHK & 6 BHK**: Sharp jump in price per square foot (Median jumps to **₹8,500 - ₹11,000/sq ft**).
  - *Economic Insight:* In Bengaluru, large configurations (4+ BHK) are overwhelmingly independent villas or luxury penthouses in gated communities, which carry a significant per-square-foot land and prestige premium.

### C. Physical Scaling (Total Area vs. Total Price)
- **Heteroscedasticity Pattern**: The scatter plot shows a classic fan-shaped expansion. Price dispersion is tight below 1,500 sq. ft., but expands drastically past 3,000 sq. ft.
- **Severe Spatial Anomalies Detected**:
  - Properties with $>3,000\text{ sq ft}$ listed below ₹100 Lakhs (mostly agricultural or peripheral growth corridors).
  - Miniature plots ($<800\text{ sq ft}$) listed above ₹300 Lakhs (commercial plots in central business districts like Indiranagar or Brigade Road).

### D. Correlation Matrix & Feature Multicollinearity
- **Strongest Correlation with Price**:
  - `clean_sqft` ($r = 0.58$): Primary physical determinant of price.
  - `bath` ($r = 0.46$) and `bhk` ($r = 0.40$).
- **High Multicollinearity Warning**:
  - `bhk` and `bath` exhibit a very high correlation ($r \approx 0.90$).
  - *ML Engineering Impact:* In linear models, severe multicollinearity between bedrooms and bathrooms can destabilize coefficient estimates. Tree-based ensembles are naturally robust against this collinearity.
- **Weakest Linear Predictor**:
  - `balcony` ($r \approx 0.12$): Weak linear correlation with total price.

---

## 3. Concrete Action Plan for Data Cleaning Pipeline (Next Step)

Based on these verified visualizations, our data cleaning pipeline must execute the following 5 operations:

1. **Drop Structural Columns**: Remove `society` (>41% nulls) and `availability` (>79% "Ready To Move").
2. **Remove Exact Duplicates**: Drop the 529 duplicate listing rows.
3. **Parse `total_sqft`**: Convert ranges to midpoints and discard unconvertible string units.
4. **Enforce Domain Outlier Boundaries**:
   - Filter $\frac{\text{total\_sqft}}{\text{bhk}} \ge 300$ (minimum space per bedroom).
   - Filter $\text{bath} \le \text{bhk} + 2$ (filter impossible bathroom counts like 40 bathrooms).
   - Remove price-per-sqft anomalies outside $1\sigma$ per neighborhood.
5. **Consolidate Rare Localities**: Group locations with $\le 10$ listings into `"other"` to prevent dimensionality explosion.

