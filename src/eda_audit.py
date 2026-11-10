"""
EstateIQ AI - Phase 2: Exploratory Data Analysis (EDA)
Section 1: Ingestion & Structural Inspection
Section 2: Missing Values, Duplicates & Statistical Summaries
Section 3: Suspicious Values & Deep-Dive Anomaly Audit
"""

from pathlib import Path
import re
import pandas as pd

# Clean display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"

df = pd.read_csv(data_path)

# ======================================================================
# SECTION 3: SUSPICIOUS VALUES & DEEP-DIVE ANOMALY AUDIT
# ======================================================================
print("=" * 70)
print("SECTION 3: SUSPICIOUS VALUES & DEEP-DIVE ANOMALIES")
print("=" * 70)

# 1. Investigate non-numeric 'total_sqft' values
def is_float(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

non_numeric_sqft = df[~df["total_sqft"].apply(is_float)]
print(f"\n1. Non-Numeric 'total_sqft' Values Identified: {len(non_numeric_sqft)} records")
print("Sample Non-Numeric Patterns in total_sqft:")
print(non_numeric_sqft[["total_sqft", "size", "price"]].head(10))

# 2. Extract integer BHK to inspect bedroom anomalies
df_size_clean = df.dropna(subset=["size"]).copy()
df_size_clean["bhk"] = df_size_clean["size"].apply(lambda x: int(str(x).split()[0]))

print("\n" + "-" * 70)
print(f"2. Extreme 'bhk' (Bedroom) Counts:")
print(f"BHK Distribution Summary:\n{df_size_clean['bhk'].value_counts().sort_index().tail(10)}")
extreme_bhk = df_size_clean[df_size_clean["bhk"] > 10]
print(f"\nListings with > 10 Bedrooms ({len(extreme_bhk)} records):")
print(extreme_bhk[["location", "size", "bhk", "total_sqft", "bath", "price"]])

# 3. Investigate extreme bathroom counts
print("\n" + "-" * 70)
print("3. Extreme Bathroom Counts (bath > 10):")
extreme_bath = df[df["bath"] > 10]
print(f"Listings with > 10 Bathrooms ({len(extreme_bath)} records):")
print(extreme_bath[["location", "size", "total_sqft", "bath", "price"]])

# 4. Investigate Bathroom-to-BHK discrepancy
bath_bhk_mismatch = df_size_clean[df_size_clean["bath"] > (df_size_clean["bhk"] + 2)]
print("\n" + "-" * 70)
print(f"4. Severe Bathroom Discrepancy (bath > bhk + 2): {len(bath_bhk_mismatch)} records")
print(bath_bhk_mismatch[["location", "size", "bhk", "bath", "price"]].head(5))

print("\n" + "=" * 70)
print("Section 3 Anomaly Audit Complete!")
print("=" * 70)
