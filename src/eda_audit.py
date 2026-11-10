"""
EstateIQ AI - Phase 2: Exploratory Data Analysis (EDA)
Section 1: Ingestion & Structural Inspection
Section 2: Missing Values, Duplicates & Statistical Summaries
"""

from pathlib import Path
import pandas as pd

# Clean display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Resolve path relative to project root
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"

print("=" * 70)
print("🏢 EstateIQ AI — Phase 2: EDA Dataset Audit")
print("=" * 70)

df = pd.read_csv(data_path)

# ----------------------------------------------------------------------
# SECTION 1: SHAPE & STRUCTURAL OVERVIEW
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 1: DATASET DIMENSIONS & COLUMNS")
print("=" * 70)
print(f"Total Rows (Listings)   : {df.shape[0]:,}")
print(f"Total Columns (Features): {df.shape[1]}")
print(f"\nColumn Names:\n{list(df.columns)}")

# ----------------------------------------------------------------------
# SECTION 2: MISSING VALUE ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2.1: MISSING VALUE BREAKDOWN")
print("=" * 70)
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df)) * 100
missing_df = pd.DataFrame({
    "Data Type": df.dtypes,
    "Non-Null Count": df.notnull().sum(),
    "Null Count": null_counts,
    "Null %": null_pct.round(2)
})
# Sort descending by null count to spotlight problem columns
print(missing_df.sort_values(by="Null Count", ascending=False).to_string())

# ----------------------------------------------------------------------
# SECTION 2.2: DUPLICATE ROW AUDIT
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2.2: DUPLICATE RECORDS AUDIT")
print("=" * 70)
num_duplicates = df.duplicated().sum()
dup_pct = (num_duplicates / len(df)) * 100
print(f"Duplicate Rows Identified : {num_duplicates:,} ({dup_pct:.2f}% of total data)")
if num_duplicates > 0:
    print("Interpretation: 529 identical records exist in the raw dataset. In real estate,")
    print("scraped listings are frequently duplicated by multiple broker agencies.")

# ----------------------------------------------------------------------
# SECTION 2.3: NUMERICAL STATISTICAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2.3: NUMERICAL FEATURES SUMMARY (df.describe())")
print("=" * 70)
num_summary = df.describe().round(2)
print(num_summary)

# ----------------------------------------------------------------------
# SECTION 2.4: CATEGORICAL FEATURES SUMMARY
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2.4: CATEGORICAL FEATURES SUMMARY")
print("=" * 70)
cat_cols = df.select_dtypes(include=["object", "str"]).columns
cat_summary = df[cat_cols].describe()
print(cat_summary)

print("\n" + "=" * 70)
print("Audit Section 1 & 2 Execution Complete!")
print("=" * 70)
