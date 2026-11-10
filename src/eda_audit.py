"""
EstateIQ AI - Phase 2: Exploratory Data Analysis (EDA)
Section 1: Dataset Ingestion & Structural Inspection
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
print("🏢 EstateIQ AI — Phase 2: EDA Dataset Ingestion & Structural Audit")
print("=" * 70)
print(f"Loading data from: {data_path}")

df = pd.read_csv(data_path)

print("\n" + "=" * 70)
print("1. SHAPE & COLUMNS")
print("=" * 70)
print(f"Total Rows (Listings)   : {df.shape[0]:,}")
print(f"Total Columns (Features): {df.shape[1]}")
print(f"\nColumn Names:\n{list(df.columns)}")

print("\n" + "=" * 70)
print("2. FIRST 5 ROWS (HEAD)")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("3. LAST 5 ROWS (TAIL)")
print("=" * 70)
print(df.tail())

print("\n" + "=" * 70)
print("4. DATASET INFO & DATA TYPES")
print("=" * 70)
df.info()
print("=" * 70)

