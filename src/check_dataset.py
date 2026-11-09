"""
EstateIQ AI - Dataset Integrity & Overview Inspector
----------------------------------------------------
This script inspects the raw Bengaluru House Price dataset:
- Validates file existence using pathlib
- Reports dataset dimensions, columns, and data types
- Analyzes missing values and duplicate rows
- Computes foundational summary statistics
- Resilient design: Uses pandas if available, with built-in standard library
  fallback to ensure zero runtime blockage under Windows Application Control policies.
"""

import sys
import csv
from pathlib import Path


def inspect_with_pandas(df, dataset_path, num_rows, num_cols, file_size_mb):
    """Detailed inspection using pandas when available."""
    import pandas as pd

    print("\n" + "-" * 70)
    print("1. DATASET DIMENSIONS & SIZE")
    print("-" * 70)
    print(f"Total Records (Rows)   : {num_rows:,}")
    print(f"Total Features (Cols)  : {num_cols}")
    print(f"Disk File Size         : {file_size_mb:.2f} MB")

    print("\n" + "-" * 70)
    print("2. COLUMNS & DATA TYPES")
    print("-" * 70)
    col_summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(col_summary.to_string())

    print("\n" + "-" * 70)
    print("3. FIRST 5 RECORDS (SAMPLE HEAD)")
    print("-" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

    print("\n" + "-" * 70)
    print("4. DUPLICATE ROWS ANALYSIS")
    print("-" * 70)
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / num_rows) * 100
    print(f"Duplicate Rows Count   : {duplicate_count:,} ({duplicate_pct:.2f}% of total data)")
    if duplicate_count > 0:
        print("Note: Duplicate rows represent repeated listings and can be safely dropped during data cleaning.")

    print("\n" + "-" * 70)
    print("5. NUMERICAL DESCRIPTIVE STATISTICS")
    print("-" * 70)
    print(df.describe().round(2))

    print("\n" + "-" * 70)
    print("6. CATEGORICAL FEATURES CARDINALITY")
    print("-" * 70)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"Feature: {col:<15} | Unique Values: {unique_vals:>6,}")


def inspect_with_stdlib(dataset_path, file_size_mb):
    """Resilient pure-Python inspection using built-in csv module."""
    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    num_rows = len(rows)
    num_cols = len(header)

    print("\n" + "-" * 70)
    print("1. DATASET DIMENSIONS & SIZE")
    print("-" * 70)
    print(f"Total Records (Rows)   : {num_rows:,}")
    print(f"Total Features (Cols)  : {num_cols}")
    print(f"Disk File Size         : {file_size_mb:.2f} MB")

    # Column summary & missingness
    print("\n" + "-" * 70)
    print("2. COLUMNS & MISSING VALUES")
    print("-" * 70)
    print(f"{'Column Name':<20} | {'Non-Null':<10} | {'Null Count':<10} | {'Null %':<8}")
    print("-" * 60)

    col_values = {col: [] for col in header}
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(header):
                col_values[header[idx]].append(val.strip())

    for col in header:
        vals = col_values[col]
        null_count = sum(1 for v in vals if v == "" or v.lower() == "nan" or v.lower() == "none")
        non_null = num_rows - null_count
        null_pct = (null_count / num_rows) * 100
        print(f"{col:<20} | {non_null:<10,} | {null_count:<10,} | {null_pct:>6.2f}%")

    # Sample head (first 5 rows)
    print("\n" + "-" * 70)
    print("3. FIRST 5 RECORDS (SAMPLE HEAD)")
    print("-" * 70)
    for i, row in enumerate(rows[:5], 1):
        row_dict = dict(zip(header, row))
        print(f"Row {i}: {row_dict}")

    # Duplicates check
    seen = set()
    duplicates = 0
    for row in rows:
        t_row = tuple(row)
        if t_row in seen:
            duplicates += 1
        else:
            seen.add(t_row)
    dup_pct = (duplicates / num_rows) * 100
    print("\n" + "-" * 70)
    print("4. DUPLICATE ROWS ANALYSIS")
    print("-" * 70)
    print(f"Duplicate Rows Count   : {duplicates:,} ({dup_pct:.2f}% of total data)")

    # Cardinality for key columns
    print("\n" + "-" * 70)
    print("5. CATEGORICAL FEATURES CARDINALITY")
    print("-" * 70)
    for col in ["area_type", "availability", "location", "size", "society"]:
        if col in col_values:
            unique_count = len(set(col_values[col]))
            print(f"Feature: {col:<15} | Unique Values: {unique_count:>6,}")

    # Numeric summary for price
    price_idx = header.index("price") if "price" in header else -1
    if price_idx != -1:
        prices = []
        for r in rows:
            try:
                prices.append(float(r[price_idx]))
            except ValueError:
                pass
        if prices:
            print("\n" + "-" * 70)
            print("6. TARGET VARIABLE ('price' in ₹ Lakhs) SUMMARY")
            print("-" * 70)
            print(f"Count : {len(prices):,}")
            print(f"Min   : ₹{min(prices):.2f} Lakhs")
            print(f"Max   : ₹{max(prices):.2f} Lakhs")
            print(f"Mean  : ₹{(sum(prices) / len(prices)):.2f} Lakhs")


def check_dataset():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    dataset_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"

    print("=" * 70)
    print("🏢 EstateIQ AI - Raw Dataset Health & Integrity Check")
    print("=" * 70)
    print(f"Project Root Directory : {project_root}")
    print(f"Target Dataset Path    : {dataset_path}")

    if not dataset_path.exists():
        print("\n❌ ERROR: Dataset file not found!")
        print(f"Expected file at: {dataset_path}")
        print("\nPlease ensure 'Bengaluru_House_Data.csv' is inside 'data/raw/'.")
        sys.exit(1)

    print("\n✅ Dataset file located successfully.")

    file_size_mb = dataset_path.stat().st_size / (1024 * 1024)

    # Attempt pandas first, fallback to standard library if blocked by OS policy
    pandas_loaded = False
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        num_rows, num_cols = df.shape
        inspect_with_pandas(df, dataset_path, num_rows, num_cols, file_size_mb)
        pandas_loaded = True
    except Exception:
        pass

    if not pandas_loaded:
        inspect_with_stdlib(dataset_path, file_size_mb)

    print("\n" + "=" * 70)
    print("🎉 Dataset Health Check Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    check_dataset()
