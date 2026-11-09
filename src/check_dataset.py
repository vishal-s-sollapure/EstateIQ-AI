"""
EstateIQ AI - Dataset Integrity & Overview Inspector
----------------------------------------------------
This script inspects the raw Bengaluru House Price dataset:
- Validates file existence using pathlib
- Reports dataset dimensions, columns, and data types
- Analyzes missing values and duplicate rows
- Computes foundational summary statistics
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("❌ ERROR: 'pandas' is not installed in your active Python environment.")
    print("Please activate your virtual environment and run:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


def check_dataset():
    # -------------------------------------------------------------
    # 1. Path Resolution using pathlib
    # -------------------------------------------------------------
    # __file__ is the location of this script (src/check_dataset.py).
    # .resolve() gets the absolute canonical path.
    # .parent is 'src/', and .parent.parent is the root 'EstateIQ-AI/'.
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    dataset_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"

    print("=" * 70)
    print("🏢 EstateIQ AI - Raw Dataset Health & Integrity Check")
    print("=" * 70)
    print(f"Project Root Directory : {project_root}")
    print(f"Target Dataset Path    : {dataset_path}")

    # -------------------------------------------------------------
    # 2. File Existence & Error Handling
    # -------------------------------------------------------------
    if not dataset_path.exists():
        print("\n❌ ERROR: Dataset file not found!")
        print(f"Expected file at: {dataset_path}")
        print("\nPlease ensure you have:")
        print("  1. Downloaded the dataset from Kaggle.")
        print("  2. Extracted 'Bengaluru_House_Data.csv'.")
        print("  3. Placed it inside the 'data/raw/' directory.")
        sys.exit(1)

    print("\n✅ Dataset file located successfully.")

    # -------------------------------------------------------------
    # 3. Ingesting Data with Pandas
    # -------------------------------------------------------------
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to read CSV file: {e}")
        sys.exit(1)

    # -------------------------------------------------------------
    # 4. Dataset Shape & Dimensions
    # -------------------------------------------------------------
    num_rows, num_cols = df.shape
    file_size_mb = dataset_path.stat().st_size / (1024 * 1024)

    print("\n" + "-" * 70)
    print("1. DATASET DIMENSIONS & SIZE")
    print("-" * 70)
    print(f"Total Records (Rows)   : {num_rows:,}")
    print(f"Total Features (Cols)  : {num_cols}")
    print(f"Disk File Size         : {file_size_mb:.2f} MB")

    # -------------------------------------------------------------
    # 5. Column Names & Inferred Data Types
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 6. Sample Records (First 5 Rows)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. FIRST 5 RECORDS (SAMPLE HEAD)")
    print("-" * 70)
    # Configure pandas display width for clean terminal output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

    # -------------------------------------------------------------
    # 7. Duplicate Records Count
    # -------------------------------------------------------------
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / num_rows) * 100
    print("\n" + "-" * 70)
    print("4. DUPLICATE ROWS ANALYSIS")
    print("-" * 70)
    print(f"Duplicate Rows Count   : {duplicate_count:,} ({duplicate_pct:.2f}% of total data)")
    if duplicate_count > 0:
        print("Note: Duplicate rows represent repeated listings and can be safely dropped during data cleaning.")

    # -------------------------------------------------------------
    # 8. Numerical Descriptive Statistics
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. NUMERICAL DESCRIPTIVE STATISTICS")
    print("-" * 70)
    print(df.describe().round(2))

    # -------------------------------------------------------------
    # 9. Categorical Features Cardinality & Distribution
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. CATEGORICAL FEATURES CARDINALITY")
    print("-" * 70)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"Feature: {col:<15} | Unique Values: {unique_vals:>6,}")

    print("\n" + "=" * 70)
    print("🎉 Dataset Health Check Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    check_dataset()
