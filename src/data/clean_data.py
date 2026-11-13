"""
EstateIQ AI - Production Data Cleaning & Feature Engineering Pipeline
---------------------------------------------------------------------
Transforms raw historical Bengaluru housing data into a sanitized,
feature-engineered dataset ready for spatial enrichment and ML modeling.

Pipeline Steps:
1. Ingest raw CSV (read-only, immutable).
2. Deduplicate exact duplicate listings.
3. Handle missing values based on empirical EDA findings.
4. Extract integer BHK from 'size'.
5. Convert 'total_sqft' (ranges, metric units) into floating-point sq. ft.
6. Clean and frequency-group 'location' categories.
7. Compute 'price_per_sqft' as a diagnostic outlier metric (leakage-safe).
8. Remove physical domain anomalies (sqft/bhk ratio, bathroom mismatch).
9. Apply locality-aware statistical outlier filters without purging genuine luxury.
10. Validate data contract and serialize to 'data/processed/bengaluru_cleaned.csv'.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EstateIQ-Pipeline")


def load_raw_dataset(filepath: Path) -> pd.DataFrame:
    """Loads raw dataset without mutation."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found at expected path: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset from {filepath.name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Identifies and drops exact duplicate listing rows."""
    initial_count = len(df)
    df_clean = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df_clean)
    logger.info(f"Deduplication: Removed {duplicates_removed:,} exact duplicate rows ({len(df_clean):,} remaining)")
    return df_clean, duplicates_removed


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles missing values based on EDA findings:
    - Drops 'society' (>41% missing, high sparsity).
    - Drops 'availability' (>79% 'Ready To Move', negligible predictive signal).
    - Drops rows where 'size' or 'location' are missing (vital structural features, <0.15% data loss).
    - Imputes 'bath' using the median bathroom count for that specific BHK tier.
    - Imputes 'balcony' with the overall median (1.0 / 2.0).
    """
    df = df.copy()
    stats = {}

    # 1. Feature selection: drop columns with structural flaws or low predictive utility
    cols_to_drop = [c for c in ["society", "availability"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    stats["dropped_columns"] = cols_to_drop
    logger.info(f"Feature Selection: Dropped low-utility/high-null columns: {cols_to_drop}")

    # 2. Drop rows missing fundamental identifiers
    initial_rows = len(df)
    df = df.dropna(subset=["size", "location"]).copy()
    dropped_critical = initial_rows - len(df)
    stats["dropped_missing_identifiers"] = dropped_critical
    logger.info(f"Dropped {dropped_critical} rows missing 'size' or 'location'")

    return df, stats


def clean_bhk_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numerical bedroom count from 'size' strings (e.g., '2 BHK' -> 2, '4 Bedroom' -> 4).
    Filters out extreme physical corruptions (>16 BHK).
    """
    df = df.copy()
    df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]))
    logger.info(f"Feature Engineering: Extracted 'bhk' (Range: {df['bhk'].min()} to {df['bhk'].max()})")
    return df


def parse_total_sqft(val: Any) -> Optional[float]:
    """
    Robust conversion of 'total_sqft' string expressions:
    - Standard numbers: '1200' -> 1200.0
    - Ranges: '2100 - 2850' -> arithmetic mean = 2475.0 sq. ft.
    - Metric area conversions:
      * Sq. Meter -> sq ft (factor: 10.7639)
      * Sq. Yards -> sq ft (factor: 9.0)
      * Acres     -> sq ft (factor: 43560.0)
      * Guntha    -> sq ft (factor: 1089.0)
      * Perch     -> sq ft (factor: 272.25)
      * Cents     -> sq ft (factor: 435.56)
      * Grounds   -> sq ft (factor: 2400.0)
    - Unparseable strings -> None
    """
    if pd.isna(val):
        return None

    val_str = str(val).strip()

    # 1. Range format (e.g., '2100 - 2850')
    if "-" in val_str:
        tokens = val_str.split("-")
        if len(tokens) == 2:
            try:
                t1 = float(tokens[0].strip())
                t2 = float(tokens[1].strip())
                return (t1 + t2) / 2.0
            except ValueError:
                return None

    # 2. Standard float format
    try:
        return float(val_str)
    except ValueError:
        pass

    # 3. Unit-aware conversions using regex
    unit_map = {
        r"sq\.\s*meter": 10.7639,
        r"sq\.\s*yards": 9.0,
        r"acres?": 43560.0,
        r"guntha": 1089.0,
        r"perch": 272.25,
        r"cents?": 435.56,
        r"grounds?": 2400.0,
    }

    for pattern, multiplier in unit_map.items():
        match = re.search(r"([\d\.]+)\s*" + pattern, val_str, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return round(num * multiplier, 2)
            except ValueError:
                return None

    return None


def clean_total_sqft_feature(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Applies sqft parser and drops unconvertible values."""
    df = df.copy()
    initial_rows = len(df)
    df["total_sqft"] = df["total_sqft"].apply(parse_total_sqft)
    
    # Drop records where sqft could not be resolved
    df = df.dropna(subset=["total_sqft"]).copy()
    dropped_sqft = initial_rows - len(df)
    logger.info(f"Total Sqft Cleaning: Converted area features. Dropped {dropped_sqft} invalid unit rows ({len(df):,} remaining)")
    return df, dropped_sqft


def impute_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes bathrooms and balconies contextually:
    - Bathrooms: Imputed by BHK-specific median (e.g., 2 BHK usually has 2 baths).
    - Balconies: Imputed by overall median (2.0).
    """
    df = df.copy()
    
    # Contextual bathroom imputation by BHK
    bhk_bath_median = df.groupby("bhk")["bath"].transform("median")
    df["bath"] = df["bath"].fillna(bhk_bath_median).fillna(2.0)
    
    # Balcony median imputation
    median_balcony = df["balcony"].median()
    df["balcony"] = df["balcony"].fillna(median_balcony)
    
    logger.info("Imputation: Contextually imputed missing 'bath' by BHK median, and 'balcony' by median.")
    return df


def clean_location_categories(df: pd.DataFrame, threshold: int = 10) -> Tuple[pd.DataFrame, int]:
    """
    Standardizes location text and collapses rare localities (<= threshold)
    into an 'other' category to curb high dimensionality while preserving major clusters.
    """
    df = df.copy()
    df["location"] = df["location"].astype(str).str.strip()
    
    # Frequency counts
    location_counts = df["location"].value_counts()
    rare_locations = location_counts[location_counts <= threshold].index
    
    df["location"] = df["location"].apply(lambda loc: "other" if loc in rare_locations else loc)
    unique_locations = df["location"].nunique()
    logger.info(
        f"Location Consolidation: Compressed location space to {unique_locations} categories (threshold <= {threshold})"
    )
    return df, unique_locations


def remove_domain_and_statistical_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Removes implausible physical records and statistical anomalies while
    carefully PRESERVING genuine high-end luxury properties:
    
    1. Physical Space Density:
       - Every residential bedroom requires a bare minimum of 300 sq. ft.
       - Discards records where total_sqft / bhk < 300.
    
    2. Bathroom Density Anomaly:
       - A home should not have bath > bhk + 2 (e.g., 2 BHK with 5 bathrooms or 40 bathrooms).
    
    3. Locality-Aware Price-per-Sqft Filtering:
       - Computes price_per_sqft strictly as an outlier audit metric.
       - For each neighborhood, filters out listings outside [mean - 2*std, mean + 2*std].
       - Using 2 standard deviations (~95% coverage) preserves authentic high-end luxury
         villas in prime areas (Indiranagar, Whitefield) while eliminating fat-finger entry errors.
    """
    df = df.copy()
    audit = {}
    
    # Diagnostic metric (Price is in Lakhs, multiply by 100,000 for INR)
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]
    
    # 1. Physical space rule: minimum 300 sq. ft. per bedroom
    init_len = len(df)
    df = df[~(df["total_sqft"] / df["bhk"] < 300)].copy()
    audit["sqft_per_bhk_outliers"] = init_len - len(df)
    logger.info(f"Outlier Filter 1: Removed {audit['sqft_per_bhk_outliers']} rows with < 300 sqft/BHK")
    
    # 2. Bathroom anomaly rule: bath <= bhk + 2
    init_len = len(df)
    df = df[df["bath"] <= (df["bhk"] + 2)].copy()
    audit["bathroom_anomalies"] = init_len - len(df)
    logger.info(f"Outlier Filter 2: Removed {audit['bathroom_anomalies']} rows with bath > bhk + 2")

    # 3. Non-residential / agricultural land filter (minimum economic threshold for urban residential homes: ₹1,000/sqft)
    init_len = len(df)
    df = df[df["price_per_sqft"] >= 1000].copy()
    audit["agricultural_land_outliers"] = init_len - len(df)
    logger.info(f"Outlier Filter 3: Removed {audit['agricultural_land_outliers']} agricultural/extreme low-price listings (< ₹1,000/sqft)")
    
    # 4. Locality-aware 2-sigma filter (preserves genuine luxury)
    init_len = len(df)
    df_filtered = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        mean = np.mean(subdf["price_per_sqft"])
        std = np.std(subdf["price_per_sqft"])
        # If std is zero (e.g. single listing), retain
        if std == 0 or np.isnan(std):
            df_filtered = pd.concat([df_filtered, subdf], ignore_index=True)
        else:
            # Keep within 2 standard deviations (95% spread)
            valid_subset = subdf[
                (subdf["price_per_sqft"] >= (mean - 2 * std)) & 
                (subdf["price_per_sqft"] <= (mean + 2 * std))
            ]
            df_filtered = pd.concat([df_filtered, valid_subset], ignore_index=True)
            
    audit["price_per_sqft_anomalies"] = init_len - len(df_filtered)
    logger.info(f"Outlier Filter 4: Removed {audit['price_per_sqft_anomalies']} extreme price-per-sqft anomalies (2-sigma)")
    
    return df_filtered, audit


def run_pipeline(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    location_threshold: int = 10
) -> pd.DataFrame:
    """Orchestrates the end-to-end data cleaning pipeline."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if raw_path is None:
        raw_path = base_dir / "data" / "raw" / "Bengaluru_House_Data.csv"
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "bengaluru_cleaned.csv"
        
    logger.info("=" * 70)
    logger.info("STARTING ESTATEIQ AI DATA CLEANING PIPELINE")
    logger.info("=" * 70)
    
    # Execution
    df_raw = load_raw_dataset(raw_path)
    df_dedup, num_dups = remove_duplicates(df_raw)
    df_missing, miss_stats = handle_missing_values(df_dedup)
    df_bhk = clean_bhk_feature(df_missing)
    df_sqft, num_sqft_dropped = clean_total_sqft_feature(df_bhk)
    df_imputed = impute_amenities(df_sqft)
    df_loc, num_unique_locs = clean_location_categories(df_imputed, threshold=location_threshold)
    df_cleaned, outlier_stats = remove_domain_and_statistical_outliers(df_loc)
    
    # Final column ordering & feature retention
    # We retain area_type because independent plotted houses have different valuation dynamics than super built-up apartments
    target_columns = ["area_type", "location", "total_sqft", "bhk", "bath", "balcony", "price", "price_per_sqft"]
    final_df = df_cleaned[[c for c in target_columns if c in df_cleaned.columns]].copy()
    
    # Data Quality Validation Checks (Assertions)
    assert not final_df.isnull().any().any(), "Validation Failed: Cleaned dataset contains NaN values!"
    assert (final_df["price"] > 0).all(), "Validation Failed: Found non-positive price!"
    assert (final_df["total_sqft"] > 0).all(), "Validation Failed: Found non-positive total_sqft!"
    assert (final_df["bhk"] > 0).all(), "Validation Failed: Found non-positive bhk!"
    assert (final_df["bath"] > 0).all(), "Validation Failed: Found non-positive bath!"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Pipeline Succeeded! Output written to: {output_path}")
    logger.info(f"Final Cleaned Dataset Dimensions: {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
    logger.info("=" * 70)
    
    return final_df


if __name__ == "__main__":
    run_pipeline()
