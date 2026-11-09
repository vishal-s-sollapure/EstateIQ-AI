"""
EstateIQ AI - Data Cleaning & Preprocessing Pipeline
-----------------------------------------------------
Transforms raw Bengaluru real estate data into a clean, feature-ready dataset:
1. Handles missing values in critical columns.
2. Standardizes 'size' into an integer 'bhk' column.
3. Converts non-standard 'total_sqft' (ranges, mixed units) to numeric square footage.
4. Reduces 'location' cardinality using frequency thresholding (the 'other' bucket).
5. Computes 'price_per_sqft' for domain-specific outlier filtering.
6. Removes unrealistic entries based on real estate domain rules.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("EstateIQ-Cleaner")


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Loads the raw Bengaluru house price CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    return df


def clean_bhk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the integer number of bedrooms from 'size' strings
    (e.g., '2 BHK' -> 2, '4 Bedroom' -> 4).
    """
    df = df.copy()
    # Drop rows where 'size' is missing as it is fundamental
    df = df.dropna(subset=["size"])
    df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]))
    logger.info(f"Extracted 'bhk' feature. Range: {df['bhk'].min()} to {df['bhk'].max()}")
    return df


def convert_sqft_to_num(val: str) -> Optional[float]:
    """
    Helper function to convert string sqft values:
    - Normal floats/ints: '1200' -> 1200.0
    - Ranges: '2100 - 2850' -> (2100 + 2850) / 2
    - Other units (Acres, Sq. Meter) -> None / NaN
    """
    tokens = str(val).split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
        except ValueError:
            return None
    try:
        return float(val)
    except ValueError:
        return None


def clean_total_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes 'total_sqft' into clean floating-point numbers."""
    df = df.copy()
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
    df = df.dropna(subset=["total_sqft"])
    logger.info(f"Cleaned 'total_sqft'. Valid records remaining: {len(df)}")
    return df


def clean_location(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Strips whitespace from location names and groups infrequent locations
    (<= threshold occurrences) into an 'other' category to prevent curse of dimensionality.
    """
    df = df.copy()
    df["location"] = df["location"].astype(str).apply(lambda x: x.strip())
    
    # Handle rare locations
    location_stats = df["location"].value_counts()
    locations_less_than_threshold = location_stats[location_stats <= threshold]
    df["location"] = df["location"].apply(
        lambda x: "other" if x in locations_less_than_threshold else x
    )
    logger.info(
        f"Reduced unique locations to {df['location'].nunique()} categories (threshold: {threshold})"
    )
    return df


def remove_domain_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies real-world domain constraints:
    1. A standard bedroom requires at least 300 sq. ft. of space.
    2. Removes statistical price-per-sqft anomalies per location (outside 1 std dev).
    3. Removes bathroom anomalies (bath > bhk + 2).
    """
    df = df.copy()

    # Calculate price per sqft (Price is in Lakhs, multiply by 100,000 for INR)
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

    # Constraint 1: Minimum 300 sqft per BHK
    df = df[~(df["total_sqft"] / df["bhk"] < 300)]
    logger.info(f"Filtered records with sqft/bhk < 300. Remaining: {len(df)}")

    # Constraint 2: Statistical outlier removal per location
    df_out = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        mean = np.mean(subdf["price_per_sqft"])
        std = np.std(subdf["price_per_sqft"])
        # Keep data within 1 standard deviation of local location mean
        reduced_df = subdf[(subdf["price_per_sqft"] > (mean - std)) & (subdf["price_per_sqft"] <= (mean + std))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    # Constraint 3: Bathroom anomaly (e.g. 2 BHK with 5 bathrooms)
    df_out = df_out.dropna(subset=["bath"])
    df_out = df_out[df_out["bath"] < (df_out["bhk"] + 2)]
    logger.info(f"Filtered statistical & bathroom anomalies. Final records: {len(df_out)}")

    return df_out


def run_cleaning_pipeline(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Orchestrates the complete data cleaning pipeline and saves output."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if raw_path is None:
        raw_path = base_dir / "data" / "raw" / "Bengaluru_House_Data.csv"
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "bengaluru_cleaned.csv"

    df = load_raw_data(raw_path)
    df = clean_bhk(df)
    df = clean_total_sqft(df)
    df = clean_location(df)
    df = remove_domain_outliers(df)

    # Keep relevant core features for downstream modeling and geospatial enrichment
    columns_to_keep = ["location", "bhk", "total_sqft", "bath", "balcony", "price", "price_per_sqft"]
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    cleaned_df = df[existing_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to: {output_path} (Shape: {cleaned_df.shape})")

    return cleaned_df


if __name__ == "__main__":
    run_cleaning_pipeline()
