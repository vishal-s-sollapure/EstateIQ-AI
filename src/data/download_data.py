"""
EstateIQ AI - Data Acquisition Script
-------------------------------------
Downloads the standard Bengaluru House Price dataset from the public repository
and stores it in 'data/raw/Bengaluru_House_Data.csv'.
"""

import os
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("EstateIQ-Download")

# Canonical raw dataset URL (Bengaluru House Prices)
DATASET_URL = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/Bengaluru_House_Data.csv"

def download_dataset(dest_path: Path = None):
    if dest_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        dest_path = base_dir / "data" / "raw" / "Bengaluru_House_Data.csv"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"Dataset already exists at: {dest_path}")
        return dest_path

    logger.info(f"Downloading Bengaluru real estate dataset from: {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, dest_path)
        logger.info(f"Successfully downloaded dataset to: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise e

if __name__ == "__main__":
    download_dataset()
