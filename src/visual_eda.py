"""
EstateIQ AI - Visual Exploratory Data Analysis (EDA)
----------------------------------------------------
Generates target price distribution and log-transformation diagnostics.
Saves high-resolution figures to 'reports/figures/'.
"""

import os
import tempfile
from pathlib import Path

# Set Matplotlib cache directory to temporary directory
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting settings
plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["font.size"] = 11
sns.set_theme(style="whitegrid", palette="muted")

# 1. Resolve paths
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🏢 EstateIQ AI — Visual Exploratory Data Analysis")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}")

df_raw = pd.read_csv(data_path)
print(f"Dataset successfully loaded: {df_raw.shape[0]:,} listings across {df_raw.shape[1]} features\n")

# 2. Extract Target Price
prices = df_raw["price"]
log_prices = np.log(prices)

raw_skew = prices.skew()
log_skew = log_prices.skew()

# 3. Create Side-by-Side Diagnostic Subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Raw Price Distribution
sns.histplot(prices, kde=True, color="#1f77b4", ax=axes[0], bins=50)
axes[0].axvline(prices.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: ₹{prices.mean():.1f} L")
axes[0].axvline(prices.median(), color="green", linestyle="-", linewidth=2, label=f"Median: ₹{prices.median():.1f} L")
axes[0].set_title(f"Raw Price Distribution (Skewness: {raw_skew:.2f})", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Price in Lakhs (₹)")
axes[0].set_ylabel("Count (Frequency)")
axes[0].set_xlim(0, 1000)  # Zoom in on main density for visual clarity
axes[0].legend(frameon=True)

# Subplot 2: Log-Transformed Price Distribution
sns.histplot(log_prices, kde=True, color="#2ca02c", ax=axes[1], bins=40)
axes[1].axvline(log_prices.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {log_prices.mean():.2f}")
axes[1].axvline(log_prices.median(), color="green", linestyle="-", linewidth=2, label=f"Median: {log_prices.median():.2f}")
axes[1].set_title(f"Log-Transformed Price Distribution (Skewness: {log_skew:.2f})", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Log(Price in Lakhs)")
axes[1].set_ylabel("Count (Frequency)")
axes[1].legend(frameon=True)

plt.suptitle("EstateIQ AI — Target Variable (Price) Distribution Diagnostics", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

# 4. Save High-Resolution Plot
save_path = figures_dir / "01_price_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Figure successfully saved to: {save_path}")

print("-" * 70)
print(f"Raw Price Skewness        : {raw_skew:.2f} (Extremely Right-Skewed)")
print(f"Log-Transformed Skewness  : {log_skew:.2f} (Near-Symmetric Normal Distribution)")
print("-" * 70)

# Display interactive plot window
plt.show()
