"""
EstateIQ AI - Visual Exploratory Data Analysis (EDA)
----------------------------------------------------
Section 1: Target Price Distribution & Log-Transformation Diagnostics (Saved: 01_price_distribution.png)
Section 2: BHK Box Plots & Area vs Price Scatter Analysis (Saved: 02_bhk_and_area_analysis.png)
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

# ----------------------------------------------------------------------
# HELPER: Temporary parser for plotting without modifying df_raw
# ----------------------------------------------------------------------
def parse_sqft(val):
    tokens = str(val).split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

# Construct temporary diagnostic dataframe
df_plot = df_raw.copy()
df_plot["clean_sqft"] = df_plot["total_sqft"].apply(parse_sqft)
df_plot["bhk"] = df_plot["size"].dropna().apply(lambda x: int(str(x).split()[0]))
df_plot["price_per_sqft"] = (df_plot["price"] * 100000) / df_plot["clean_sqft"]

# ----------------------------------------------------------------------
# FIGURE 2: BHK vs Price-Per-Sqft & Area vs Price Scaling
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Filter to common residential BHKs (1 to 6) and clip extreme visualization outliers for clean display
common_bhk = df_plot[(df_plot["bhk"] >= 1) & (df_plot["bhk"] <= 6) & (df_plot["price_per_sqft"] <= 25000)]

# Subplot 1: Box Plot of Price per Sq Ft across BHK
sns.boxplot(
    data=common_bhk,
    x="bhk",
    y="price_per_sqft",
    palette="Blues",
    ax=axes[0],
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"}
)
axes[0].set_title("Price per Sq. Ft. across BHK Categories (1 to 6 BHK)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Bedrooms (BHK)")
axes[0].set_ylabel("Price per Sq. Ft. (₹/sq ft)")
axes[0].axhline(common_bhk["price_per_sqft"].median(), color="green", linestyle="--", label=f"Overall Median: ₹{common_bhk['price_per_sqft'].median():.0f}/sqft")
axes[0].legend(loc="upper right")

# Subplot 2: Scatter Plot of Total Sqft vs Price (colored by BHK)
scatter_sample = common_bhk[(common_bhk["clean_sqft"] <= 6000) & (common_bhk["price"] <= 800)]
scatter = axes[1].scatter(
    scatter_sample["clean_sqft"],
    scatter_sample["price"],
    c=scatter_sample["bhk"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="none",
    s=30
)
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label("BHK Configuration", rotation=270, labelpad=15)
axes[1].set_title("Property Area (sq. ft.) vs. Price (₹ Lakhs)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Total Area (sq. ft.)")
axes[1].set_ylabel("Price in Lakhs (₹)")

plt.suptitle("EstateIQ AI — Physical Feature Scaling & Unit Economics", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

# Save figure
save_path2 = figures_dir / "02_bhk_and_area_analysis.png"
plt.savefig(save_path2, dpi=300, bbox_inches="tight")
print(f"✅ Figure 2 successfully saved to: {save_path2}")

# Display median price per sqft per BHK
print("\n" + "-" * 70)
print("MEDIAN PRICE PER SQFT BY BHK CATEGORY:")
print("-" * 70)
bhk_stats = common_bhk.groupby("bhk")["price_per_sqft"].agg(["count", "median", "mean"]).round(1)
print(bhk_stats)
print("-" * 70)

# Display interactive plot window
plt.show()
