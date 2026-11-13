"""
EstateIQ AI - Visual Exploratory Data Analysis (EDA)
----------------------------------------------------
Section 1: Target Price Distribution & Log-Transformation Diagnostics (Saved: 01_price_distribution.png)
Section 2: BHK Box Plots & Area vs Price Scatter Analysis (Saved: 02_bhk_and_area_analysis.png)
Section 3: Locality Comparison & Feature Correlation Heatmap (Saved: 03_locality_and_correlation.png)
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
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["font.size"] = 11
sns.set_theme(style="whitegrid", palette="muted")

# 1. Resolve paths
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "Bengaluru_House_Data.csv"
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🏢 EstateIQ AI — Visual Exploratory Data Analysis: Section 3")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}")

df_raw = pd.read_csv(data_path)

# Helper: parse sqft safely
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

df_plot = df_raw.copy()
df_plot["clean_sqft"] = df_plot["total_sqft"].apply(parse_sqft)
df_plot["bhk"] = df_plot["size"].dropna().apply(lambda x: int(str(x).split()[0]))
df_plot["price_per_sqft"] = (df_plot["price"] * 100000) / df_plot["clean_sqft"]
df_plot["clean_location"] = df_plot["location"].astype(str).str.strip()

# ----------------------------------------------------------------------
# FIGURE 3: Locality Price Comparison & Correlation Matrix
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Top 10 High-Volume Localities Comparison (Median Price/SqFt)
top_locations = df_plot["clean_location"].value_counts().head(10).index
loc_df = df_plot[df_plot["clean_location"].isin(top_locations)]
loc_stats = loc_df.groupby("clean_location")["price_per_sqft"].median().sort_values(ascending=True)

colors = sns.color_palette("viridis", len(loc_stats))
bars = axes[0].barh(loc_stats.index, loc_stats.values, color=colors, edgecolor="black", height=0.6)
axes[0].set_title("Median Price per Sq. Ft. in Top 10 Most Listed Localities", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Median Price per Sq. Ft. (₹/sq ft)")
axes[0].set_ylabel("Bengaluru Neighborhood")

# Add data labels onto bars
for bar in bars:
    w = bar.get_width()
    axes[0].text(w + 100, bar.get_y() + bar.get_height() / 2, f"₹{w:,.0f}", va="center", fontsize=10, fontweight="bold")

axes[0].set_xlim(0, max(loc_stats.values) * 1.2)

# Subplot 2: Correlation Heatmap for Numerical Features
num_cols = ["price", "clean_sqft", "bhk", "bath", "balcony"]
corr_matrix = df_plot[num_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    linewidths=1,
    square=True,
    ax=axes[1],
    annot_kws={"size": 11, "weight": "bold"}
)
axes[1].set_title("Correlation Heatmap of Key Numerical Features", fontsize=13, fontweight="bold")

plt.suptitle("EstateIQ AI — Spatial Locality Variance & Feature Correlations", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

# Save figure
save_path3 = figures_dir / "03_locality_and_correlation.png"
plt.savefig(save_path3, dpi=300, bbox_inches="tight")
print(f"✅ Figure 3 successfully saved to: {save_path3}")

print("\n" + "-" * 70)
print("PEARSON CORRELATION MATRIX:")
print("-" * 70)
print(corr_matrix.round(3))
print("-" * 70)

# Display interactive plot window
plt.show()
