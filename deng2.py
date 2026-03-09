# ============================================================
# 2-PANEL CORRELATION LOLLIPOP PLOT
# STARS ONLY + IMPROVED DESIGN
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# FILE PATHS
# ============================================================

input_file = r"C:\Users\nahid_vv0xche\Downloads\mymensingh_aedes_realistic_dataset.csv"
output_dir = r"C:\Users\nahid_vv0xche\Desktop\Dengue R2\New folder (2)"
os.makedirs(output_dir, exist_ok=True)

out_png = os.path.join(output_dir, "2panel_correlation_lollipop_aedes_ns1_STARS_ONLY.png")
out_pdf = os.path.join(output_dir, "2panel_correlation_lollipop_aedes_ns1_STARS_ONLY.pdf")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file)
df.columns = [c.strip() for c in df.columns]

# ============================================================
# DATE VARIABLES
# ============================================================

if "Date_time_year" in df.columns:
    df["Date_time_year"] = pd.to_datetime(df["Date_time_year"], dayfirst=True, errors="coerce")
    df["Year"] = df["Date_time_year"].dt.year
    df["Month_num"] = df["Date_time_year"].dt.month

# ============================================================
# COORDINATES
# ============================================================

def split_coordinates(coord):
    try:
        if pd.isna(coord):
            return np.nan, np.nan
        parts = str(coord).split(",")
        if len(parts) != 2:
            return np.nan, np.nan
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except:
        return np.nan, np.nan

if "Map_Cor" in df.columns:
    df[["Latitude", "Longitude"]] = df["Map_Cor"].apply(lambda x: pd.Series(split_coordinates(x)))

# ============================================================
# BINARY VARIABLES
# ============================================================

if "Previous_dengue_cases_last_month" in df.columns:
    df["Previous_dengue_cases_last_month_bin"] = (
        df["Previous_dengue_cases_last_month"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

if "NS1_Antigen" in df.columns:
    df["NS1_Antigen_bin"] = (
        df["NS1_Antigen"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"positive": 1, "negative": 0})
    )

# ============================================================
# PREDICTORS
# ============================================================

predictor_cols = [
    "Temperature_C",
    "Rainfall_mm",
    "Humidity_pct",
    "NDVI",
    "Soil_Moisture",
    "Distance_to_Water_km",
    "Population_density_per_km2",
    "House_density_per_hectare",
    "Socioeconomic_index",
    "Waste_management_score",
    "Sanitation_coverage_pct",
    "Vector_control_freq_per_month",
    "Spraying_effectiveness",
    "Previous_dengue_cases_last_month_bin",
    "Latitude",
    "Longitude",
    "Year",
    "Month_num"
]

predictor_cols = [c for c in predictor_cols if c in df.columns]

for c in predictor_cols + ["Aedes_Prevalence_percent", "NS1_Antigen_bin"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# SAFE SPEARMAN
# ============================================================

def safe_spearman(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    res = spearmanr(x, y)
    return float(res.correlation), float(res.pvalue)

# ============================================================
# BUILD CORRELATION TABLE
# ============================================================

def build_corr_table(data, predictors, outcome):
    rows = []
    for p in predictors:
        rho, pval = safe_spearman(data[p], data[outcome])
        rows.append({
            "Predictor": p,
            "rho": rho,
            "pval": pval
        })
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["rho"]).sort_values("rho").reset_index(drop=True)
    return out

corr_aedes = build_corr_table(df, predictor_cols, "Aedes_Prevalence_percent")
corr_ns1 = build_corr_table(df, predictor_cols, "NS1_Antigen_bin")

# ============================================================
# PRETTY LABELS
# ============================================================

pretty_labels = {
    "Temperature_C": "Temperature",
    "Rainfall_mm": "Rainfall",
    "Humidity_pct": "Humidity",
    "NDVI": "NDVI",
    "Soil_Moisture": "Soil moisture",
    "Distance_to_Water_km": "Distance to water",
    "Population_density_per_km2": "Population density",
    "House_density_per_hectare": "House density",
    "Socioeconomic_index": "Socioeconomic index",
    "Waste_management_score": "Waste management",
    "Sanitation_coverage_pct": "Sanitation coverage",
    "Vector_control_freq_per_month": "Vector control frequency",
    "Spraying_effectiveness": "Spraying effectiveness",
    "Previous_dengue_cases_last_month_bin": "Previous dengue case",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Year": "Year",
    "Month_num": "Month"
}

corr_aedes["label"] = corr_aedes["Predictor"].map(pretty_labels).fillna(corr_aedes["Predictor"])
corr_ns1["label"] = corr_ns1["Predictor"].map(pretty_labels).fillna(corr_ns1["Predictor"])

# ============================================================
# SIGNIFICANCE STARS
# ============================================================

def star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

corr_aedes["star"] = corr_aedes["pval"].apply(star)
corr_ns1["star"] = corr_ns1["pval"].apply(star)

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def draw_lollipop(ax, data, title, stem_color, point_color):
    y = np.arange(len(data))
    x = data["rho"].values

    # soft background
    ax.set_facecolor("#F8F9FA")

    # stems
    for yi, xi in zip(y, x):
        ax.hlines(
            yi, 0, xi,
            color=stem_color,
            linewidth=2.8,
            alpha=0.85,
            zorder=2
        )

    # points
    ax.scatter(
        x, y,
        s=95,
        color=point_color,
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    # zero line
    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=1
    )

    # grid
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    # titles and labels
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Spearman correlation (rho)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predictor", fontsize=16, fontweight="bold")

    # tick styles
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    # spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    # stars only
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    offset = 0.03 * x_range

    for yi, xi, st in zip(y, x, data["star"]):
        if st == "":
            continue

        if xi >= 0:
            xt = xi + offset
            ha = "left"
        else:
            xt = xi - offset
            ha = "right"

        ax.text(
            xt,
            yi,
            st,
            va="center",
            ha=ha,
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=4
        )

# ============================================================
# CREATE FIGURE
# ============================================================

vals = pd.concat([corr_aedes["rho"], corr_ns1["rho"]]).dropna()
xmin = min(-0.50, float(vals.min()) - 0.08)
xmax = max(0.50, float(vals.max()) + 0.08)

fig, axes = plt.subplots(1, 2, figsize=(24, 11), sharex=True)

axes[0].set_xlim(xmin, xmax)
axes[1].set_xlim(xmin, xmax)

draw_lollipop(
    axes[0],
    corr_aedes,
    title="Aedes prevalence",
    stem_color="#4C78A8",
    point_color="#2F5597"
)

draw_lollipop(
    axes[1],
    corr_ns1,
    title="NS1 antigen positivity",
    stem_color="#E07A5F",
    point_color="#C44E52"
)

plt.subplots_adjust(
    left=0.26,
    right=0.98,
    bottom=0.12,
    top=0.90,
    wspace=0.30
)

plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, dpi=400, bbox_inches="tight", facecolor="white")
plt.close()

print("Finished")
print(out_png)
print(out_pdf)