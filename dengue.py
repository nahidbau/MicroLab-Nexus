# ============================================================
# FULL GEOSPATIAL + SPATIO-TEMPORAL DRIVER ANALYSIS
# AEDES PREVALENCE + NS1 POSITIVITY
# ============================================================

import os
import math
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# Optional network package
HAS_NETWORKX = True
try:
    import networkx as nx
except Exception:
    HAS_NETWORKX = False

# ============================================================
# 1. PATHS
# ============================================================
input_file = r"C:\Users\nahid_vv0xche\Downloads\mymensingh_aedes_realistic_dataset.csv"
output_dir = r"C:\Users\nahid_vv0xche\Desktop\Dengue R2\New folder"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. READ DATA
# ============================================================
df = pd.read_csv(input_file)

# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================
required_cols = [
    "Map_Cor",
    "Date_time_year",
    "Aedes_Prevalence_percent",
    "NS1_Antigen",
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
    "Previous_dengue_cases_last_month"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ============================================================
# 4. CLEAN DATA
# ============================================================
df["Date_time"] = pd.to_datetime(
    df["Date_time_year"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

bad_dt = df["Date_time"].isna()
if bad_dt.any():
    df.loc[bad_dt, "Date_time"] = pd.to_datetime(
        df.loc[bad_dt, "Date_time_year"],
        dayfirst=True,
        errors="coerce"
    )

coord_split = df["Map_Cor"].astype(str).str.split(",", n=1, expand=True)
if coord_split.shape[1] != 2:
    raise ValueError("Map_Cor must contain latitude and longitude separated by a comma.")

df["Latitude"] = pd.to_numeric(coord_split[0].str.strip(), errors="coerce")
df["Longitude"] = pd.to_numeric(coord_split[1].str.strip(), errors="coerce")

df["Aedes_Prevalence_percent"] = pd.to_numeric(df["Aedes_Prevalence_percent"], errors="coerce")
df["NS1_Antigen"] = df["NS1_Antigen"].astype(str).str.strip().str.lower()
df["NS1_binary"] = np.where(df["NS1_Antigen"] == "positive", 1, 0)

df["Previous_dengue_cases_last_month_bin"] = np.where(
    df["Previous_dengue_cases_last_month"].astype(str).str.strip().str.lower() == "yes",
    1, 0
)

numeric_cols = [
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
    "Spraying_effectiveness"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Time-derived variables
df["Year"] = df["Date_time"].dt.year
df["Month"] = df["Date_time"].dt.month
df["DayOfYear"] = df["Date_time"].dt.dayofyear

df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

min_date = df["Date_time"].min()
df["Days_since_start"] = (df["Date_time"] - min_date).dt.total_seconds() / (24 * 3600)

# ============================================================
# 5. ANALYSIS DATASET
# ============================================================
base_predictors = [
    "Latitude",
    "Longitude",
    "Days_since_start",
    "Month_sin",
    "Month_cos",
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
    "Previous_dengue_cases_last_month_bin"
]

analysis_df = df.dropna(
    subset=["Date_time", "Latitude", "Longitude", "Aedes_Prevalence_percent", "NS1_binary"] + base_predictors
).copy()

if analysis_df.empty:
    raise ValueError("No usable rows remained after cleaning.")

# ============================================================
# 6. LOCAL SPATIAL LAG FEATURES
#    k-nearest-neighbor averages
# ============================================================
coords = analysis_df[["Latitude", "Longitude"]].values
n_obs = len(analysis_df)
k_neighbors = min(6, max(2, n_obs - 1))

nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
nbrs.fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Exclude self at position 0
neighbor_indices = indices[:, 1:]

local_aedes = []
local_ns1 = []
local_prev_dengue = []

aedes_vals = analysis_df["Aedes_Prevalence_percent"].values
ns1_vals = analysis_df["NS1_binary"].values
prev_vals = analysis_df["Previous_dengue_cases_last_month_bin"].values

for i in range(n_obs):
    neigh = neighbor_indices[i]
    local_aedes.append(np.mean(aedes_vals[neigh]))
    local_ns1.append(np.mean(ns1_vals[neigh]))
    local_prev_dengue.append(np.mean(prev_vals[neigh]))

analysis_df["Local_Aedes_Lag"] = local_aedes
analysis_df["Local_NS1_Lag"] = local_ns1
analysis_df["Local_PrevDengue_Lag"] = local_prev_dengue

# ============================================================
# 7. GEOSPATIAL REGRESSION PREDICTORS
# ============================================================
predictor_cols = base_predictors + [
    "Local_Aedes_Lag",
    "Local_NS1_Lag",
    "Local_PrevDengue_Lag"
]

# ============================================================
# 8. STANDARDIZE PREDICTORS
# ============================================================
scaled_df = analysis_df.copy()
scale_info = {}

for col in predictor_cols:
    mu = scaled_df[col].mean()
    sd = scaled_df[col].std(ddof=0)
    scale_info[col] = {"mean": mu, "sd": sd}
    if pd.isna(sd) or sd == 0:
        scaled_df[col] = 0.0
    else:
        scaled_df[col] = (scaled_df[col] - mu) / sd

# ============================================================
# 9. MODELS
# ============================================================
# Aedes model: OLS
X_aedes = sm.add_constant(scaled_df[predictor_cols], has_constant="add")
y_aedes = scaled_df["Aedes_Prevalence_percent"]
aedes_model = sm.OLS(y_aedes, X_aedes).fit()

aedes_coef = pd.DataFrame({
    "Variable": aedes_model.params.index,
    "Coefficient": aedes_model.params.values,
    "Std_Error": aedes_model.bse.values,
    "t_value": aedes_model.tvalues.values,
    "p_value": aedes_model.pvalues.values,
    "CI_lower": aedes_model.conf_int()[0].values,
    "CI_upper": aedes_model.conf_int()[1].values
})
aedes_coef["Significant"] = np.where(aedes_coef["p_value"] < 0.05, "Yes", "No")
aedes_coef.to_csv(os.path.join(output_dir, "aedes_geospatial_model_coefficients.csv"), index=False)

scaled_df["Aedes_predicted"] = aedes_model.predict(X_aedes)
scaled_df["Aedes_residual"] = scaled_df["Aedes_Prevalence_percent"] - scaled_df["Aedes_predicted"]

# NS1 model: logistic regression
X_ns1 = sm.add_constant(scaled_df[predictor_cols], has_constant="add")
y_ns1 = scaled_df["NS1_binary"]
ns1_model = sm.Logit(y_ns1, X_ns1).fit(disp=False)

ns1_coef = pd.DataFrame({
    "Variable": ns1_model.params.index,
    "LogOdds_Coefficient": ns1_model.params.values,
    "Std_Error": ns1_model.bse.values,
    "z_value": ns1_model.tvalues.values,
    "p_value": ns1_model.pvalues.values,
    "CI_lower": ns1_model.conf_int()[0].values,
    "CI_upper": ns1_model.conf_int()[1].values
})
ns1_coef["Odds_Ratio"] = np.exp(ns1_coef["LogOdds_Coefficient"])
ns1_coef["OR_CI_lower"] = np.exp(ns1_coef["CI_lower"])
ns1_coef["OR_CI_upper"] = np.exp(ns1_coef["CI_upper"])
ns1_coef["Significant"] = np.where(ns1_coef["p_value"] < 0.05, "Yes", "No")
ns1_coef.to_csv(os.path.join(output_dir, "ns1_geospatial_model_coefficients.csv"), index=False)

scaled_df["NS1_pred_prob"] = ns1_model.predict(X_ns1)

# ============================================================
# 10. VIF
# ============================================================
vif_df = pd.DataFrame({
    "Variable": predictor_cols,
    "VIF": [variance_inflation_factor(scaled_df[predictor_cols].values, i)
            for i in range(len(predictor_cols))]
})
vif_df.to_csv(os.path.join(output_dir, "geospatial_predictor_vif.csv"), index=False)

# ============================================================
# 11. HELPER FUNCTIONS
# ============================================================
def pretty_name(v):
    mapping = {
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Days_since_start": "Days since start",
        "Month_sin": "Month sine",
        "Month_cos": "Month cosine",
        "Temperature_C": "Temperature (°C)",
        "Rainfall_mm": "Rainfall (mm)",
        "Humidity_pct": "Humidity (%)",
        "NDVI": "NDVI",
        "Soil_Moisture": "Soil moisture",
        "Distance_to_Water_km": "Distance to water (km)",
        "Population_density_per_km2": "Population density/km²",
        "House_density_per_hectare": "House density/hectare",
        "Socioeconomic_index": "Socioeconomic index",
        "Waste_management_score": "Waste management score",
        "Sanitation_coverage_pct": "Sanitation coverage (%)",
        "Vector_control_freq_per_month": "Vector control freq/month",
        "Spraying_effectiveness": "Spraying effectiveness",
        "Previous_dengue_cases_last_month_bin": "Previous dengue case",
        "Local_Aedes_Lag": "Local Aedes lag",
        "Local_NS1_Lag": "Local NS1 lag",
        "Local_PrevDengue_Lag": "Local previous dengue lag"
    }
    return mapping.get(v, v)

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def point_sizes(preval):
    return 25 + np.clip(preval, 0, None) * 8

def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"

# ============================================================
# 12. SIGNIFICANT / IMPORTANT VARIABLES
# ============================================================
sig_aedes = aedes_coef[(aedes_coef["Variable"] != "const") & (aedes_coef["p_value"] < 0.05)].copy()
sig_ns1 = ns1_coef[(ns1_coef["Variable"] != "const") & (ns1_coef["p_value"] < 0.05)].copy()

if len(sig_aedes) < 4:
    sig_aedes = aedes_coef[aedes_coef["Variable"] != "const"].copy()
    sig_aedes = sig_aedes.reindex(sig_aedes["Coefficient"].abs().sort_values(ascending=False).index).head(8)

if len(sig_ns1) < 4:
    sig_ns1 = ns1_coef[ns1_coef["Variable"] != "const"].copy()
    sig_ns1 = sig_ns1.reindex(sig_ns1["LogOdds_Coefficient"].abs().sort_values(ascending=False).index).head(8)

all_important_vars = sorted(set(sig_aedes["Variable"]).union(set(sig_ns1["Variable"])))
all_important_vars = [v for v in all_important_vars if v in predictor_cols]

# ============================================================
# 13. MONTHLY SUMMARY
# ============================================================
monthly = (
    analysis_df.groupby(pd.Grouper(key="Date_time", freq="MS"))
    .agg(
        Aedes_Prevalence=("Aedes_Prevalence_percent", "mean"),
        NS1_Prevalence=("NS1_binary", lambda x: x.mean() * 100),
        Pools_n=("NS1_binary", "count")
    )
    .reset_index()
    .dropna()
)
monthly.to_csv(os.path.join(output_dir, "monthly_aedes_ns1_summary.csv"), index=False)

# ============================================================
# 14. PLOT 1: SPATIAL HOTSPOT SURFACE FOR AEDES
# ============================================================
x = analysis_df["Longitude"].values
y = analysis_df["Latitude"].values
z = analysis_df["Aedes_Prevalence_percent"].values

grid_x, grid_y = np.mgrid[
    x.min():x.max():200j,
    y.min():y.max():200j
]

grid_z = griddata(
    points=np.column_stack((x, y)),
    values=z,
    xi=(grid_x, grid_y),
    method="linear"
)

grid_z_nearest = griddata(
    points=np.column_stack((x, y)),
    values=z,
    xi=(grid_x, grid_y),
    method="nearest"
)

grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

plt.figure(figsize=(10, 8))
im = plt.imshow(
    grid_z.T,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    aspect="auto",
    alpha=0.9
)

plt.scatter(
    analysis_df["Longitude"],
    analysis_df["Latitude"],
    s=18,
    color="white",
    edgecolors="black",
    linewidths=0.3,
    alpha=0.8
)

sub_ns1 = analysis_df[analysis_df["NS1_binary"] == 1]
sub_prev = analysis_df[analysis_df["Previous_dengue_cases_last_month_bin"] == 1]

plt.scatter(
    sub_ns1["Longitude"],
    sub_ns1["Latitude"],
    marker="^",
    s=80,
    facecolors="none",
    edgecolors="black",
    linewidths=1.2,
    label="NS1 positive"
)

plt.scatter(
    sub_prev["Longitude"],
    sub_prev["Latitude"],
    marker="*",
    s=140,
    color="red",
    edgecolors="black",
    linewidths=0.5,
    label="Previous dengue case"
)

plt.xlabel("Longitude", fontsize=16, fontweight="bold")
plt.ylabel("Latitude", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
clean_axes(plt.gca())

cbar = plt.colorbar(im)
cbar.set_label("Aedes prevalence (%)", fontsize=13, fontweight="bold")
cbar.ax.tick_params(labelsize=11)

plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot1_spatial_hotspot_aedes.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 15. PLOT 2: NS1 SPATIAL DISTRIBUTION
# ============================================================
plt.figure(figsize=(10, 8))
neg = analysis_df[analysis_df["NS1_binary"] == 0]
pos = analysis_df[analysis_df["NS1_binary"] == 1]

plt.scatter(
    neg["Longitude"],
    neg["Latitude"],
    s=40,
    alpha=0.65,
    label="NS1 negative"
)

plt.scatter(
    pos["Longitude"],
    pos["Latitude"],
    s=90,
    marker="^",
    alpha=0.9,
    label="NS1 positive"
)

plt.xlabel("Longitude", fontsize=16, fontweight="bold")
plt.ylabel("Latitude", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
clean_axes(plt.gca())
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot2_spatial_ns1_distribution.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 16. PLOT 3: LOCAL SPATIAL LAG VS OBSERVED AEDES
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(
    analysis_df["Local_Aedes_Lag"],
    analysis_df["Aedes_Prevalence_percent"],
    s=55,
    alpha=0.75
)

# regression line
xv = analysis_df["Local_Aedes_Lag"].values
yv = analysis_df["Aedes_Prevalence_percent"].values
m, b = np.polyfit(xv, yv, 1)
xx = np.linspace(xv.min(), xv.max(), 100)
yy = m * xx + b
plt.plot(xx, yy, linewidth=2)

r_lag, p_lag = pearsonr(xv, yv)

plt.xlabel("Local neighborhood mean Aedes prevalence", fontsize=16, fontweight="bold")
plt.ylabel("Observed Aedes prevalence (%)", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
clean_axes(plt.gca())
plt.text(
    0.03, 0.96,
    f"r = {r_lag:.3f}, p = {format_p(p_lag)}",
    transform=plt.gca().transAxes,
    fontsize=11,
    va="top"
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot3_local_lag_vs_observed_aedes.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 17. PLOT 4: SEMIVARIOGRAM-LIKE SPATIAL DEPENDENCE
# ============================================================
coords_mat = analysis_df[["Latitude", "Longitude"]].values
dists = pdist(coords_mat, metric="euclidean")
vals = analysis_df["Aedes_Prevalence_percent"].values.reshape(-1, 1)
semivars = 0.5 * pdist(vals, metric="sqeuclidean")

n_bins = min(15, max(6, len(dists) // 50))
bins = np.linspace(dists.min(), dists.max(), n_bins + 1)
bin_centers = []
bin_semivars = []

for i in range(n_bins):
    mask = (dists >= bins[i]) & (dists < bins[i + 1])
    if np.any(mask):
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_semivars.append(np.mean(semivars[mask]))

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, bin_semivars, marker="o", linewidth=2)
plt.xlabel("Pairwise spatial distance", fontsize=16, fontweight="bold")
plt.ylabel("Semivariance of Aedes prevalence", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
clean_axes(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot4_semivariogram_aedes.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 18. PLOT 5: MONTHLY TEMPORAL TRENDS
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(monthly["Date_time"], monthly["Aedes_Prevalence"], marker="o", linewidth=2.3, label="Aedes prevalence")
plt.plot(monthly["Date_time"], monthly["NS1_Prevalence"], marker="s", linewidth=2.3, label="NS1 positivity")
plt.xlabel("Month", fontsize=16, fontweight="bold")
plt.ylabel("Prevalence (%)", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
clean_axes(plt.gca())
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot5_monthly_trends.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 19. PLOT 6: AEDES COEFFICIENT PLOT
# ============================================================
plot_a = sig_aedes.copy().sort_values("Coefficient")

plt.figure(figsize=(9, 7))
plt.errorbar(
    x=plot_a["Coefficient"],
    y=np.arange(len(plot_a)),
    xerr=[
        plot_a["Coefficient"] - plot_a["CI_lower"],
        plot_a["CI_upper"] - plot_a["Coefficient"]
    ],
    fmt="o",
    capsize=4
)
plt.axvline(0, linestyle="--", linewidth=1)
plt.yticks(np.arange(len(plot_a)), [pretty_name(v) for v in plot_a["Variable"]], fontsize=11)
plt.xticks(fontsize=12)
plt.xlabel("Standardized coefficient", fontsize=16, fontweight="bold")
plt.ylabel("Predictor", fontsize=16, fontweight="bold")
clean_axes(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot6_aedes_coefficients.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 20. PLOT 7: NS1 ODDS RATIO PLOT
# ============================================================
plot_n = sig_ns1.copy().sort_values("Odds_Ratio")

plt.figure(figsize=(9, 7))
plt.errorbar(
    x=plot_n["Odds_Ratio"],
    y=np.arange(len(plot_n)),
    xerr=[
        plot_n["Odds_Ratio"] - plot_n["OR_CI_lower"],
        plot_n["OR_CI_upper"] - plot_n["Odds_Ratio"]
    ],
    fmt="o",
    capsize=4
)
plt.axvline(1, linestyle="--", linewidth=1)
plt.yticks(np.arange(len(plot_n)), [pretty_name(v) for v in plot_n["Variable"]], fontsize=11)
plt.xticks(fontsize=12)
plt.xlabel("Odds ratio", fontsize=16, fontweight="bold")
plt.ylabel("Predictor", fontsize=16, fontweight="bold")
clean_axes(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot7_ns1_odds_ratio.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 21. PLOT 8: CORRELATION HEATMAP
# ============================================================
corr_vars = list(set(all_important_vars + ["Aedes_Prevalence_percent", "NS1_binary"]))
corr_df = analysis_df[corr_vars].corr(numeric_only=True)
corr_df.to_csv(os.path.join(output_dir, "important_variable_correlation_matrix.csv"))

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_df.values, aspect="auto")
ax.set_xticks(np.arange(len(corr_df.columns)))
ax.set_yticks(np.arange(len(corr_df.index)))
ax.set_xticklabels([pretty_name(v) for v in corr_df.columns], rotation=90, fontsize=10)
ax.set_yticklabels([pretty_name(v) for v in corr_df.index], fontsize=10)
ax.set_xlabel("Variables", fontsize=16, fontweight="bold")
ax.set_ylabel("Variables", fontsize=16, fontweight="bold")
clean_axes(ax)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Correlation", fontsize=13, fontweight="bold")
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot8_correlation_heatmap.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 22. PLOT 9: OVERLAP BAR PLOT
#     Replaces unstable UpSet plotting
# ============================================================
aedes_set = set(sig_aedes["Variable"])
ns1_set = set(sig_ns1["Variable"])

overlap_rows = []
for var in predictor_cols:
    overlap_rows.append({
        "Variable": pretty_name(var),
        "Aedes": 1 if var in aedes_set else 0,
        "NS1": 1 if var in ns1_set else 0,
        "Group": (
            "Both" if (var in aedes_set and var in ns1_set)
            else "Aedes only" if var in aedes_set
            else "NS1 only" if var in ns1_set
            else "Neither"
        )
    })

overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(os.path.join(output_dir, "predictor_overlap_table.csv"), index=False)

group_counts = overlap_df["Group"].value_counts().reindex(
    ["Aedes only", "NS1 only", "Both", "Neither"],
    fill_value=0
)

plt.figure(figsize=(8, 6))
plt.bar(group_counts.index, group_counts.values)
plt.xlabel("Association category", fontsize=16, fontweight="bold")
plt.ylabel("Number of predictors", fontsize=16, fontweight="bold")
plt.xticks(rotation=20, fontsize=12)
plt.yticks(fontsize=12)
clean_axes(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot9_predictor_overlap_bar.png"), dpi=400, bbox_inches="tight")
plt.close()

# ============================================================
# 23. PLOT 10: NETWORK ANALYSIS
# ============================================================
network_file = None
network_edges_df = None

if HAS_NETWORKX:
    G = nx.Graph()

    G.add_node("Aedes prevalence", node_type="outcome")
    G.add_node("NS1 positivity", node_type="outcome")

    for _, row in sig_aedes.iterrows():
        var = row["Variable"]
        G.add_node(pretty_name(var), node_type="predictor")
        G.add_edge(
            pretty_name(var),
            "Aedes prevalence",
            weight=float(abs(row["Coefficient"])),
            p=float(row["p_value"])
        )

    for _, row in sig_ns1.iterrows():
        var = row["Variable"]
        G.add_node(pretty_name(var), node_type="predictor")
        G.add_edge(
            pretty_name(var),
            "NS1 positivity",
            weight=float(abs(row["LogOdds_Coefficient"])),
            p=float(row["p_value"])
        )

    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "Node_1": u,
            "Node_2": v,
            "Weight": d.get("weight", np.nan),
            "p_value": d.get("p", np.nan)
        })
    network_edges_df = pd.DataFrame(edge_rows)
    network_edges_df.to_csv(os.path.join(output_dir, "network_edges.csv"), index=False)

    plt.figure(figsize=(11, 9))
    pos = nx.spring_layout(G, seed=42, k=1.1)

    node_colors = []
    node_sizes = []
    for node, data in G.nodes(data=True):
        if data["node_type"] == "outcome":
            node_colors.append("lightcoral")
            node_sizes.append(2600)
        else:
            node_colors.append("lightblue")
            node_sizes.append(1500)

    widths = []
    for _, _, d in G.edges(data=True):
        widths.append(1.5 + 3.5 * d.get("weight", 0))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.75)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.axis("off")
    plt.tight_layout()
    network_file = "plot10_network_predictor_outcome.png"
    plt.savefig(os.path.join(output_dir, network_file), dpi=400, bbox_inches="tight")
    plt.close()

# ============================================================
# 24. PLOT 11+: 3D PLOTS FOR ALL IMPORTANT VARIABLES
#     x = Longitude, y = Latitude, z = variable
#     color = Aedes prevalence
#     red star = previous dengue
#     triangle = NS1 positive
# ============================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

three_d_files = []

for var in all_important_vars:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        analysis_df["Longitude"],
        analysis_df["Latitude"],
        analysis_df[var],
        c=analysis_df["Aedes_Prevalence_percent"],
        s=point_sizes(analysis_df["Aedes_Prevalence_percent"]),
        alpha=0.8
    )

    if not sub_prev.empty:
        ax.scatter(
            sub_prev["Longitude"],
            sub_prev["Latitude"],
            sub_prev[var],
            marker="*",
            s=170,
            color="red",
            edgecolors="black"
        )

    if not sub_ns1.empty:
        ax.scatter(
            sub_ns1["Longitude"],
            sub_ns1["Latitude"],
            sub_ns1[var],
            marker="^",
            s=90,
            facecolors="none",
            edgecolors="black"
        )

    ax.set_xlabel("Longitude", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Latitude", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_zlabel(pretty_name(var), fontsize=15, fontweight="bold", labelpad=10)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="z", labelsize=10)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Aedes prevalence (%)", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    legend_handles = []
    if not sub_prev.empty:
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='red', linestyle='None',
                       markersize=10, markeredgecolor='black', label='Previous dengue case')
        )
    if not sub_ns1.empty:
        legend_handles.append(
            plt.Line2D([0], [0], marker='^', color='black', linestyle='None',
                       markersize=9, markerfacecolor='none', label='NS1 positive')
        )
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left")

    plt.tight_layout()
    file_name = f"plot_3d_{var}.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=400, bbox_inches="tight")
    plt.close()
    three_d_files.append(file_name)

# ============================================================
# 25. GEOSPATIAL SUMMARY TABLES
# ============================================================
geo_summary = pd.DataFrame({
    "Metric": [
        "Usable observations",
        "Latitude min",
        "Latitude max",
        "Longitude min",
        "Longitude max",
        "Mean Aedes prevalence (%)",
        "NS1 positivity (%)",
        "Mean local Aedes lag",
        "Mean local NS1 lag"
    ],
    "Value": [
        len(analysis_df),
        round(analysis_df["Latitude"].min(), 6),
        round(analysis_df["Latitude"].max(), 6),
        round(analysis_df["Longitude"].min(), 6),
        round(analysis_df["Longitude"].max(), 6),
        round(analysis_df["Aedes_Prevalence_percent"].mean(), 4),
        round(analysis_df["NS1_binary"].mean() * 100, 4),
        round(analysis_df["Local_Aedes_Lag"].mean(), 4),
        round(analysis_df["Local_NS1_Lag"].mean(), 4),
    ]
})
geo_summary.to_csv(os.path.join(output_dir, "geospatial_summary.csv"), index=False)

# ============================================================
# 26. TEXT REPORT
# ============================================================
text_file = os.path.join(output_dir, "full_geospatial_analysis_report.txt")

with open(text_file, "w", encoding="utf-8") as f:
    f.write("FULL GEOSPATIAL + SPATIO-TEMPORAL DRIVER ANALYSIS\n")
    f.write("=" * 120 + "\n\n")

    f.write("INPUT FILE\n")
    f.write("-" * 120 + "\n")
    f.write(f"{input_file}\n\n")

    f.write("DATA SUMMARY\n")
    f.write("-" * 120 + "\n")
    f.write(f"Usable observations: {len(analysis_df)}\n")
    f.write(f"Study period start: {analysis_df['Date_time'].min()}\n")
    f.write(f"Study period end:   {analysis_df['Date_time'].max()}\n")
    f.write(f"Latitude range:  {analysis_df['Latitude'].min():.6f} to {analysis_df['Latitude'].max():.6f}\n")
    f.write(f"Longitude range: {analysis_df['Longitude'].min():.6f} to {analysis_df['Longitude'].max():.6f}\n")
    f.write(f"Mean Aedes prevalence (%): {analysis_df['Aedes_Prevalence_percent'].mean():.4f}\n")
    f.write(f"NS1 positivity (%): {(analysis_df['NS1_binary'].mean() * 100):.4f}\n\n")

    f.write("GEOSPATIAL REGRESSION TERMS\n")
    f.write("-" * 120 + "\n")
    f.write("Local_Aedes_Lag: mean Aedes prevalence among nearest neighboring pools\n")
    f.write("Local_NS1_Lag: mean NS1 positivity among nearest neighboring pools\n")
    f.write("Local_PrevDengue_Lag: mean previous dengue history among nearest neighboring pools\n\n")

    f.write("AEDES MODEL SUMMARY\n")
    f.write("-" * 120 + "\n")
    f.write(aedes_model.summary().as_text())
    f.write("\n\n")

    f.write("NS1 LOGISTIC MODEL SUMMARY\n")
    f.write("-" * 120 + "\n")
    f.write(ns1_model.summary().as_text())
    f.write("\n\n")

    f.write("SIGNIFICANT / IMPORTANT AEDES PREDICTORS\n")
    f.write("-" * 120 + "\n")
    f.write(sig_aedes.to_string(index=False))
    f.write("\n\n")

    f.write("SIGNIFICANT / IMPORTANT NS1 PREDICTORS\n")
    f.write("-" * 120 + "\n")
    f.write(sig_ns1.to_string(index=False))
    f.write("\n\n")

    f.write("ALL IMPORTANT VARIABLES USED IN 3D PLOTS\n")
    f.write("-" * 120 + "\n")
    for v in all_important_vars:
        f.write(f"{v} -> {pretty_name(v)}\n")
    f.write("\n")

    f.write("VIF TABLE\n")
    f.write("-" * 120 + "\n")
    f.write(vif_df.to_string(index=False))
    f.write("\n\n")

    f.write("SPATIAL AUTOCORRELATION-LIKE CHECK\n")
    f.write("-" * 120 + "\n")
    f.write(f"Observed vs local-lag Aedes correlation: r = {r_lag:.4f}, p = {format_p(p_lag)}\n\n")

    f.write("OVERLAP OF ASSOCIATED PREDICTORS\n")
    f.write("-" * 120 + "\n")
    f.write(overlap_df.to_string(index=False))
    f.write("\n\n")

    if network_edges_df is not None:
        f.write("NETWORK EDGES\n")
        f.write("-" * 120 + "\n")
        f.write(network_edges_df.to_string(index=False))
        f.write("\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 120 + "\n")
    output_files = [
        "aedes_geospatial_model_coefficients.csv",
        "ns1_geospatial_model_coefficients.csv",
        "geospatial_predictor_vif.csv",
        "monthly_aedes_ns1_summary.csv",
        "important_variable_correlation_matrix.csv",
        "predictor_overlap_table.csv",
        "geospatial_summary.csv",
        "plot1_spatial_hotspot_aedes.png",
        "plot2_spatial_ns1_distribution.png",
        "plot3_local_lag_vs_observed_aedes.png",
        "plot4_semivariogram_aedes.png",
        "plot5_monthly_trends.png",
        "plot6_aedes_coefficients.png",
        "plot7_ns1_odds_ratio.png",
        "plot8_correlation_heatmap.png",
        "plot9_predictor_overlap_bar.png",
        "full_geospatial_analysis_report.txt"
    ] + three_d_files

    if network_file:
        output_files.append(network_file)
        output_files.append("network_edges.csv")

    for item in output_files:
        f.write(item + "\n")

print("Full geospatial analysis completed successfully.")
print(f"Results saved in: {output_dir}")
print(f"Main report saved as: {text_file}")
print(f"3D plots generated: {len(three_d_files)}")