import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

input_file = r"D:\Redundency\redundancy-B.xlsx"
output_dir = r"D:\Redundency\rda_final B"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(input_file)

site_col = "Site"

water_quality_cols = [
    "Temperature (°C)",
    "pH",
    "TDS (ppm)",
    "Salinity (ppm)",
    "EC (ms)"
]

response_groups = {
    "1_Water_quality_with_abundance_and_size": [
        "Abundance", "<0.5", "0.5-1", "1-5"
    ],
    "2_Water_quality_with_colour": [
        "Black", "Red", "Green", "Blue", "Transparent"
    ],
    "3_Water_quality_with_shape": [
        "Filament", "Round", "Angular", "Irrigular"
    ],
    "4_Water_quality_with_morphotypes": [
        "Fiber", "Fragment", "Particle"
    ]
}

required_cols = [site_col] + water_quality_cols
for vals in response_groups.values():
    required_cols.extend(vals)

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in Excel file: {missing_cols}")

label_offsets = {
    "1_Water_quality_with_abundance_and_size": {
        "response": {
            "Abundance": (42, 8),
            "<0.5": (10, 24),
            "0.5-1": (36, -14),
            "1-5": (36, -2),
        },
        "env": {
            "Temperature (°C)": (18, 18),
            "pH": (12, 10),
            "TDS (ppm)": (4, -24),
            "Salinity (ppm)": (22, 12),
            "EC (ms)": (-8, 24),
        },
        "site": {
            "S1": (12, 10),
            "S2": (-14, -10),
            "S3": (-14, 10),
            "S4": (-12, 8),
            "S5": (12, -10),
            "S6": (12, 10),
            "S7": (-14, -10),
            "S8": (-14, 10),
            "S9": (-12, 8),
            "S10": (12, -10),
        },
    },
    "2_Water_quality_with_colour": {
        "response": {
            "Blue": (18, 30),
            "Green": (40, 22),
            "Red": (40, -6),
            "Black": (40, -24),
            "Transparent": (38, -36),
        },
        "env": {
            "TDS (ppm)": (12, 34),
            "pH": (20, 8),
            "Temperature (°C)": (24, -22),
            "Salinity (ppm)": (22, -30),
            "EC (ms)": (10, -40),
        },
        "site": {
            "S1": (14, -10),
            "S2": (14, 10),
            "S3": (-14, -8),
            "S4": (-14, 10),
            "S5": (14, -10),
            "S6": (14, -10),
            "S7": (14, 10),
            "S8": (-14, -8),
            "S9": (-14, 10),
            "S10": (14, -10),
        },
    },
    "3_Water_quality_with_shape": {
        "response": {
            "Filament": (-12, 28),
            "Round": (34, 20),
            "Angular": (18, 26),
            "Irrigular": (46, -2),
        },
        "env": {
            "TDS (ppm)": (-28, 16),
            "pH": (-10, 10),
            "Temperature (°C)": (18, 14),
            "Salinity (ppm)": (24, 18),
            "EC (ms)": (38, 8),
        },
        "site": {
            "S1": (14, 10),
            "S2": (-12, -10),
            "S3": (14, -12),
            "S4": (-12, -12),
            "S5": (-12, 10),
            "S6": (14, 10),
            "S7": (-12, -10),
            "S8": (14, -12),
            "S9": (-12, -12),
            "S10": (-12, 10),
        },
    },
    "4_Water_quality_with_morphotypes": {
        "response": {
            "Fiber": (-10, 30),
            "Fragment": (44, 0),
            "Particle": (40, 20),
        },
        "env": {
            "TDS (ppm)": (-28, 14),
            "pH": (8, 24),
            "Temperature (°C)": (18, 18),
            "Salinity (ppm)": (24, 18),
            "EC (ms)": (38, 8),
        },
        "site": {
            "S1": (14, 10),
            "S2": (-12, -10),
            "S3": (14, -12),
            "S4": (-12, -12),
            "S5": (-12, 10),
            "S6": (14, 10),
            "S7": (-12, -10),
            "S8": (14, -12),
            "S9": (-12, -12),
            "S10": (-12, 10),
        },
    },
}

response_colors = {
    "Abundance": "#1b9e77",
    "<0.5": "#66a61e",
    "0.5-1": "#7570b3",
    "1-5": "#e7298a",
    "Black": "#333333",
    "Red": "#d73027",
    "Green": "#1a9850",
    "Blue": "#4575b4",
    "Transparent": "#8c8c8c",
    "Filament": "#1f78b4",
    "Round": "#ffb000",
    "Angular": "#8e44ad",
    "Irrigular": "#e67e22",
    "Fiber": "#00a087",
    "Fragment": "#b2182b",
    "Particle": "#2166ac"
}

env_colors = {
    "Temperature (°C)": "#2c7fb8",
    "pH": "#7fcdbb",
    "TDS (ppm)": "#f03b20",
    "Salinity (ppm)": "#feb24c",
    "EC (ms)": "#756bb1"
}

site_palette = [
    "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
    "#a6cee3", "#fb9a99", "#b2df8a", "#cab2d6", "#fdbf6f"
]

def run_rda(data, x_cols, y_cols, site_col="Site"):
    sub = data[[site_col] + x_cols + y_cols].copy()

    for col in x_cols + y_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub = sub.dropna(subset=x_cols + y_cols).reset_index(drop=True)

    if sub.shape[0] < 3:
        raise ValueError(f"Not enough complete rows for RDA: {y_cols}")

    site_names = sub[site_col].astype(str).tolist()
    X = sub[x_cols].values
    Y = sub[y_cols].values

    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)

    X_design = np.column_stack([np.ones(Xs.shape[0]), Xs])
    B, _, _, _ = np.linalg.lstsq(X_design, Ys, rcond=None)
    Y_hat = X_design @ B

    ncomp = min(2, Y_hat.shape[0], Y_hat.shape[1])
    pca = PCA(n_components=ncomp)
    site_scores = pca.fit_transform(Y_hat)
    explained = pca.explained_variance_ratio_ * 100

    species_scores = pca.components_.T * np.sqrt(pca.explained_variance_[:ncomp])

    env_scores = np.zeros((len(x_cols), ncomp))
    for i in range(len(x_cols)):
        for j in range(ncomp):
            if np.std(site_scores[:, j]) == 0:
                env_scores[i, j] = 0
            else:
                env_scores[i, j] = np.corrcoef(Xs[:, i], site_scores[:, j])[0, 1]

    total_variance = np.var(Ys, axis=0, ddof=1).sum()
    constrained_variance = np.var(Y_hat, axis=0, ddof=1).sum()
    r2 = constrained_variance / total_variance if total_variance != 0 else np.nan

    return {
        "site_names": site_names,
        "site_scores": site_scores,
        "species_scores": species_scores,
        "env_scores": env_scores,
        "explained": explained,
        "r2": r2,
        "x_cols": x_cols,
        "y_cols": y_cols
    }

def save_scores_to_excel(result, out_xlsx):
    site_scores = result["site_scores"]
    species_scores = result["species_scores"]
    env_scores = result["env_scores"]
    site_names = result["site_names"]
    y_cols = result["y_cols"]
    x_cols = result["x_cols"]

    if site_scores.shape[1] == 1:
        site_df = pd.DataFrame({"Site": site_names, "RDA1": site_scores[:, 0]})
        response_df = pd.DataFrame({"Response_variable": y_cols, "RDA1": species_scores[:, 0]})
        env_df = pd.DataFrame({"Water_quality_variable": x_cols, "RDA1": env_scores[:, 0]})
        summary_df = pd.DataFrame({
            "Metric": ["RDA1_explained_percent", "RDA2_explained_percent", "Constrained_variance_R2"],
            "Value": [result["explained"][0], 0, result["r2"]]
        })
    else:
        site_df = pd.DataFrame({"Site": site_names, "RDA1": site_scores[:, 0], "RDA2": site_scores[:, 1]})
        response_df = pd.DataFrame({"Response_variable": y_cols, "RDA1": species_scores[:, 0], "RDA2": species_scores[:, 1]})
        env_df = pd.DataFrame({"Water_quality_variable": x_cols, "RDA1": env_scores[:, 0], "RDA2": env_scores[:, 1]})
        summary_df = pd.DataFrame({
            "Metric": ["RDA1_explained_percent", "RDA2_explained_percent", "Constrained_variance_R2"],
            "Value": [result["explained"][0], result["explained"][1], result["r2"]]
        })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        site_df.to_excel(writer, sheet_name="Site_scores", index=False)
        response_df.to_excel(writer, sheet_name="Response_scores", index=False)
        env_df.to_excel(writer, sheet_name="Water_quality_scores", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

def annotate_point(ax, x, y, text, offset=(10, 10), fontsize=11, fontweight="normal", color="black"):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops=dict(
            arrowstyle="-",
            color="#999999",
            lw=0.6,
            shrinkA=0,
            shrinkB=0
        ),
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="white",
            edgecolor="none",
            alpha=0.82
        )
    )

def plot_rda_standard(result, analysis_name, out_png, out_tiff, out_pdf):
    site_scores = result["site_scores"]
    species_scores = result["species_scores"]
    env_scores = result["env_scores"]
    explained = result["explained"]
    site_names = result["site_names"]
    x_cols = result["x_cols"]
    y_cols = result["y_cols"]

    if site_scores.shape[1] == 1:
        x_site = site_scores[:, 0]
        y_site = np.zeros_like(x_site)
        x_species = species_scores[:, 0]
        y_species = np.zeros_like(x_species)
        x_env = env_scores[:, 0]
        y_env = np.zeros_like(x_env)
        xlab = f"RDA1 ({explained[0]:.2f}%)"
        ylab = "RDA2 (0.00%)"
    else:
        x_site = site_scores[:, 0]
        y_site = site_scores[:, 1]
        x_species = species_scores[:, 0]
        y_species = species_scores[:, 1]
        x_env = env_scores[:, 0]
        y_env = env_scores[:, 1]
        xlab = f"RDA1 ({explained[0]:.2f}%)"
        ylab = f"RDA2 ({explained[1]:.2f}%)"

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.axhline(0, color="#b0b0b0", lw=1.0, zorder=1)
    ax.axvline(0, color="#b0b0b0", lw=1.0, zorder=1)

    max_site = max(
        np.max(np.abs(x_site)) if len(x_site) else 1,
        np.max(np.abs(y_site)) if len(y_site) else 1,
        1
    )
    max_species = max(
        np.max(np.abs(x_species)) if len(x_species) else 1e-6,
        np.max(np.abs(y_species)) if len(y_species) else 1e-6
    )
    max_env = max(
        np.max(np.abs(x_env)) if len(x_env) else 1e-6,
        np.max(np.abs(y_env)) if len(y_env) else 1e-6
    )

    species_scale = (max_site * 0.92) / max_species
    env_scale = (max_site * 0.78) / max_env

    site_color_map = {s: site_palette[i % len(site_palette)] for i, s in enumerate(site_names)}

    ax.scatter(
        x_site, y_site,
        s=180,
        color=[site_color_map[s] for s in site_names],
        edgecolor="white",
        linewidth=1.3,
        zorder=4
    )

    response_coords = {}
    for i, var in enumerate(y_cols):
        xx = x_species[i] * species_scale
        yy = y_species[i] * species_scale
        response_coords[var] = (xx, yy)

        ax.arrow(
            0, 0, xx, yy,
            color=response_colors.get(var, "#333333"),
            linewidth=2.0,
            head_width=max_site * 0.045,
            head_length=max_site * 0.07,
            length_includes_head=True,
            zorder=2
        )

    env_coords = {}
    for i, var in enumerate(x_cols):
        xx = x_env[i] * env_scale
        yy = y_env[i] * env_scale
        env_coords[var] = (xx, yy)

        ax.arrow(
            0, 0, xx, yy,
            color=env_colors.get(var, "#555555"),
            linewidth=1.6,
            linestyle="--",
            head_width=max_site * 0.028,
            head_length=max_site * 0.045,
            length_includes_head=True,
            zorder=2
        )

    all_x = list(x_site) + [v[0] for v in response_coords.values()] + [v[0] for v in env_coords.values()]
    all_y = list(y_site) + [v[1] for v in response_coords.values()] + [v[1] for v in env_coords.values()]

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    xr = xmax - xmin if xmax != xmin else 1
    yr = ymax - ymin if ymax != ymin else 1

    ax.set_xlim(xmin - xr * 0.35, xmax + xr * 0.45)
    ax.set_ylim(ymin - yr * 0.35, ymax + yr * 0.45)

    cfg = label_offsets[analysis_name]

    for s, (xs, ys) in zip(site_names, zip(x_site, y_site)):
        annotate_point(
            ax=ax,
            x=xs,
            y=ys,
            text=s,
            offset=cfg["site"].get(s, (10, 10)),
            fontsize=12,
            fontweight="bold",
            color=site_color_map[s]
        )

    for var, (xx, yy) in response_coords.items():
        annotate_point(
            ax=ax,
            x=xx,
            y=yy,
            text=var,
            offset=cfg["response"].get(var, (12, 12)),
            fontsize=12,
            fontweight="bold",
            color=response_colors.get(var, "#333333")
        )

    for var, (xx, yy) in env_coords.items():
        annotate_point(
            ax=ax,
            x=xx,
            y=yy,
            text=var,
            offset=cfg["env"].get(var, (12, 12)),
            fontsize=11,
            fontweight="normal",
            color=env_colors.get(var, "#555555")
        )

    ax.set_xlabel(xlab, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14, width=1.1)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    if not np.isnan(result["r2"]):
        ax.text(
            0.02, 0.98,
            f"Constrained variance (R²) = {result['r2']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="black",
                alpha=0.92
            )
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=800, bbox_inches="tight")
    plt.savefig(out_tiff, dpi=800, bbox_inches="tight", format="tiff")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

summary_rows = []

for analysis_name, y_cols in response_groups.items():
    print(f"Running: {analysis_name}")

    result = run_rda(
        data=df,
        x_cols=water_quality_cols,
        y_cols=y_cols,
        site_col=site_col
    )

    png_file = os.path.join(output_dir, f"{analysis_name}_RDA_plot.png")
    tiff_file = os.path.join(output_dir, f"{analysis_name}_RDA_plot.tiff")
    pdf_file = os.path.join(output_dir, f"{analysis_name}_RDA_plot.pdf")
    excel_file = os.path.join(output_dir, f"{analysis_name}_RDA_scores.xlsx")

    plot_rda_standard(result, analysis_name, png_file, tiff_file, pdf_file)
    save_scores_to_excel(result, excel_file)

    summary_rows.append({
        "Analysis": analysis_name,
        "RDA1_explained_percent": result["explained"][0] if len(result["explained"]) > 0 else np.nan,
        "RDA2_explained_percent": result["explained"][1] if len(result["explained"]) > 1 else 0,
        "Constrained_variance_R2": result["r2"]
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(
    os.path.join(output_dir, "RDA_summary_all_analyses.xlsx"),
    index=False
)

print("\nAll 4 RDA analyses completed successfully.")
print(f"Outputs saved in: {output_dir}")