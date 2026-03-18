import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# SETTINGS
# ============================================================
DATA_DIR = r"C:\Users\nahid_vv0xche\Downloads\New folder (27)"
OUT_DIR = os.path.join(DATA_DIR, "html_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(DATA_DIR, "prrsv_sequences_metadata.csv")
MUTATION_FILE = os.path.join(DATA_DIR, "mutation_matrix.csv")

REAL_EVOLUTION_HTML = os.path.join(OUT_DIR, "real_evolution_remade.html")
FITNESS_LANDSCAPE_HTML = os.path.join(OUT_DIR, "fitness_landscape_model_dense.html")

GRID_SIZE = 180
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# LOAD DATA
# ============================================================
meta = pd.read_csv(METADATA_FILE)
mut = pd.read_csv(MUTATION_FILE)

df = meta.merge(mut, on="sample_id", how="inner")
df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

mutation_cols = [c for c in df.columns if c.startswith("POS_")]
if len(mutation_cols) == 0:
    raise ValueError("No mutation columns found. Expected columns starting with 'POS_'.")

for col in ["viral_load_log10", "immune_escape_score", "neutralization_drop", "clinical_score", "evolutionary_risk_score"]:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

if "lineage" not in df.columns:
    df["lineage"] = "Unknown"

# ============================================================
# EMBEDDING
# ============================================================
X = df[mutation_cols].fillna(0).astype(float).values
X_scaled = StandardScaler().fit_transform(X)
coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# ============================================================
# FITNESS
# ============================================================
df["mutation_burden"] = df[mutation_cols].fillna(0).sum(axis=1)

df["fitness_raw"] = (
    0.30 * df["viral_load_log10"]
    + 0.25 * df["immune_escape_score"]
    + 0.15 * df["neutralization_drop"]
    + 0.10 * df["evolutionary_risk_score"]
    - 0.08 * df["clinical_score"]
    + 0.04 * ((df["mutation_burden"] - df["mutation_burden"].mean()) / (df["mutation_burden"].std() + 1e-12))
)

fmin = df["fitness_raw"].min()
fmax = df["fitness_raw"].max()
df["fitness"] = (df["fitness_raw"] - fmin) / (fmax - fmin + 1e-12)

# ============================================================
# TEMPORAL TRAJECTORY FOR REAL EVOLUTION
# ============================================================
df["year_month"] = df["collection_date"].dt.to_period("M").astype(str)

traj = (
    df.groupby(["year_month", "lineage"], as_index=False)
      .agg(
          x=("x", "mean"),
          y=("y", "mean"),
          fitness=("fitness", "mean"),
          n=("sample_id", "count")
      )
      .sort_values(["year_month", "lineage"])
)

main_lineage = (
    df.groupby("lineage", as_index=False)["sample_id"]
      .count()
      .sort_values("sample_id", ascending=False)
      .iloc[0, 0]
)

traj_main = (
    traj[traj["lineage"] == main_lineage]
    .sort_values("year_month")
    .copy()
)

# ============================================================
# TOP STRAINS
# ============================================================
top_df = (
    df.sort_values(["fitness", "collection_date"], ascending=[False, True])
      .drop_duplicates(subset=["sample_id"])
      .head(18)
      .copy()
)

ancestor_like = (
    df.sort_values(["collection_date", "fitness"], ascending=[True, False])
      .drop_duplicates(subset=["sample_id"])
      .head(6)
      .copy()
)

# ============================================================
# DENSE MOUNTAIN LANDSCAPE
# ============================================================
x_grid = np.linspace(df["x"].min() - 0.8, df["x"].max() + 0.8, GRID_SIZE)
y_grid = np.linspace(df["y"].min() - 0.8, df["y"].max() + 0.8, GRID_SIZE)
Xg, Yg = np.meshgrid(x_grid, y_grid)

Z = np.zeros_like(Xg, dtype=float)

# major peaks from top strains
major_peaks = (
    df.sort_values("fitness", ascending=False)
      .drop_duplicates(subset=["sample_id"])
      .head(45)
      .copy()
)

sigma_major = 0.18
sigma_minor = 0.10
sigma_micro = 0.06

for _, row in major_peaks.iterrows():
    amp = 0.55 + 0.70 * row["fitness"]
    Z += amp * np.exp(
        -((Xg - row["x"]) ** 2 + (Yg - row["y"]) ** 2) / (2 * sigma_major ** 2)
    )

# many secondary peaks
secondary_pool = (
    df.sample(min(350, len(df)), random_state=RANDOM_STATE)
      .copy()
)

for _, row in secondary_pool.iterrows():
    amp = 0.05 + 0.14 * row["fitness"]
    Z += amp * np.exp(
        -((Xg - row["x"]) ** 2 + (Yg - row["y"]) ** 2) / (2 * sigma_minor ** 2)
    )

# micro ruggedness for richer topology
micro_pool = (
    df.sample(min(700, len(df)), random_state=RANDOM_STATE + 1)
      .copy()
)

for _, row in micro_pool.iterrows():
    amp = 0.015 + 0.035 * row["fitness"]
    Z += amp * np.exp(
        -((Xg - row["x"]) ** 2 + (Yg - row["y"]) ** 2) / (2 * sigma_micro ** 2)
    )

# wave-like ruggedness to increase dimensional richness
radial = np.sqrt((Xg - df["x"].mean()) ** 2 + (Yg - df["y"].mean()) ** 2)
Z += 0.06 * np.cos(4.2 * radial) * np.exp(-0.35 * radial)
Z += 0.03 * np.sin(2.4 * Xg) * np.cos(2.8 * Yg)

# broad basin shaping
Z -= 0.025 * ((Xg - df["x"].mean()) ** 2 + 0.8 * (Yg - df["y"].mean()) ** 2)

Z = Z - Z.min()
Z = Z / (Z.max() + 1e-12)

def nearest_surface_height(x, y):
    ix = np.argmin(np.abs(x_grid - x))
    iy = np.argmin(np.abs(y_grid - y))
    return float(Z[iy, ix])

top_df["surface_z"] = top_df.apply(lambda r: nearest_surface_height(r["x"], r["y"]) + 0.05, axis=1)

# ============================================================
# JSON SERIALIZATION HELPERS
# ============================================================
def to_list(series, digits=6):
    return np.round(series.astype(float).values, digits).tolist()

def to_str_list(series):
    return series.astype(str).tolist()

x_vals = to_list(df["x"])
y_vals = to_list(df["y"])
fit_vals = to_list(df["fitness"])

sample_ids = to_str_list(df["sample_id"])
lineages = to_str_list(df["lineage"])
dates = df["collection_date"].dt.strftime("%Y-%m-%d").fillna("NA").astype(str).tolist()
mutation_burden = to_list(df["mutation_burden"], digits=3)

traj_x = to_list(traj_main["x"])
traj_y = to_list(traj_main["y"])
traj_z = to_list(traj_main["fitness"])
traj_labels = to_str_list(traj_main["year_month"])

top_x = to_list(top_df["x"])
top_y = to_list(top_df["y"])
top_z = to_list(top_df["surface_z"])
top_names = (top_df["sample_id"].astype(str) + " | " + top_df["lineage"].astype(str)).tolist()

anc_x = to_list(ancestor_like["x"])
anc_y = to_list(ancestor_like["y"])
anc_z = to_list(ancestor_like["fitness"] + 0.03)
anc_names = (ancestor_like["sample_id"].astype(str) + " | " + ancestor_like["lineage"].astype(str)).tolist()

# lineage colors
unique_lineages = sorted(df["lineage"].astype(str).unique().tolist())
palette = [
    "#7fd8ff", "#ffb06a", "#8effa1", "#d8a6ff", "#ffd54a",
    "#ff7f9f", "#6ef0d1", "#c7c9ff", "#ffa94d", "#9effb8"
]
lineage_color_map = {lin: palette[i % len(palette)] for i, lin in enumerate(unique_lineages)}
point_colors = [lineage_color_map[v] for v in df["lineage"].astype(str).tolist()]

Xg_list = np.round(Xg, 6).tolist()
Yg_list = np.round(Yg, 6).tolist()
Z_list = np.round(Z, 6).tolist()

# ============================================================
# HTML: REAL EVOLUTION REMADE
# ============================================================
real_evolution_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Real Evolution Remade</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background:
        radial-gradient(1200px 600px at 10% 10%, #193456 0%, transparent 50%),
        radial-gradient(900px 500px at 90% 20%, #3a2146 0%, transparent 40%),
        linear-gradient(180deg, #08111f, #10233d);
      color: white;
    }}
    .wrap {{
      max-width: 1450px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 34px;
    }}
    p {{
      color: #c8d5e8;
      line-height: 1.6;
      max-width: 980px;
    }}
    #plot {{
      height: 840px;
      margin-top: 16px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Real Evolution</h1>
      <p>
        This remade interactive view emphasizes real viral evolution as continuous movement through genetic space rather than only branching. 
        Strains are colored by lineage, early ancestor-like strains are highlighted in cyan, top adaptive strains are highlighted in red, and the white path tracks temporal evolutionary movement of the dominant lineage.
      </p>
      <div id="plot"></div>
    </div>
  </div>

  <script>
    const xVals = {json.dumps(x_vals)};
    const yVals = {json.dumps(y_vals)};
    const zVals = {json.dumps(fit_vals)};
    const sampleIds = {json.dumps(sample_ids)};
    const lineages = {json.dumps(lineages)};
    const dates = {json.dumps(dates)};
    const burdens = {json.dumps(mutation_burden)};
    const pointColors = {json.dumps(point_colors)};

    const trajX = {json.dumps(traj_x)};
    const trajY = {json.dumps(traj_y)};
    const trajZ = {json.dumps(traj_z)};
    const trajLabels = {json.dumps(traj_labels)};

    const topX = {json.dumps(top_x)};
    const topY = {json.dumps(top_y)};
    const topZ = {json.dumps([v + 0.02 for v in top_df["fitness"].tolist()])};
    const topNames = {json.dumps(top_names)};

    const ancX = {json.dumps(anc_x)};
    const ancY = {json.dumps(anc_y)};
    const ancZ = {json.dumps(anc_z)};
    const ancNames = {json.dumps(anc_names)};

    const strainCloud = {{
      type: "scatter3d",
      mode: "markers",
      x: xVals,
      y: yVals,
      z: zVals,
      marker: {{
        size: 4,
        color: pointColors,
        opacity: 0.78,
        line: {{color: "rgba(10,16,30,0.7)", width: 0.4}}
      }},
      text: sampleIds.map((s, i) =>
        s + "<br>Lineage: " + lineages[i] + "<br>Date: " + dates[i] + "<br>Mutation burden: " + burdens[i]
      ),
      hovertemplate: "%{{text}}<br>Adaptive state: %{{z:.3f}}<extra></extra>",
      name: "Strains"
    }};

    const temporalTrajectory = {{
      type: "scatter3d",
      mode: "lines+markers+text",
      x: trajX,
      y: trajY,
      z: trajZ,
      line: {{
        color: "white",
        width: 9
      }},
      marker: {{
        size: 4,
        color: "white"
      }},
      text: trajLabels,
      textposition: "top center",
      hovertemplate: "Month: %{{text}}<br>Adaptive state: %{{z:.3f}}<extra></extra>",
      name: "Temporal trajectory"
    }};

    const evolvedStrains = {{
      type: "scatter3d",
      mode: "markers+text",
      x: topX,
      y: topY,
      z: topZ,
      marker: {{
        size: 8,
        color: "red",
        symbol: "diamond"
      }},
      text: topNames,
      textposition: "top center",
      hovertemplate: "%{{text}}<br>Adaptive state: %{{z:.3f}}<extra></extra>",
      name: "Top adaptive strains"
    }};

    const ancestorStrains = {{
      type: "scatter3d",
      mode: "markers+text",
      x: ancX,
      y: ancY,
      z: ancZ,
      marker: {{
        size: 8,
        color: "deepskyblue",
        symbol: "circle"
      }},
      text: ancNames,
      textposition: "top center",
      hovertemplate: "%{{text}}<br>Adaptive state: %{{z:.3f}}<extra></extra>",
      name: "Ancestor-like strains"
    }};

    const layout = {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      scene: {{
        xaxis: {{
          title: "Genetic dimension 1",
          showbackground: true,
          backgroundcolor: "rgba(240,245,250,0.60)",
          gridcolor: "rgba(255,255,255,0.18)"
        }},
        yaxis: {{
          title: "Genetic dimension 2",
          showbackground: true,
          backgroundcolor: "rgba(240,245,250,0.60)",
          gridcolor: "rgba(255,255,255,0.18)"
        }},
        zaxis: {{
          title: "Real evolutionary state",
          showbackground: true,
          backgroundcolor: "rgba(245,248,252,0.65)",
          gridcolor: "rgba(255,255,255,0.18)"
        }},
        camera: {{
          eye: {{x: 1.7, y: -1.9, z: 1.18}}
        }},
        aspectmode: "manual",
        aspectratio: {{x: 1.35, y: 1.18, z: 0.80}}
      }},
      margin: {{l: 0, r: 0, t: 10, b: 0}},
      legend: {{
        bgcolor: "rgba(255,255,255,0.08)",
        font: {{size: 12}}
      }}
    }};

    Plotly.newPlot(
      "plot",
      [strainCloud, temporalTrajectory, ancestorStrains, evolvedStrains],
      layout,
      {{responsive: true, displaylogo: false}}
    );
  </script>
</body>
</html>
"""

with open(REAL_EVOLUTION_HTML, "w", encoding="utf-8") as f:
    f.write(real_evolution_html)

# ============================================================
# HTML: DENSE FITNESS LANDSCAPE
# ============================================================
fitness_landscape_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Fitness Landscape Model</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background:
        radial-gradient(1100px 650px at 15% 10%, #18385f 0%, transparent 48%),
        radial-gradient(900px 540px at 85% 15%, #4a2640 0%, transparent 40%),
        linear-gradient(180deg, #07101b, #13243e);
      color: white;
    }}
    .wrap {{
      max-width: 1450px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 34px;
    }}
    p {{
      color: #c8d5e8;
      line-height: 1.6;
      max-width: 980px;
    }}
    #plot {{
      height: 860px;
      margin-top: 16px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Fitness Landscape Model</h1>
      <p>
        This dense, mountain-like 3D fitness landscape is intentionally made with many more peaks and local ridges, representing a richer high-dimensional adaptive surface. 
        The labeled strains indicate important summit-associated variants near major peaks.
      </p>
      <div id="plot"></div>
    </div>
  </div>

  <script>
    const Xg = {json.dumps(Xg_list)};
    const Yg = {json.dumps(Yg_list)};
    const Zg = {json.dumps(Z_list)};

    const topX = {json.dumps(top_x)};
    const topY = {json.dumps(top_y)};
    const topZ = {json.dumps(top_z)};
    const topNames = {json.dumps(top_names)};

    const surface = {{
      type: "surface",
      x: Xg[0],
      y: Yg.map(r => r[0]),
      z: Zg,
      colorscale: "Turbo",
      opacity: 0.985,
      showscale: true,
      colorbar: {{ title: "Relative fitness" }},
      lighting: {{
        ambient: 0.42,
        diffuse: 0.96,
        fresnel: 0.14,
        roughness: 0.25,
        specular: 0.48
      }},
      lightposition: {{x: 220, y: 130, z: 300}},
      contours: {{
        z: {{
          show: false
        }}
      }},
      hovertemplate: "Genetic dim 1: %{{x:.2f}}<br>Genetic dim 2: %{{y:.2f}}<br>Fitness: %{{z:.2f}}<extra></extra>",
      name: "Landscape"
    }};

    const summitStrains = {{
      type: "scatter3d",
      mode: "markers+text",
      x: topX,
      y: topY,
      z: topZ,
      marker: {{
        size: 8,
        color: "red",
        symbol: "diamond"
      }},
      text: topNames,
      textposition: "top center",
      hovertemplate: "%{{text}}<br>Surface height: %{{z:.3f}}<extra></extra>",
      name: "Peak strains"
    }};

    const layout = {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      scene: {{
        xaxis: {{
          title: "Genetic dimension 1",
          showbackground: true,
          backgroundcolor: "rgba(240,244,250,0.62)",
          gridcolor: "rgba(255,255,255,0.16)"
        }},
        yaxis: {{
          title: "Genetic dimension 2",
          showbackground: true,
          backgroundcolor: "rgba(240,244,250,0.62)",
          gridcolor: "rgba(255,255,255,0.16)"
        }},
        zaxis: {{
          title: "Fitness",
          showbackground: true,
          backgroundcolor: "rgba(245,248,252,0.68)",
          gridcolor: "rgba(255,255,255,0.16)"
        }},
        camera: {{
          eye: {{x: 1.65, y: -1.95, z: 1.18}}
        }},
        aspectmode: "manual",
        aspectratio: {{x: 1.38, y: 1.18, z: 0.88}}
      }},
      margin: {{l: 0, r: 0, t: 10, b: 0}},
      legend: {{
        bgcolor: "rgba(255,255,255,0.08)"
      }}
    }};

    Plotly.newPlot(
      "plot",
      [surface, summitStrains],
      layout,
      {{responsive: true, displaylogo: false}}
    );
  </script>
</body>
</html>
"""

with open(FITNESS_LANDSCAPE_HTML, "w", encoding="utf-8") as f:
    f.write(fitness_landscape_html)

print("Done.")
print("Saved files:")
print(REAL_EVOLUTION_HTML)
print(FITNESS_LANDSCAPE_HTML)